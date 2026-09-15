# IoT Graduate — Distributed Edge Traffic Monitoring

Experimental distributed traffic-monitoring system for NVIDIA Jetson edge devices.
Detects vehicles, tracks them, estimates speed via homography, reads license plates,
records violations, and redistributes camera workloads between Jetson nodes under load.

> **Status:** research prototype. Designed for a trusted LAN; not hardened for open networks.

---

## Architecture — Three Parts

```
Camera/              Edge/ (Jetson A / B / C)          Server/
MediaMTX RTSP   →    ONE process per Jetson:        →  MediaMTX relay (:8554, RTSP push)
Docker sim           DeepStream pipeline               Zenoh router :7447 (optional relay)
                     + HealthAgent  (in-process)       aiohttp dashboard :9090
                     + PeerOrchestrator                   WebSocket to browsers
                     + Zenoh command/offload subs         Violation store (JSON)
```

**Camera** — Docker Compose runs MediaMTX (`:8554`) + FFmpeg containers that loop MP4
files as RTSP streams simulating live cameras.

**Edge** — Each Jetson runs a **single Python process** (`run_edge.sh` → `main.py`
→ `run_python_mode()`). Inside that one process:

1. The DeepStream pipeline (below).
2. `HealthAgent` — collects Jetson metrics (jtop), computes `load_score`,
   publishes the **sole** heartbeat on Zenoh (1 s cadence, hard-enforced),
   forwards overspeed events to Server.
 3. `PeerOrchestrator` — local load decisions (L0/L1/L2 offload), RFO voting,
    failover rescue, reclaim.
4. `ZenohCommandSubscriber` / `OffloadPublisher` / `OffloadReceiver` — camera
   ADD/REMOVE commands and crop-offload data plane.

All components share **one Zenoh session**. A single heartbeat publisher exists per
node; duplicate publishers were removed deliberately.

**Server** — aiohttp app (`:9090`): WebSocket hub receiving health + overspeed payloads
from all Jetson nodes; serves a browser dashboard; stores violation records; optionally
runs a Zenoh router (`ZENOH_ROUTER=tcp/<server>:7447`) so cross-subnet nodes converge
without relying on multicast alone.

---

## Physical Cluster Specifications (Hardware Setup)

The system is deployed and benchmarked on a fully homogeneous 3-node physical Jetson AGX Orin cluster connected via dedicated Gigabit LAN:

| Node ID | Physical Hardware | IP Address | Fixed Camera Ownership | Workload Profile |
|---|---|---|---|---|
| `jetson_A` | **NVIDIA Jetson AGX Orin DevKit (32GB)** | `192.168.212.20` | `cam_01`, `cam_02` | Sparse traffic (`video_in_N.mp4`, `video_out_S.mp4`) |
| `jetson_B` | **NVIDIA Jetson AGX Orin DevKit (32GB)** | `192.168.212.21` | `cam_03`, `cam_04` | Standard traffic (`video_in_N.mp4`, `video_out_S.mp4`) |
| `jetson_C` | **NVIDIA Jetson AGX Orin DevKit (32GB)** | `192.168.212.22` | `cam_05`, `cam_06` | Moderate traffic (`video_in_N.mp4`, `video_out_S.mp4`) |
| `Server` | **x86_64 Edge Server (Ubuntu 22.04)** | `116.118.9.125` | Aggregator / Dashboard | MediaMTX (:8554), Web (:9090), Zenoh (:7447) |

### Jetson AGX Orin Node Specifications (Homogeneous across all 3 nodes):
- **SoC**: NVIDIA Jetson AGX Orin Developer Kit
- **CPU**: 12-core ARM Cortex-A78AE v8.2 64-bit @ 2.2 GHz
- **GPU**: NVIDIA Ampere architecture with 2048 CUDA cores and 64 Tensor cores (up to 275 TOPS)
- **Memory**: 32 GB 256-bit LPDDR5 (29 GiB usable), 204.8 GB/s memory bandwidth
- **Storage**: 64 GB eMMC 5.1 (`/dev/mmcblk0p1` 57.8 GB rootfs `/`) + 14 GB swap on zram (12 × 1.2 GB partitions)
- **OS / L4T**: Ubuntu 22.04 LTS, JetPack 6.x (L4T R36.4.7 / R36.5.2), Linux 5.15 aarch64
- **Networking**: Gigabit Ethernet (1 Gbps LAN via unmanaged switch)

---

## Jetson DeepStream Pipeline

```
N × uridecodebin
      │
  nvstreammux  (1920×1080 mux, batch-size = n_cameras)
      │
   PGIE  (YOLO11 — vehicle detection, interval=3)
      │
  NvDCF tracker
      │
  SGIE  (LPD — license plate detection, single secondary classifier)
      │
 nvdsanalytics  (ROI polygon, line crossing)
      │
  SpeedProbe (GLib buffer probe, C-accelerated hot path)
      │
  display EGL / file sink / rtsp_push sink → MediaMTX relay on Server
```

License-plate **recognition (LPR)** runs **off-pipeline** — not in the DeepStream
graph. The graph detects plates via the single SGIE (LPD only); sgie2 was removed
in Phase 1. Recognized plate crops are decoded either by the local `LocalLprWorker`
pool (default, L0) or, under L2 plate-crop offload, shipped to a peer's
`OffloadReceiver` for LPR (`offload/plates/{src}/{dst}`). Per-camera homography
maps pixel displacement to world coordinates (meters);
median-filtered speed estimate triggers an overspeed event when `speed_kmh > SPEED_LIMIT_KMH`.

**Permanent mux pads** (`SPEEDFLOW_SLOT_CAPACITY=16`): all mux sink pads are created
once at build time and never released — dynamic ADD/REMOVE only swaps upstream branches.
Releasing live request pads corrupts `NvBufSurfacePool` and deadlocks the Tegra kernel.

**NVDEC session gate**: `dynamic_add_stream()` refuses ADD when the live nvv4l2decoder
count reaches `SPEEDFLOW_NVDEC_SESSION_LIMIT` (default 8; Jetson .env typically 14 —
conservative margin under the hardware ceiling ~16–32). Exceeding the ceiling yields
unrecoverable `OutputBufferUnavailable` (reboot required).

The SpeedProbe writes a unified JSON snapshot to `/dev/shm/speedflow_fps.json` every 1 s
(atomic tmp+rename+fsync): session id, sequence, per-camera FPS (bounded by configured
camera FPS), per-camera workload features (`n_track`, `n_plate`, `stationary_fraction`),
and offload counters (sender/receiver gates, queue depth, session drops, backpressure
drops). `HealthAgent` reads it each cycle with sequence + session-id + staleness guards
(3× interval) and rejects stale or non-advancing payloads. Both sides run on the same
1 s cadence (`HEALTH_INTERVAL=1.0`/`TELEMETRY_INTERVAL=1.0`, RuntimeError otherwise).

Note: per-camera FPS after `nvstreammux` does not exist — GPU saturation throttles all
streams together. Workload counts (`n_track`, `n_plate`) are the per-camera signal.

---

## Load Score — Asymptotic Multi-Dimensional Kernel

`load_score` is a continuous pressure scalar in `[0.0, 100.0)` computed via an asymptotic kernel:

$$S_{\text{load}} = 100.0 \times \frac{\rho}{1.0 + \rho}$$

where $\rho = \rho_s + \rho_d + \rho_r + \rho_v$ aggregates 4 orthogonal load dimensions:

1. **$\rho_s$ — Stream Concurrency**: Non-linear convex knee on the Jetson EMC LPDDR5 memory bus:
   $$\rho_s = \left(\frac{\max(0.0, n_{\text{active}} - 1.0)}{k_s}\right)^{p_s}$$
   (default $k_s = 2.5$, $p_s = 2.2$).
2. **$\rho_d$ — Workload Demand**: Vehicle workload ratio vs capacity threshold:
   $$\rho_d = \frac{\max(0.0, w_{\text{eff}})}{w_{\text{high}}}$$
   (default $w_{\text{high}} = 10.0$, where $w_{\text{eff}}$ is the effective vehicle count `eff_wl`).
3. **$\rho_r$ — Hardware Resource Contention**: CPU and RAM saturation (GPU% is excluded from $\rho$ due to DVFS noise):
   $$\rho_r = \begin{cases} 0.0 & \text{if } u \le u_{\text{safe}} \\ \frac{u - u_{\text{safe}}}{\max(0.05, 1.05 - u)} & \text{if } u > u_{\text{safe}} \end{cases}$$
   where $u = \max(\text{CPU}\%, \text{RAM}\%) / 100.0$ (default $u_{\text{safe}} = 0.60$).
4. **$\rho_v$ — Service Completion Deficit**: Bounded completion deficit vs floor (penalizes missed plates):
   $$\rho_v = \rho_{v,\max} \times \max\left(0.0, \min\left(1.0, \frac{s_{\text{target}} - \text{svc}}{s_{\text{target}} - s_{\text{floor}}}\right)\right)$$
   (default $s_{\text{target}} = 0.95$, $s_{\text{floor}} = 0.50$, $\rho_{v,\max} = 1.0$).

### Role of FPS (Safety Witness & Emergency Fuse Floor)

Per ADR-0001, FPS is a safety witness, not the primary driver of load calculation. FPS acts as an emergency fuse floor:
- If `eff_fps < fps_emergency` (default 12.0), score is floored at `hw_fuse_score_floor` (75.0).
- If GPU $\ge 99\%$ and `eff_fps < 15.0`, score is floored at `hw_fuse_score_floor` (75.0).

Every health cycle publishes `load_score_breakdown` containing `rho_s`, `rho_d`, `rho_r`, `rho_v`, and total `rho`.

### Workload-primary QoS mapping

When `workload_policy.enabled: true` (default), HealthAgent smooths telemetry with
EMA(α=0.33) and uses workload-primary bands: w_low=6 / w_high=10,
fps_confirm=22, fps_critical=15. QoS states (healthy/moderate/degraded/overloaded)
are published in the heartbeat; orchestration honours recognized states and falls back
to load-score thresholds otherwise.

---

## Multi-Level Offload Policy

Each Jetson runs an independent `PeerOrchestrator`; no master node. Decisions are
strictly local; Zenoh is only the message layer. Canonical tiers:

- **L0 — local full DeepStream analytics** (default). Every static-owned camera runs
  the full graph (decode + PGIE + tracker + single LPD SGIE + off-pipeline LPR) on its
  owner edge. Runtime offload level `0`.
- **L2 — plate-crop offload for off-pipeline LPR** (lighter). Only
  plate crops are shipped to a peer for LPR inference (`offload/plates/{src}/{dst}`);
  the decode/tracking stream stays local. Relieves the **LPR crop queue only** — explicitly
  *not* decode/tracking/GPU load. Runtime offload level `1` (stored via
  `set_offload_level(cam, 1, peer)`).
  **Wire legacy**: the Zenoh plate-crop payload still carries `"level": 3` (legacy artifact from
  the pre-ratification enum); the receiver synthesizes a session on first crop receipt.
  Result payloads use `level: 1`.
- **L1 — owner-authoritative full-stream RTSP camera migration** (heavier, triggers after L2/hold).
  The entire camera stream is migrated to a peer via the RFO/lease path. The only mechanism that
  relieves decode/tracking/resource pressure; owner retains lease authority, receiver is host-only.
  Runtime offload level `2` (tracked by lease/RFO machinery).

The **vehicle-crop tier is retired** (was legacy `offload_level==2` in the oldest enum). No orchestrator
trigger emits a vehicle-crop session handshake; receiver drops unsessioned vehicle crops. In the
current canonical enum `2` denotes full-stream RTSP migration (L1), tracked by lease machinery.

### Mandatory Escalation Ladder (L0 → L2 → L1)

The system strictly enforces a sequential offload ladder under node overload:

1. **L0 → L2 Escalation**: When a node is overloaded (`load_score >= overload_threshold`,
   default 55.0 sustained for `overload_duration_s=3.0s`), the orchestrator initiates
   L2 plate-crop offload for its **heaviest local camera** (`rank_by="workload"`).
   If no peer/candidate is available, the node fast-escalates to L1 instead of holding
   in overload.
2. **L2 Observation Hold Window**: The node enters an observation hold window
   (`ladder_l2_hold_s`, default 8.0s), holding L2 while peer LPR relieves
   local queue and service deficit before considering L1.
3. **L2 → L1 Escalation**: Only if the node **remains overloaded** (`load_score >= 55.0`
   and `stream_pressure >= 0.30`) after the hold has elapsed does the orchestrator
   escalate to L1 full-stream RTSP migration (RFO broadcast). If local load recovers below
   the threshold during the hold window, L1 migration is avoided entirely.
4. **L2 Cleanup on L1**: When L1 migration is triggered for a camera, any existing L2
   plate-crop offload for that camera is de-escalated back to L0 (`set_offload_level(cam, 0)`).
5. **Direct L2 Queue Relief**: In addition to the overload ladder, L2 plate-crop offload
   can activate independently when the local LPR worker queue saturates
   (`lpr_queue_ratio > lpr_offload_up_threshold`, default 0.75), and reclaims when the
   queue drains (`lpr_offload_down_threshold`, default 0.35).

### De-escalation (return to level 0)

When the relevant signal clears and stays clear for `offload_release_dwell_s` (default 5.0s),
the level is cleared: `set_offload_level(cam, 0)` stops crop production at the SpeedProbe.
A short dwell prevents flapping when the signal oscillates near the threshold.

### Crop offload backpressure (L2 plate-crop)

L2 plate-crop offload is sender-gated, not a load-band handshake:

- For plate crops the receiver **synthesizes** a session on first receipt (no explicit
  `start`/`stop` handshake is published for L1); crops without a synthesized session are
  dropped at the subscriber callback (counter `session_dropped_count`) — no GPU cost.
- **Backpressure**: the receiver exports `offload_queue_full` (≥ 80% of its 32-slot
  queue) and queue-depth ratio into the heartbeat; the sender's orchestrator tracks per-peer
  saturation with a release hysteresis, and SpeedProbe skips crop production for
  saturated targets (counter `l2_dropped_backpressure`). Dropping at the sender is almost
  free; dropping at the receiver costs queue + GPU pressure — the system deliberately
  drops early.

### Camera selection & ownership policy (hard rules)

- Per-camera ranking uses **workload** (`n_track + n_plate`), never post-mux FPS:
  L2 migration (full-stream L1) picks the *lightest* owned camera, L2 plate-crop picks the *heaviest* camera.
- **A node never migrates away a camera it does not own** (ownership defined by
  `node_camera_map`, built from all nodes' `cameras.yml`). Foreign/rescued cameras are
  skipped for L2 — never re-homed to third nodes; they may only return to their owner
  via the owner's reclaim path.
- **A node always keeps ≥ 1 locally-owned camera active.** L2 returns no candidate when
  ≤ 1 owned camera remains.
- Ownership commits only after the receiver reaches PLAYING **and** the sender has
  removed the camera **and** the sender's remaining streams are verified healthy.
- Ownership changes carry a monotonic **owner epoch** in all ADD/REMOVE/vote/ACK
  messages; receivers reject stale-epoch removals (fail-closed).
- Migration/reclaim targets are validated against current cluster state (heartbeats),
  not local snapshots: RFO rejected if an alive peer already owns the camera,
  per-camera single-flight, dynamic holders yield to static owners.

### L2 migration flow (make-before-break)

1. Overloaded node broadcasts RFO on `peers/vote/request`.
2. Capable peers respond on `peers/vote/proposal` within `vote_window_s=5`;
   saturated peers are excluded as targets.
3. Winner selected by ε-constraint (thermal, stream count, forecast FPS, RTT).
4. Winner receives decision on `peers/vote/decision`, ADDs the camera, waits for
   PLAYING (`add_ack_timeout_s=8`, config `edge_node.yml`).
5. Winner ACKs on `peers/vote/ack/{cam}`.
6. Sender receives ACK → REMOVEs its local branch. Stream continuity preserved.
7. Rollback: no ACK within `migration_timeout_s=12` (config `edge_node.yml`) → sender aborts; repeated failures
   exclude a peer (`zombie_timeout_count=3`). Losing bidders' reservations decay on a
   timer after the decision.

Reclaim verifies the current holder via heartbeats, uses persistent attempt counters
(`reclaim_max_attempts=5`), and gives up cleanly instead of looping forever.

### Failover rescue

Peers declare a node OFFLINE after `heartbeat_timeout_s=10` (config `edge_node.yml`) plus a
failover/convergence grace; `recovery_wait_s=60` (config `edge_node.yml`) waits for short bounces before
rescuing.

Surviving nodes rescue orphaned cameras using **HRW/Rendezvous hashing**
(`sha256(camera_id:peer_id)`, highest weight wins) — only the dead node's cameras remap.
Each rescuer publishes a priority claim on `peers/failover/claim` (weight truncated to
15 hex digits for msgpack int64 safety), waits `rescue_claim_window_s=8` (claim lease
`rescue_claim_lease_s=15`), and yields to higher-weight peers (split-brain guard).
Orphan detection intersects the dead peer's last-known `camera_configs`; empty
`active_cameras` fields never suppress rescue nor overwrite cached ownership.

### Result filtering

Crop offload results return on `offload/results/{receiver}/{sender}` with
`{stid, camera_id, frame_no, plate_text, confidence, inference_ok}`. The original node
filters by matching `stid + camera_id` against live tracks before inserting plate text
into violation records.

---

## Heartbeat & Transport Resilience

Lessons baked into code after field incidents:

- **Single publisher, shared session.** `HealthAgent`, `PeerOrchestrator`, and the
  command/offload subscribers share one Zenoh session; self-heartbeat loopback works
  without mesh reconvergence after restarts.
- **Publish-failure recovery.** A failed heartbeat put closes and undeclares the
  publisher, then reconnects immediately instead of waiting out a retry interval.
  Send/error/consecutive-error counters and throttled warnings track transport health.
- **Self-stale gates.** When this node's own heartbeat is stale beyond the offline
  threshold, offline detection of peers is *skipped* — absence of peer updates over a
  broken transport must not produce false OFFLINE flags. The same gate suppresses
  overload decisions.
- **No lock-across-callbacks.** Camera ADD/REMOVE ACK waits run on a bounded thread
  pool; stream-pad operations release `CameraManager._lock` before blocking calls
  (the GLib main loop needs the same lock).

## RTSP Source Resilience

- **rtspsrc auto-reconnect.** Each RTSP source bin sets `reconnect-delay=5` on the underlying `rtspsrc` element (via the `uridecodebin` source-setup signal). When the camera container restarts or closes the TCP connection, `rtspsrc` automatically re-establishes the RTSP session after 5 s without bubbling EOS to the pipeline.
- **EOS source discrimination.** The GStreamer bus `on_message` EOS handler distinguishes file-source EOS (legitimate end-of-stream → allow `loop.quit()`) from RTSP-source EOS (transient TCP disruption → suppress quit, log warning, let rtspsrc reconnect). This prevents camera container restarts from killing the entire DeepStream pipeline and triggering an NVDEC churn cycle that risks the CQHCI/eMMC kernel lockup on 5.15.199-tegra.

## Stream Lifecycle Robustness

- **Teardown cascade elimination (`NOT_LINKED` probe + prefix matching).** During dynamic
  removal (`_teardown_source_branch`), an IDLE block pad probe is placed on `conv_pad`
  to drop all buffers and downstream events before `conv_pad.unlink(mux_sinkpad)`,
  completely eliminating `GST_FLOW_NOT_LINKED (-1)` bus errors from upstream queues.
  The GStreamer bus error handler matches `q_` and `conv_` element prefixes to isolate
  branch teardown errors from the main pipeline. Synchronous teardown waits (`get_state`,
  `drained.wait`) execute decoupled from the GLib main loop.
- **Synchronous removal.** Teardown is a sequential PLAYING→PAUSED→READY→NULL state
  walk with get_state waits (nvv4l2decoder hardware-register requirement — direct
  PLAYING→NULL deadlocks the kernel and requires a power cycle). A post-teardown audit
  verifies the inner NVDEC actually reached NULL and logs the session count + RSS delta.
- **Stale-source cleanup on ADD.** Re-adding a camera detects and tears down leftover
  elements/pads from a previous timed-out ADD before building a new source bin.
- **ADD timeout handling.** Timed-out configs are disabled and cleaned up; reclaim
  retries re-enable them.
- **Frozen-slot display fix.** After removal, a black `videotestsrc` fills the vacated
  tiler slot (live tiler grid resize crashes VIC on Jetson), removed again on reclaim.

---

## Zenoh Communication

Peer mode with UDP multicast scouting; if `ZENOH_ROUTER` is set, nodes also connect to
the router endpoint (used for the cross-subnet server link).

| Key expression pattern | Direction | Purpose |
|------------------------|-----------|---------|
| `peers/status/{node_id}` | broadcast | Health heartbeat (msgpack), sole publisher |
| `peers/vote/request` | broadcast | RFO — overloaded node requests offload |
| `peers/vote/proposal` | broadcast | Bid from capable peer |
| `peers/vote/decision` | broadcast | Winner selection |
| `peers/vote/ack/{cam_id}` | broadcast | Receiver confirms stream PLAYING |
| `peers/control/{node_id}` | directed | Camera ADD/REMOVE commands (+ACKs) |
| `peers/failover/claim` | broadcast | Rescue priority claims (HRW weights) |
| `traffic/events/{node_id}/**` | local | Pipeline → HealthAgent → Server (overspeed) |
| `offload/plates/{src}/{dst}` | directed | L2 plate crops (runtime offload_level==1; wire payload carries legacy `"level": 3`) |
| `offload/vehicles/{src}/{dst}` | directed | (retired — vehicle-crop tier removed, see ADR-0002) |
| `offload/session/{src}/{dst}` | directed | RFO/lease control channel (L1 migration) |
| `offload/results/{recv}/{sender}` | directed | Inference results back to origin |

---

## B-side Offload Receiver

`OffloadReceiver` subscribes to:
- `offload/plates/*/{my_node_id}` — L2 plate-crop (runtime offload_level==1; legacy wire `"level": 3`): run LPR engine on plate crop
- `offload/vehicles/*/{my_node_id}` — (retired — vehicle-crop tier removed, see ADR-0002)
- `offload/session/*/{my_node_id}` — RFO/lease control channel for L1 migration

TensorRT engines (`models/lpd.engine`, `models/lpr.engine`) are loaded lazily on first
request. Dynamic shapes: profile 0 MIN shape used for batch-1 inference (supports both
TensorRT 8 / JetPack 5 and TRT 10 / JetPack 6 APIs).

**Failed inference is not published.** Only successful runs (including a valid empty
observation) publish a result. Engine unavailable or inference exception → result
suppressed, error counter incremented, rate-limited warning logged.

Work-queue saturation logging is rate-limited (1st drop, then every 500th) instead of
one line per dropped crop.

---

## Logging

Dual handlers (configured in `run_python.py`):

- **Terminal:** INFO and above — operator sees state changes.
- **File `Edge/logs/edge_debug.log`:** DEBUG+ with rotation — full diagnostic detail.

Additionally stdout/stderr tee into `Edge/logs/run_<timestamp>.log` per launch.
The process PID of the nohup wrapper shell is written to `Edge/logs/run_edge.pid` on startup and used as the primary kill target during redeploy teardown.
Per-second DEBUG chatter in the orchestrator decision loop (overload checks, idle
states) is throttled through a block-rate logger (~60 s cooldown), so a healthy run
produces kilobytes, not megabytes.

---

## Running the Edge Node

All Edge commands run from `Edge/` with the `DoAn` conda environment.

```bash
# Default: start pipeline + HealthAgent + PeerOrchestrator (one process), RTSP push mode
./run_edge.sh

# Run in background (survives SSH disconnect):
nohup ./run_edge.sh >/dev/null 2>&1 &

# Display to screen
./run_edge.sh --mode display

# Custom RTSP push destination
./run_edge.sh --mode rtsp_push --rtsp-push-url rtsp://<server>:8554/jetson_A

# Collect calibration data (600 s alongside live pipeline)
./run_edge.sh --collect --collect-duration 600

# Full automated calibration (W_base → collect → fit → plot)
./run_edge.sh --calibrate

# Train DL load predictor from pre-collected CSVs (no pipeline needed)
./run_edge.sh --train-dataset csv_collected --load-model dl

# Stop: Ctrl+C (graceful shutdown, then SIGKILL)
```

`HEALTH_INTERVAL=1.0` is the only supported cadence (matches SpeedProbe writer).

**Offline verification** (from repo root, optional):
```bash
cd Edge/speedflow_python && python3 verify_offload_contract.py
conda run -n DoAn python3 -m py_compile Edge/speedflow_python/run_python.py
```
The test suite (`Edge/tests/`) was removed in commit `4d37705`; `verify_offload_contract.py`
asserts the canonical offload-level enum and is the only remaining in-repo contract check.

Deployment to Jetsons is rsync-based: copy changed files from the host to all three
devices using `rsync` (explicitly excluding `Edge/.env` to preserve per-node
`setup_system.sh` identity), then restart `run_edge.sh`. All nodes run the exact
same codebase and offload protocol version.

### Offline Load-Calibration & Telemetry Data Collection:

Production starts the Edge node plain (`nohup ./run_edge.sh >/dev/null 2>&1 &`); Jetsons do
**not** persist training/calibration CSV data or logs locally in production to protect eMMC life
and prevent I/O blocking.

Instead, each Jetson continuously emits full metadata and telemetry snapshots via Zenoh
(`peers/status/{node_id}`). The central Server (`Server/telemetry_store.py`) receives this stream
and persists `data/telemetry/calibration_{node_id}.csv`: the 14-column raw metrics snapshots
(containing `gpu_percent`, `fps_avg`, `n_track_total`, `n_plate_total`, `load_score`, etc.)
that the (offline, now-removed) proactive / DL calibration scripts once consumed —
without placing any file I/O overhead on the Jetsons.

Offline commands on Edge with `--collect` / `--calibrate` are strictly deprecated in production.

### Cluster Lifecycle & Experiment Redeploy Order:
- **Teardown Sequence (before syncing files)**:
  1. **Jetsons first**: Kill the Edge process by PID: read `logs/run_edge.pid` and run `kill <pid>` (kills the nohup shell wrapper and its while-loop respawner). Fall back to `echo "123456@2024" | sudo -S pkill -f python3` only if the `.pid` file is missing. Clear old logs and `nohup*.out` files.
  2. **Camera next**: Stop camera simulation containers (`docker compose down`).
  3. **Server last**: Kill all python3 processes (`pkill -f python3`).
  4. **Mid-sequence failure rule**: If any step fails or the sequence is interrupted before all three component groups are cleanly torn down, restart the entire teardown from step 1. No mid-way patching or partial restarts — a clean slate is mandatory before syncing and starting.
  5. **Preserve forensics evidence**: Before clearing logs, ensure persistent **journald** and **pstore** evidence is preserved on the Jetsons (they live outside `Edge/logs/` and are unaffected by log cleanup).
- **Fleet Sync Sequence**:
  Copy only the relevant component to each target — Jetsons receive only `Edge/`, the VPS receives only `Server/`. Never copy the full repo tree to either.
  ```bash
  # Sync Edge to all three Jetsons (exclude per-node .env — never overwrite it)
  for ip in 192.168.212.20 192.168.212.21 192.168.212.22; do
    rsync -avz --exclude='.env' --exclude='logs/' --exclude='__pycache__/' \
      Edge/ mta@$ip:/home/mta/Documents/IoT_Graduate/Edge/
  done

  # Sync Server to VPS
  rsync -avz --exclude='logs/' --exclude='__pycache__/' --exclude='violations/' \
    --exclude='data/' \
    Server/ mta@116.118.9.125:/home/mta/Documents/IoT_Graduate/Server/
  ```
  `Edge/.env` contains per-device identity (`NODE_ID`, `ADVERTISE_IP`, `RTSP_PUSH_URL`) provisioned by `Edge/setup_system.sh jetson_A` (or `jetson_B` / `jetson_C`). It must **never** be overwritten with the repository's tracked `jetson_A` template.
- **Per-Node Provisioning (run after sync, if .env identity changed or first deploy)**:
  ```bash
  echo "123456@2024" | sudo -S bash Edge/setup_system.sh <node_id>
  ```
  Run `setup_system.sh <node_id>` once per Jetson (e.g., `jetson_A`, `jetson_B`, `jetson_C`). This provisions `Edge/.env`, generates `Edge/configs/cameras.yml`, and sets up the Camera docker-compose for that node's two cameras.
- **Startup Sequence**:
  1. **Server first**: `nohup bash ./start.sh 2>&1 &` (starts MediaMTX relay, dashboard on `:9090`).
  2. **Camera next**: `cd Camera && docker compose up -d --build` (starts RTSP video sources).
  3. **Jetsons last**: Launch plain edge instances (`./run_edge.sh`) on all 3 AGX Orin nodes simultaneously — no `--collect`/`--calibrate` flags (those are offline research tools, not production).
  4. **RTSP Stream Verification Gate (300 s)**: Before proceeding to soak, verify all 6 RTSP streams are live and stable:
     ```bash
     for cam in cam_01 cam_02 cam_03 cam_04 cam_05 cam_06; do
       timeout 10 ffprobe rtsp://<advertise_ip>:8554/${cam} >/dev/null 2>&1 && echo "$cam OK" || echo "$cam FAIL"
     done
     ```
     Confirm all 6 cameras report `OK`. Monitor continuously for at least 300 seconds (5 minutes) of stable stream presence and zero dropped queues before proceeding.
   5. **Soak Gate (1800 s)**: After the startup gate passes, observe the system for a full 1800-second (30-minute) soak window. During soak: no process crashes, no NVDEC session limit warnings, no thermal throttling, heartbeat cadence stable at 1 s. Failure at any point during soak resets to teardown step 1.
- **Telemetry Collection**: No local Edge CSV/log collection. All telemetry flows server-side via Zenoh heartbeat (`peers/status/{node_id}`); the Server persists `data/telemetry/calibration_{node_id}.csv` as the authoritative data source. Research calibration commands (`--collect`, `--calibrate`) are offline-only tooling and never run in production.

---

## Camera (Simulation)

```bash
cd Camera
docker compose up -d        # starts MediaMTX :8554 + FFmpeg cam containers
docker compose down
```

Video files go in `Camera/videos/`. Edit `docker-compose.yml` or `.env` to add cameras.

> Note: Jetson kernel images without `veth.ko` cannot use Docker bridge networking;
> use `network_mode: host` there.

---

## Server

```bash
cd Server
python3 app.py              # dashboard at http://0.0.0.0:9090
```

Requires `Server/.env` with `SERVER_PORT`, `MEDIAMTX_API`, etc.
Violations stored as JSON in `Server/violations/`. The same host can run a Zenoh
router (`:7447`) for cross-subnet discovery.

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `Edge/.env` | Flat settings: `NODE_ID`, `TARGET_FPS=28`, `HEALTH_INTERVAL=1.0`, RTSP URLs, `ZENOH_ROUTER`, model paths, `SPEEDFLOW_SLOT_CAPACITY=16`, `SPEEDFLOW_NVDEC_SESSION_LIMIT`, `EDGE_BLEED_DIAGNOSTICS=1` |
| `Edge/configs/cameras.yml` | Per-camera RTSP URIs, homography, ROI, speed limit, nominal FPS. Hot-reloaded (~100 ms via inotify). Defines **ownership**. |
| `Edge/configs/edge_node.yml` | P2P thresholds (`overload_threshold=55.0`, `overload_duration_s=3.0`, `ladder_l2_hold_s=8.0`, `offload_release_dwell_s=5.0`), dwell timers, heartbeat/failover/grace windows, load_score mode/bonuses, workload policy, proactive model |
| `Server/.env` | `SERVER_PORT=9090`, `MEDIAMTX_API` |
| `Camera/.env` | RTSP port, video file paths for Docker sim |

---

## Proactive Model (Research, Disabled)

`edge_node.yml → proactive:` holds a formula-based predictor
(`W_base + Σ[α₁·N_track + α₂·N_track² + β·N_plate + γ·S]`) or a DL ONNX predictor.
Both are **disabled by default**; shadow mode emits risk telemetry without affecting
decisions. Research framing: predict sustained-FPS-threshold crossing (QoS degradation)
within short horizons — evaluated by lead time, F1, false alarms, missed events — not
by regression MAE/RMSE.

Promotion gate (not yet met): a forecaster must beat persistence consistently on fresh
Mode-A data (no offload/migration confounding) AND its lead time must exceed actual
migration/offload latency. Current result: Ridge beats persistence overall but loses
individual scenarios — nothing promoted.

`load_score` and object counts carry policy feedback once offload is active, so
training/evaluation uses Mode-A data only (confounded windows tagged or excluded).

---

## Jetson Lockup & Freeze Investigation

Persistent **journald** (system-level forensic logging, outside `Edge/logs`) is
provisioned by `Edge/setup_system.sh` (section 0a): it creates `/var/log/journal`,
enforces `Storage=persistent` in `/etc/systemd/journald.conf`, and reports the
pstore/watchdog evidence surfaces without arming the watchdog. Journald/pstore
reside outside `Edge/logs/` and are unaffected by Edge log cleanup.

The historical diagnostic tools were removed — `Edge/tools/jetson_diag_collector.sh`
and `Edge/tools/jetson_diag_watcher.sh` were collector/ring-buffer helpers only, and
`Edge/deploy/JETSON_LOCKUP_INVESTIGATION.md` was deleted (see git history). Persistent
journald is now the provisioning mechanism; none of the removed tools are required to
install or activate it.

---

## Known Constraints (Jetson-specific)

- Sequential decoder teardown only (PLAYING→PAUSED→READY→NULL with get_state waits);
  direct PLAYING→NULL leaves NVDEC registers undefined and can deadlock the kernel.
- Mux sink pads are permanent (`SPEEDFLOW_SLOT_CAPACITY=16`); pad add/remove on a live
  `nvstreammux` corrupts surface pools.
- NVDEC hardware decode sessions cap around 16–32 per Orin; ADD is gated at
  `SPEEDFLOW_NVDEC_SESSION_LIMIT` (fleet runs 14).
- Live tiler grid resize crashes VIC — vacated slots are black-filled instead.
- Docker bridge requires `veth.ko`; absent on current flashed kernels — use host
  networking.
- BSP kernel divergence (OBSERVED 2026-09-09): `jetson_A` runs L4T R36.4.7 / kernel 5.15.148-tegra; `jetson_B` and `jetson_C` run L4T R36.5.2 / kernel 5.15.199-tegra. Both hard freezes occurred on 5.15.199 — suspect CQHCI/eMMC regression (commits f4780fedeb65, c0f43b1f1f7d). Persistent journald + pstore enabled on all nodes for forensics.

## Limitations / Unproven

- L2 plate-crop offload end-to-end throughput under sustained thermal load not yet
  benchmarked after the backpressure rework.
- Simultaneous multi-node failure recovery tested only up to three-node setups.
- WAN uplink to the server is ~4.5 Mbps — well under 3 nodes × 4 cameras × 3 Mbps
  encoder defaults; per-camera push bitrate (`RTSP_PUSH_BITRATE`) must be tuned down
  or bandwidth confirmed before wide deployment.
- No TLS/auth on Zenoh, WebSocket, or RTSP endpoints — trusted LAN only.
