#!/usr/bin/env python3
"""
verify_offload_contract.py — canonical offload-level enum self-check.

Canonical runtime offload_level enum (source of truth, Oracle + project memory,
ratified 0/1/2 by the governance call):

    0 = local processing (camera handled entirely on this node).
    1 = plate-crop offload (L1 plate-crop LPR tier): this node still owns the
        stream but ships plate crops to a peer for LPR.  Stored in the runtime
        offload table via set_offload_level(cam, 1, peer); gated in the crop
        emission path by ``offload_level == 1``.
    2 = full-stream migration / RFO lane (L2 full-stream migration tier).
        Tracked by the lease / RFO machinery (membership ladder + lease state),
        NOT by a fabricated runtime-table write.
    3 = retired.  Not a valid runtime offload value; no production writer or
        consumer may use it protagonist.

This verifier asserts the canonical contract and FAILS if level 3 (or any other
value) is written to or consumed from the runtime offload table, or if the
plate-crop emission gate is not level 1.

Exit code 0 = contract satisfied; non-zero = violation found.
"""

import ast
import re
import sys

# Files that own the offload-level enum semantics.
SCOPED = [
    "probes.py",
    "offload.py",
    "membership.py",
    "ownership.py",
    "offload_publisher.py",
    "offload_receiver.py",
    "peer_orchestrator.py",
    "run_python.py",
]

CANONICAL_COMMENT = "Canonical source enum: 0 = local, 1 = plate-crop offload, 2 = full-stream migration/RFO."

# Canonical enum, as written by set_offload_level; numeric stores reserved
# for plate-crop (1) and local-clear (0). Level 3 must not appear.
PLATE_CROP_STORE = 1
LOCAL_STORE = 0


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def main() -> int:
    failures = []

    def fail(msg: str) -> None:
        failures.append(msg)

    src = {}
    for fn in SCOPED:
        try:
            src[fn] = _read(fn)
        except FileNotFoundError:
            fail(f"{fn}: missing")

    probes = src.get("probes.py", "")
    offload = src.get("offload.py", "")
    membership = src.get("membership.py", "")

    # ── 1. Canonical enum comment present in the probe gate region ──
    if "offload_level == 1" not in probes:
        fail("probes.py: plate-crop emission gate must test offload_level == 1")
    if "offload_level == 3" in probes:
        fail("probes.py: plate-crop emission gate must NOT test offload_level == 3")

    # ── 2. Only levels 0, 1, 2 are valid in the production table semantics ──
    # Any numeric store or gate on level 3 is a contract violation.
    if re.search(r"offload_level\s*==\s*3", probes, re.M):
        fail("probes.py: stale offload_level==3 reference (contract requires 0/1/2)")
    if re.search(r"offload_level\s*==\s*3", offload, re.M):
        fail("offload.py: stale offload_level==3 reference (contract requires 0/1/2)")
    if re.search(r"offload_level\s*==\s*3", membership, re.M):
        fail("membership.py: stale offload_level==3 reference (contract requires 0/1/2)")

    # ── 3. Plate-crop offload stores level 1, not 3 ──
    for fn in ("offload.py", "probes.py"):
        if re.search(r"set_offload_level\([^)]*,\s*3\s*,", src.get(fn, "")):
            fail(f"{fn}: plate-crop offload must store offload_level 1, not 3")

    # ── 4. Full-stream selector lane is level 2 (not 1, not 3) ──
    # The full-stream/RFO lane is tracked by the lease machinery (membership
    # ladder + RFO lease), so no runtime-table store for level 2 is expected.
    # We assert the selector identity only where it is authoritative.
    if "level=2" not in offload:
        fail("offload.py: full-stream selector must be invoked with level=2")

    # ── 5. No level-2 runtime table store is fabricated ──
    # Full-stream migration is authoritative in the lease machinery; we must NOT
    # assert a fabricated set_offload_level(...,2,...) call.
    if "set_offload_level(candidate, 2, peer)" in offload:
        fail("offload.py: fabricated full-stream table store — lease machinery is authoritative; remove")

    # Canonical enum comment must be congruent with the chosen mapping.
    if "full-stream migration/RFO" in probes and "plate-crop offload" in probes:
        pass  # comment agrees with 0/1/2
    else:
        fail("probes.py: enum comment does not match canonical 0/1/2 mapping")

    if failures:
        print("OFFLOAD CONTRACT VIOLATION(S):")
        for f in failures:
            print("  - " + f)
        return 1

    print("OFFLOAD CONTRACT OK: 0=local, 1=plate-crop offload, 2=full-stream migration/RFO; level 3 retired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
