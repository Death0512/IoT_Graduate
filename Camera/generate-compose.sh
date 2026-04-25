#!/bin/bash

CAMERA_COUNT=${1:-3}

cat > docker-compose.yml <<EOF
services:
  rtsp_server:
    image: bluenviron/mediamtx:latest
    container_name: rtsp_server
    ports:
      - "8554:8554"  # Port RTSP cho Jetson
      - "8888:8888"  # Port HLS (để xem trên Web nếu cần)
    restart: unless-stopped
EOF

for ((i=1;i<=CAMERA_COUNT;i++))
do
    VIDEO_FILE="./videos/cam${i}.mp4"

    if [ -f "$VIDEO_FILE" ]; then
        VIDEO_PATH="/videos/cam${i}.mp4"
    else
        VIDEO_PATH="/videos/sample.mp4"
    fi

cat >> docker-compose.yml <<EOF
  cam$i:
    build: .
    container_name: cam$i
    depends_on:
      - rtsp_server
    environment:
      - VIDEO_FILE=${VIDEO_PATH}
      - RTSP_URL=rtsp://rtsp_server:8554/cam$i
    volumes:
      - ./videos:/videos
    restart: unless-stopped

EOF

done

echo "Generated docker-compose.yml for ${CAMERA_COUNT} cameras"
