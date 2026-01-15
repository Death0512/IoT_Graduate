#!/bin/bash

# Ensure we are in the script's directory (models/)
cd "$(dirname "$0")"

# Remove old logs
rm -f build_*.log

TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# 1. Convert YOLO11n (Primary Detector)
# Config expects: YOLO_n.engine
# Model is STATIC (Batch 1) -> No dynamic shapes needed
echo "Building YOLO_n.engine..."
if [ -f "yolo11n.onnx" ]; then
    # Simply convert without shape args for static model
    $TRTEXEC --onnx=yolo11n.onnx --saveEngine=YOLO_n.engine --fp16 > build_yolo11n.log 2>&1
        
    if [ $? -eq 0 ]; then 
        echo "SUCCESS: YOLO_n.engine created."
    else 
        echo "FAILURE: YOLO_n.engine failed. See models/build_yolo11n.log"
    fi
else
    echo "SKIPPING: yolo11n.onnx not found."
fi

# 2. Convert LPD (License Plate Detection)
# Config expects: lpd.engine (Batch 16)
# Model is DYNAMIC -> Needs dynamic shapes
echo "Building lpd.engine..."
if [ -f "lpd.onnx" ]; then
    $TRTEXEC --onnx=lpd.onnx --saveEngine=lpd.engine --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:16x3x640x640 \
        > build_lpd.log 2>&1
        
    if [ $? -eq 0 ]; then 
        echo "SUCCESS: lpd.engine created."
    else 
        echo "FAILURE: lpd.engine failed. See models/build_lpd.log"
    fi
else
    echo "SKIPPING: lpd.onnx not found."
fi

# 3. Convert LPR (License Plate Recognition)
# Config expects: lpr.engine (Batch 16)
# Model might be static or dynamic. Assuming dynamic for safely.
echo "Building lpr.engine..."
if [ -f "lpr.onnx" ]; then
    $TRTEXEC --onnx=lpr.onnx --saveEngine=lpr.engine --fp16 \
        --minShapes=image_input:1x3x48x96 \
        --optShapes=image_input:8x3x48x96 \
        --maxShapes=image_input:16x3x48x96 \
        > build_lpr.log 2>&1
        
    if [ $? -eq 0 ]; then 
        echo "SUCCESS: lpr.engine created."
    else 
        echo "FAILURE: lpr.engine failed. See models/build_lpr.log"
    fi
else
    echo "SKIPPING: lpr.onnx not found."
fi

echo "Build process completed."
