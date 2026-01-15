#!/bin/bash
# System dependencies required for IoT_Graduate project
# Generated based on system analysis

# Build tools for C++ Plugin
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config

# GStreamer Development Libraries (Required for compiling speedflow_cpp)
sudo apt-get install -y \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev

# GStreamer Plugins (Runtime)
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# Python GStreamer Bindings
sudo apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    python3-dev

# Utilities
sudo apt-get install -y \
    v4l-utils

echo "System dependencies checked."
