cmake_minimum_required(VERSION 3.10)
project(gst_speedflow VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(YAML REQUIRED yaml-cpp)

# DeepStream paths
set(DEEPSTREAM_PATH "/opt/nvidia/deepstream/deepstream")
set(DEEPSTREAM_INCLUDE "${DEEPSTREAM_PATH}/sources/includes")
set(DEEPSTREAM_LIB "${DEEPSTREAM_PATH}/lib")

# CUDA paths
find_package(CUDA REQUIRED)

# Include directories - explicitly add all paths
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    /usr/local/include/opencv4
    /usr/include/opencv4
    ${OPENCV_INCLUDE_DIRS}
    ${YAML_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)

# Compiler flags
add_compile_options(${GSTREAMER_CFLAGS_OTHER})

# Link directories
link_directories(
    ${GSTREAMER_LIBRARY_DIRS}
    ${OPENCV_LIBRARY_DIRS}
    ${YAML_LIBRARY_DIRS}
    ${DEEPSTREAM_LIB}
)

# Source files
set(SOURCES
    src/gst_speedflow.cpp
    src/speed_calculator.cpp
    src/homography.cpp
    src/plate_associator.cpp
    src/metadata_utils.cpp
)

# Create shared library (GStreamer plugin)
add_library(gstspeedflow SHARED ${SOURCES})

# Target include directories
target_include_directories(gstspeedflow PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    /usr/local/include/opencv4
    /usr/include/opencv4
    ${DEEPSTREAM_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(gstspeedflow
    ${GSTREAMER_LIBRARIES}
    ${OPENCV_LIBRARIES}
    ${YAML_LIBRARIES}
    nvds_meta
    nvdsgst_meta
    nvdsgst_helper
    nvbufsurface
    nvbufsurftransform
    ${CUDA_LIBRARIES}
)

# Install to GStreamer plugin directory
set(GST_PLUGIN_DIR "${CMAKE_SOURCE_DIR}/build")
install(TARGETS gstspeedflow LIBRARY DESTINATION ${GST_PLUGIN_DIR})

# Print configuration
message(STATUS "DeepStream path: ${DEEPSTREAM_PATH}")
message(STATUS "OpenCV include: /usr/local/include/opencv4")
message(STATUS "GStreamer plugin will be installed to: ${GST_PLUGIN_DIR}")
