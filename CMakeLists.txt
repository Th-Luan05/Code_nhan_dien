cmake_minimum_required(VERSION 3.16)
project(yolo_stream)

set(CMAKE_CXX_STANDARD 17)

# ── Release build với tối ưu ARM ──────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a+fp+simd -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenCV ────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/pi4/ncnn/build/install/include)
# ── NCNN ──────────────────────────────────────
# Build từ source: https://github.com/Tencent/ncnn
# Tìm ncnn build từ source (cài vào /usr/local)
find_package(ncnn QUIET)
if(NOT ncnn_FOUND)
    # Fallback: tìm thủ công
    find_path(NCNN_INCLUDE_DIR ncnn/net.h
        PATHS /usr/local/include /usr/include)
    find_library(NCNN_LIB ncnn
        PATHS /usr/local/lib /usr/lib)
    if(NCNN_INCLUDE_DIR AND NCNN_LIB)
        message(STATUS "NCNN found manually: ${NCNN_LIB}")
        include_directories(${NCNN_INCLUDE_DIR})
        set(NCNN_LIBRARIES ${NCNN_LIB})
    else()
        message(FATAL_ERROR "NCNN not found. Build from source:\n"
            "  cd ~ && git clone https://github.com/Tencent/ncnn.git\n"
            "  cd ncnn && git submodule update --init\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=ON\n"
            "  make -j4 && sudo make install")
    endif()
else()
    set(NCNN_LIBRARIES ncnn)
endif()

# ── GStreamer ─────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST      REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP  REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_WEBRTC REQUIRED gstreamer-webrtc-1.0)
pkg_check_modules(GST_SDP  REQUIRED gstreamer-sdp-1.0)

include_directories(
    ${GST_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
    ${GST_WEBRTC_INCLUDE_DIRS}
    ${GST_SDP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/third_party   # httplib.h để ở đây
)

# ── Executable ────────────────────────────────
add_executable(yolo_stream main.cpp camera.cpp)

include_directories(${CMAKE_SOURCE_DIR})  # camera.h

target_link_libraries(yolo_stream
    ${OpenCV_LIBS}
    ${NCNN_LIBRARIES}
    ${GST_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${GST_WEBRTC_LIBRARIES}
    ${GST_SDP_LIBRARIES}
    pthread
)

target_compile_options(yolo_stream PRIVATE
    ${GST_CFLAGS_OTHER}
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
