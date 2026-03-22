cmake_minimum_required(VERSION 3.10)
project(yolov8_ncnn_pi4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=armv8-a+fp+simd -mtune=cortex-a72")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─── OpenCV ──────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ─── NCNN ────────────────────────────────────────────────────────────────────
# Mặc định trỏ vào thư mục install_ncnn_pi4.sh tạo ra ($HOME/ncnn_install)
# Override: cmake .. -DNCNN_DIR=/path/to/ncnn_install
set(NCNN_DIR "$ENV{HOME}/ncnn_install" CACHE PATH "NCNN install prefix")

if(NOT EXISTS "${NCNN_DIR}/include/ncnn/net.h")
    message(FATAL_ERROR
        "Không tìm thấy NCNN tại: ${NCNN_DIR}\n"
        "Chạy ./install_ncnn_pi4.sh trước, hoặc:\n"
        "  cmake .. -DNCNN_DIR=/đường/dẫn/ncnn_install")
endif()

include_directories(${NCNN_DIR}/include/ncnn)
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(NCNN_LIB NAMES ncnn PATHS ${NCNN_DIR}/lib NO_DEFAULT_PATH)
if(NOT NCNN_LIB)
    message(FATAL_ERROR "Không tìm thấy libncnn tại ${NCNN_DIR}/lib")
endif()

# ─── Executable ──────────────────────────────────────────────────────────────
add_executable(yolov8_pi4 yolov8_ncnn_pi4.cpp)
target_link_libraries(yolov8_pi4 ${NCNN_LIB} ${OpenCV_LIBS} pthread)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV     : ${OpenCV_VERSION}")
message(STATUS "NCNN dir   : ${NCNN_DIR}")
message(STATUS "NCNN lib   : ${NCNN_LIB}")
