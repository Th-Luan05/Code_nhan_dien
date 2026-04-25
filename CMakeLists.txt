cmake_minimum_required(VERSION 3.10)
project(yolov8_ncnn_pi4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=armv8-a+fp+simd -mtune=cortex-a72")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# NCNN
set(NCNN_DIR "$ENV{HOME}/ncnn_install" CACHE PATH "NCNN install prefix")

if(NOT EXISTS "${NCNN_DIR}/include/ncnn/net.h")
    message(FATAL_ERROR
        "Khong tim thay NCNN tai: ${NCNN_DIR}\n"
        "Chay ./install_ncnn_pi4.sh truoc, hoac:\n"
        "  cmake .. -DNCNN_DIR=/duong/dan/ncnn_install")
endif()

include_directories(${NCNN_DIR}/include/ncnn)
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(NCNN_LIB NAMES ncnn PATHS ${NCNN_DIR}/lib NO_DEFAULT_PATH)
if(NOT NCNN_LIB)
    message(FATAL_ERROR "Khong tim thay libncnn tai ${NCNN_DIR}/lib")
endif()

# wiringPi
include_directories(/usr/local/include)

find_library(WIRINGPI_LIB NAMES wiringPi PATHS /usr/local/lib /usr/lib)
if(NOT WIRINGPI_LIB)
    message(FATAL_ERROR "Khong tim thay libwiringPi! Chay: sudo apt install wiringpi")
endif()

# Executable
add_executable(yolov8_pi4
    yolov8_ncnn_pi4.cpp
    Pi4RobotDriver.cpp
)
target_link_libraries(yolov8_pi4
    ${NCNN_LIB}
    ${OpenCV_LIBS}
    ${WIRINGPI_LIB}
    pthread
    curl
)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV     : ${OpenCV_VERSION}")
message(STATUS "NCNN dir   : ${NCNN_DIR}")
message(STATUS "NCNN lib   : ${NCNN_LIB}")
message(STATUS "wiringPi   : ${WIRINGPI_LIB}")