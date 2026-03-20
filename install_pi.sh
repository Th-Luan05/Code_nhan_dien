#!/bin/bash
# install_pi.sh
# Cài tất cả dependencies trên Raspberry Pi 4 (Raspberry Pi OS 64-bit)
# Chạy: chmod +x install_pi.sh && ./install_pi.sh

set -e
echo "=============================="
echo " Cài dependencies cho Pi 4"
echo "=============================="

# Update
sudo apt update

# ── OpenCV ────────────────────────────────────
echo "[1/5] Cài OpenCV..."
sudo apt install -y libopencv-dev

# ── GStreamer + WebRTC plugins ─────────────────
echo "[2/5] Cài GStreamer..."
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-nice

# ── NCNN ──────────────────────────────────────
echo "[3/5] Cài NCNN..."
sudo apt install -y libncnn-dev

# Nếu apt không có ncnn, build từ source:
# sudo apt install -y cmake git
# git clone https://github.com/Tencent/ncnn.git
# cd ncnn && git submodule update --init
# mkdir build && cd build
# cmake .. -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_VULKAN=OFF
# make -j4 && sudo make install

# ── libcamera ─────────────────────────────────
echo "[4/5] Cài libcamera..."
sudo apt install -y libcamera-dev libcamera-apps

# ── httplib (header-only) ─────────────────────
echo "[5/5] Cài httplib..."
mkdir -p third_party
if [ ! -f third_party/httplib.h ]; then
    wget -q -O third_party/httplib.h \
        https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
    echo "  httplib.h downloaded"
else
    echo "  httplib.h sudah ada"
fi

# ── CMake ─────────────────────────────────────
sudo apt install -y cmake build-essential

echo ""
echo "=============================="
echo " Xong! Giờ build chương trình:"
echo ""
echo "   mkdir build && cd build"
echo "   cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "   make -j4"
echo ""
echo " Chạy:"
echo "   ./yolo_stream --model ../ncnn_model --port 8080"
echo ""
echo " Xem stream: http://<IP_PI>:8080"
echo "=============================="
