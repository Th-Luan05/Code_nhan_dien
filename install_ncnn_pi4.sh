#!/bin/bash
# =============================================================================
# install_ncnn_pi4.sh  –  Cài NCNN trên Raspberry Pi 4 (nhiều cách dự phòng)
# Chạy: chmod +x install_ncnn_pi4.sh && ./install_ncnn_pi4.sh
# =============================================================================
set -e

NCNN_INSTALL="$HOME/ncnn_install"
BUILD_DIR="$HOME/ncnn_build"

echo "================================================="
echo " Cài đặt NCNN trên Raspberry Pi 4"
echo " Install dir: $NCNN_INSTALL"
echo "================================================="

# ── Cài các gói phụ thuộc ────────────────────────────────────────────────────
echo ""
echo "[1/4] Cài dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake git wget curl \
    libopencv-dev \
    protobuf-compiler libprotobuf-dev \
    libgomp1

# ── Thử clone ncnn (có fallback) ─────────────────────────────────────────────
echo ""
echo "[2/4] Tải NCNN source..."

clone_ncnn() {
    local url=$1
    local dest=$2
    echo "    Thử: $url"
    if git clone --depth=1 --branch v0.20241226 "$url" "$dest" 2>/dev/null; then
        return 0
    fi
    # Nếu tag không có thì clone HEAD
    if git clone --depth=1 "$url" "$dest" 2>/dev/null; then
        return 0
    fi
    return 1
}

if [ -d "$BUILD_DIR/ncnn" ]; then
    echo "    Thư mục đã tồn tại, bỏ qua clone."
else
    mkdir -p "$BUILD_DIR"

    # Thử GitHub
    if clone_ncnn "https://github.com/Tencent/ncnn.git" "$BUILD_DIR/ncnn"; then
        echo "    ✓ Clone từ GitHub thành công."

    # Thử Gitee (mirror Trung Quốc, thường ổn định hơn)
    elif clone_ncnn "https://gitee.com/mirrors/ncnn.git" "$BUILD_DIR/ncnn"; then
        echo "    ✓ Clone từ Gitee thành công."

    # Tải tarball từ GitHub Releases
    else
        echo "    Git clone thất bại. Tải tarball..."
        TARBALL_URL="https://github.com/Tencent/ncnn/archive/refs/tags/20241226.tar.gz"
        TARBALL="$BUILD_DIR/ncnn.tar.gz"
        mkdir -p "$BUILD_DIR"
        wget -q --show-progress -O "$TARBALL" "$TARBALL_URL" \
            || curl -L -o "$TARBALL" "$TARBALL_URL"
        echo "    Giải nén..."
        tar -xf "$TARBALL" -C "$BUILD_DIR"
        mv "$BUILD_DIR"/ncnn-* "$BUILD_DIR/ncnn"
        rm -f "$TARBALL"
        echo "    ✓ Tarball giải nén thành công."
    fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Build NCNN (có thể mất 10-20 phút trên Pi 4)..."

mkdir -p "$BUILD_DIR/ncnn/build"
cd "$BUILD_DIR/ncnn/build"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$NCNN_INSTALL" \
    -DNCNN_VULKAN=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_ENABLE_LTO=ON \
    -DNCNN_SHARED_LIB=ON \
    -DCMAKE_CXX_FLAGS="-march=armv8-a+fp+simd -mtune=cortex-a72 -O2"

make -j4
make install

echo ""
echo "[4/4] Kiểm tra cài đặt..."
ls "$NCNN_INSTALL/lib/" | grep ncnn
ls "$NCNN_INSTALL/include/ncnn/" | head -5

echo ""
echo "================================================="
echo " ✓ NCNN đã cài xong!"
echo " Install dir : $NCNN_INSTALL"
echo " Lib         : $NCNN_INSTALL/lib/libncnn.so"
echo " Include     : $NCNN_INSTALL/include/ncnn/"
echo "================================================="
echo ""
echo " Tiếp theo – build project:"
echo "   cd ~/yolov8_project"
echo "   mkdir build && cd build"
echo "   cmake .. -DNCNN_DIR=$NCNN_INSTALL"
echo "   make -j4"
