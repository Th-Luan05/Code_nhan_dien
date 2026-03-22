/**
 * YOLOv8 NCNN – Raspberry Pi 4
 * ==============================
 * - Hiển thị : full frame 1280×720 qua MJPEG :8080
 * - Inference : crop ROI 480×480 trung tâm → letterbox 320×320
 * - Tọa độ   : bbox map về frame gốc 1280×720
 * - JSON      : gửi bbox + roi qua :8081 (HTTP + CORS)
 *
 * Letterbox 480×480 → 320×320:
 *   scale = 320/480 = 0.6667
 *   new_w = new_h = 320  (vuông nên pad = 0)
 *
 * Biên dịch:
 *   g++ -O2 -std=c++17 yolov8_ncnn_pi4.cpp \
 *       -I~/ncnn_install/include/ncnn \
 *       -L~/ncnn_install/lib -lncnn \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -pthread -o yolov8_pi4
 *
 * Chạy:
 *   ./yolov8_pi4
 *   ./yolov8_pi4 --port 8080
 *   libcamera-vid -t 0 --width 1280 --height 720 --framerate 10 \
 *     --codec mjpeg --nopreview -o - 2>/dev/null | \
 *   ./yolov8_pi4 --video /dev/stdin
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <unistd.h>

#include "net.h"
#include <opencv2/opencv.hpp>

// ═══════════════════════════════ CẤU HÌNH ════════════════════════════════════

static const std::string PARAM_PATH = "best_ncnn_model/model.ncnn.param";
static const std::string BIN_PATH   = "best_ncnn_model/model.ncnn.bin";
static const char* INPUT_LAYER      = "in0";
static const char* OUTPUT_LAYER     = "out0";

// Frame gốc
static const int CAM_W    = 640;
static const int CAM_H    = 360;

// Vùng nhận diện (hình chữ nhật) – chỉnh ROI_W và ROI_H theo ý muốn
static const int ROI_W  = 300;                    // ← chiều rộng
static const int ROI_H  = 200;                    // ← chiều cao
static const int ROI_X0 = (CAM_W - ROI_W) / 2;   // tự căn giữa = 320
static const int ROI_Y0 = (CAM_H - ROI_H) / 2;   // tự căn giữa = 120

// Model input
static const int NET_SIZE = 320;

// Letterbox ROI → NET_SIZE×NET_SIZE, tính động trong init_letterbox()
static float LB_SCALE = 1.f;
static int   LB_PAD_X = 0;
static int   LB_PAD_Y = 0;

void init_letterbox()
{
    float sw = (float)NET_SIZE / ROI_W;
    float sh = (float)NET_SIZE / ROI_H;
    LB_SCALE  = std::min(sw, sh);
    int nw    = (int)std::round(ROI_W * LB_SCALE);
    int nh    = (int)std::round(ROI_H * LB_SCALE);
    LB_PAD_X  = (NET_SIZE - nw) / 2;
    LB_PAD_Y  = (NET_SIZE - nh) / 2;
    std::cout << "[INFO] ROI " << ROI_W << "x" << ROI_H
              << " @ (" << ROI_X0 << "," << ROI_Y0 << ")\n"
              << "[INFO] Letterbox scale=" << LB_SCALE
              << " pad=(" << LB_PAD_X << "," << LB_PAD_Y << ")\n";
}

static const int   NUM_CLASSES  = 7;
static const float CONF_THRESH  = 0.45f;
static const float NMS_THRESH   = 0.45f;

// Skip frame: inference 1 frame, bỏ qua SKIP_FRAMES-1 frame kế tiếp
// SKIP_FRAMES=1 → inference mỗi frame (không skip)
// SKIP_FRAMES=2 → inference 1, skip 1  → MJPEG FPS x2
// SKIP_FRAMES=3 → inference 1, skip 2  → MJPEG FPS x3
static const int SKIP_FRAMES = 1;

static const std::vector<std::string> CLASS_NAMES = {
    "Background",
    "LapPhuong_Do",
    "LapPhuong_Vang",
    "LapPhuong_Xanh",
    "HinhTru_Do",
    "HinhTru_Vang",
    "HinhTru_Xanh",
};

// ═══════════════════════════════ STRUCT ══════════════════════════════════════

struct Detection {
    int   class_id;
    float conf;
    int   x, y, w, h;   // tọa độ trong frame gốc 1280×720
};

// ═══════════════════════════════ TIỀN XỬ LÝ ══════════════════════════════════

/**
 * Crop ROI 480×480 từ frame → letterbox → 320×320
 * ROI vuông nên pad=0, chỉ resize.
 */
ncnn::Mat preprocess(const cv::Mat& frame)
{
    // Static canvas: tái dùng bộ nhớ, không alloc mỗi frame
    // pad=0 (ROI vuông) → resize thẳng vào canvas, không cần copyTo
    static cv::Mat canvas(NET_SIZE, NET_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));

    // Không .clone() – tạo header ROI trỏ vào frame, resize đọc trực tiếp
    cv::Mat roi = frame(cv::Rect(ROI_X0, ROI_Y0, ROI_W, ROI_H));
    cv::resize(roi, canvas, cv::Size(NET_SIZE, NET_SIZE), 0, 0, cv::INTER_LINEAR);

    ncnn::Mat in = ncnn::Mat::from_pixels(canvas.data,
                                          ncnn::Mat::PIXEL_BGR2RGB,
                                          NET_SIZE, NET_SIZE);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean, norm);
    return in;
}

// ═══════════════════════════════ GIẢI MÃ ĐẦU RA ═══════════════════════════════

/**
 * Ánh xạ ngược:
 *   NET_SIZE space
 *     → bỏ letterbox pad, chia scale  → ROI space (480×480)
 *     → cộng ROI_X0, ROI_Y0           → frame gốc (1280×720)
 */
std::vector<Detection> decode_output(const ncnn::Mat& out)
{
    static bool first = true;
    if (first) {
        std::cout << "[DEBUG] output dims=" << out.dims
                  << " h=" << out.h << " w=" << out.w << "\n";
        first = false;
    }

    std::vector<cv::Rect2d> boxes;
    std::vector<float>      scores;
    std::vector<int>        class_ids;

    // Lấy pointer 1 lần trước vòng lặp – tránh tính lại mỗi anchor
    const float* row0 = out.row(0);
    const float* row1 = out.row(1);
    const float* row2 = out.row(2);
    const float* row3 = out.row(3);
    const float* cls_row[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) cls_row[c] = out.row(4 + c);

    for (int i = 0; i < out.w; i++) {
        float cx = row0[i];
        float cy = row1[i];
        float bw = row2[i];
        float bh = row3[i];

        float best_score = -1.f;
        int   best_cls   = -1;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float s = cls_row[c][i];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < CONF_THRESH) continue;
        if (best_cls == 0) continue;

        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        // Bước 1: bỏ letterbox pad, chia scale → ROI space
        x1 = (x1 - LB_PAD_X) / LB_SCALE;
        y1 = (y1 - LB_PAD_Y) / LB_SCALE;
        x2 = (x2 - LB_PAD_X) / LB_SCALE;
        y2 = (y2 - LB_PAD_Y) / LB_SCALE;

        // Clamp trong ROI
        x1 = std::max(0.f, std::min(x1, (float)ROI_W));
        y1 = std::max(0.f, std::min(y1, (float)ROI_H));
        x2 = std::max(0.f, std::min(x2, (float)ROI_W));
        y2 = std::max(0.f, std::min(y2, (float)ROI_H));
        if (x2 <= x1 || y2 <= y1) continue;

        // Bước 2: dịch về frame gốc 1280×720
        x1 += ROI_X0;
        y1 += ROI_Y0;
        x2 += ROI_X0;
        y2 += ROI_Y0;

        boxes.push_back(cv::Rect2d(x1, y1, x2-x1, y2-y1));
        scores.push_back(best_score);
        class_ids.push_back(best_cls);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, idx);

    std::vector<Detection> result;
    for (int i : idx) {
        Detection d;
        d.class_id = class_ids[i];
        d.conf     = scores[i];
        d.x = (int)boxes[i].x;
        d.y = (int)boxes[i].y;
        d.w = (int)boxes[i].width;
        d.h = (int)boxes[i].height;
        result.push_back(d);
    }
    return result;
}

// ═══════════════════════════════ JSON BUILDER ═════════════════════════════════

int make_json(char* buf, int bufsz, float fps_infer, int frame_count,
              const std::vector<Detection>& dets)
{
    // src = kích thước frame gốc để viewer scale bbox đúng
    // roi = vùng nhận diện để viewer vẽ khung xanh
    int n = std::snprintf(buf, bufsz,
        "{\"fps_infer\":%.1f,\"frame\":%d,"
        "\"src\":[%d,%d],"
        "\"roi\":[%d,%d,%d,%d],"
        "\"detections\":[",
        fps_infer, frame_count,
        CAM_W, CAM_H,
        ROI_X0, ROI_Y0, ROI_W, ROI_H);

    for (int i = 0; i < (int)dets.size(); i++) {
        const auto& d = dets[i];
        n += std::snprintf(buf + n, bufsz - n,
            "%s{\"cls\":%d,\"name\":\"%s\","
            "\"conf\":%d,\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d}",
            i ? "," : "",
            d.class_id,
            (d.class_id < NUM_CLASSES ? CLASS_NAMES[d.class_id].c_str() : "?"),
            (int)(d.conf * 100),
            d.x, d.y, d.w, d.h);
    }
    n += std::snprintf(buf + n, bufsz - n, "]}\n");
    return n;
}

// ═══════════════════════════════ SOCKET HELPER ═══════════════════════════════

int make_server(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    // Buffer gửi lớn hơn để không bị block khi gửi frame 720p
    int sndbuf = 1024 * 1024;  // 1MB
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[ERROR] bind port " << port << " thất bại\n";
        return -1;
    }
    listen(fd, 1);
    fcntl(fd, F_SETFL, O_NONBLOCK);
    return fd;
}

int try_accept(int srv)
{
    struct sockaddr_in cli{};
    socklen_t len = sizeof(cli);
    return accept(srv, (struct sockaddr*)&cli, &len);
}

// ═══════════════════════════════ MAIN ════════════════════════════════════════

int main(int argc, char* argv[])
{
    std::string video_path;
    int mjpeg_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--video" && i+1 < argc) video_path = argv[++i];
        else if (a == "--port"  && i+1 < argc) mjpeg_port = std::stoi(argv[++i]);
        else if (a == "--help") {
            std::cout
                << "YOLOv8 NCNN Pi4\n\n"
                << "  ./yolov8_pi4\n"
                << "  ./yolov8_pi4 --port 8080\n"
                << "  libcamera-vid -t 0 --width 1280 --height 720 \\\n"
                << "    --framerate 10 --codec mjpeg --nopreview -o - 2>/dev/null \\\n"
                << "  | ./yolov8_pi4 --video /dev/stdin\n";
            return 0;
        }
    }
    int json_port = mjpeg_port + 1;

    // ── Load model ────────────────────────────────────────────────────────────
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads        = 4;
    if (net.load_param(PARAM_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load param: " << PARAM_PATH << "\n"; return -1;
    }
    if (net.load_model(BIN_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load bin: "   << BIN_PATH   << "\n"; return -1;
    }
    std::cout << "[INFO] Model OK\n";
    init_letterbox();

    // ── Mở camera ─────────────────────────────────────────────────────────────
    cv::VideoCapture cap;
    if (video_path.empty()) {
        std::string gst =
            "libcamerasrc ! "
            "video/x-raw,width=" + std::to_string(CAM_W) +
            ",height=" + std::to_string(CAM_H) +
            ",framerate=10/1,format=NV12 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=2 sync=false";

        std::cout << "[INFO] Thử GStreamer...\n";
        cap.open(gst, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cout << "[WARN] GStreamer thất bại, thử V4L2...\n";
            cap.open(0, cv::CAP_V4L2);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH,  CAM_W);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_H);
                cap.set(cv::CAP_PROP_FPS, 10);
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            }
        }
    } else {
        if (video_path.find("!") != std::string::npos)
            cap.open(video_path, cv::CAP_GSTREAMER);
        else
            cap.open(video_path);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Không mở được camera.\n\n"
                  << "Thử pipe:\n"
                  << "  libcamera-vid -t 0 --width 1280 --height 720 \\\n"
                  << "    --framerate 10 --codec mjpeg --nopreview -o - 2>/dev/null \\\n"
                  << "  | ./yolov8_pi4 --video /dev/stdin\n";
        return -1;
    }
    std::cout << "[INFO] Camera OK: " << CAM_W << "x" << CAM_H << "\n";

    // ── TCP servers ───────────────────────────────────────────────────────────
    int mjpeg_srv = make_server(mjpeg_port);
    int json_srv  = make_server(json_port);
    if (mjpeg_srv < 0 || json_srv < 0) return -1;

    // Lấy IP Pi
    char ip_buf[64] = "?.?.?.?";
    struct ifaddrs* ifa;
    getifaddrs(&ifa);
    for (auto* p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr) continue;
        if (p->ifa_addr->sa_family == AF_INET &&
            std::string(p->ifa_name) != "lo") {
            inet_ntop(AF_INET,
                &((struct sockaddr_in*)p->ifa_addr)->sin_addr,
                ip_buf, sizeof(ip_buf));
            break;
        }
    }
    freeifaddrs(ifa);

    std::cout
        << "\n╔══════════════════════════════════════════════╗\n"
        << "║  MJPEG  http://" << ip_buf << ":" << mjpeg_port << "/        ║\n"
        << "║  JSON   http://" << ip_buf << ":" << json_port  << "/        ║\n"
        << "╚══════════════════════════════════════════════╝\n\n"
        << "[INFO] Chờ kết nối... Ctrl+C để dừng.\n";

    // ── Vòng lặp chính ───────────────────────────────────────────────────────
    cv::Mat frame;
    int     frame_count = 0;

    int mjpeg_cli = -1;
    int json_cli  = -1;

    std::vector<uchar> jpeg_buf;
    std::vector<int>   jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 55};

    // Skip frame: giữ kết quả detect của lần inference gần nhất
    std::vector<Detection> last_dets;
    int infer_count = 0;

    // FPS inference: đo thời gian giữa 2 lần chạy model
    float fps_infer = 0.f;
    auto  t_infer   = std::chrono::steady_clock::now();

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        if (frame.cols != CAM_W || frame.rows != CAM_H)
            cv::resize(frame, frame, cv::Size(CAM_W, CAM_H));

        // ── Inference với skip frame ──────────────────────────────────────────
        // Chỉ inference 1 trong SKIP_FRAMES frame, frame còn lại dùng kết quả cũ
        bool do_infer = (frame_count % SKIP_FRAMES == 0);
        if (do_infer) {
            ncnn::Mat in = preprocess(frame);
            ncnn::Extractor ex = net.create_extractor();
            ex.input(INPUT_LAYER, in);
            ncnn::Mat out;
            ex.extract(OUTPUT_LAYER, out);
            last_dets = decode_output(out);
            infer_count++;
        }
        // Dùng kết quả detect gần nhất cho mọi frame
        const auto& dets = last_dets;

        // ── FPS infer: chỉ cập nhật khi thực sự chạy model ─────────────────────
        if (do_infer) {
            auto  t_now = std::chrono::steady_clock::now();
            float dt    = std::chrono::duration<float>(t_now - t_infer).count();
            t_infer     = t_now;
            fps_infer   = fps_infer * 0.9f + (1.f / (dt + 1e-9f)) * 0.1f;
        }
        frame_count++;

        // ── Terminal log mỗi 30 frame ─────────────────────────────────────────
        if (frame_count % 30 == 0) {
            std::cout << "[F" << frame_count << "]"
                      << " infer=" << (int)fps_infer << "fps"
                      << " obj="   << dets.size();
            for (auto& d : dets)
                std::cout << " " << CLASS_NAMES[d.class_id]
                          << "(" << (int)(d.conf*100) << "%)";
            std::cout << "\n";
            std::cout.flush();
        }

        // ── Accept client mới (non-blocking) ──────────────────────────────────
        if (mjpeg_cli < 0) {
            int fd = try_accept(mjpeg_srv);
            if (fd >= 0) {
                mjpeg_cli = fd;
                // TCP_NODELAY: gửi ngay không chờ buffer đầy (giảm lag)
                int nodelay = 1;
                setsockopt(mjpeg_cli, IPPROTO_TCP, TCP_NODELAY,
                           &nodelay, sizeof(nodelay));
                // Buffer gửi 1MB cho client này
                int sndbuf = 1024 * 1024;
                setsockopt(mjpeg_cli, SOL_SOCKET, SO_SNDBUF,
                           &sndbuf, sizeof(sndbuf));
                const char* hdr =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace;boundary=f\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "\r\n";
                send(mjpeg_cli, hdr, strlen(hdr), MSG_NOSIGNAL);
                std::cout << "[INFO] MJPEG client kết nối\n";
            }
        }
        if (json_cli < 0) {
            int fd = try_accept(json_srv);
            if (fd >= 0) {
                json_cli = fd;
                char req_buf[1024] = {};
                recv(json_cli, req_buf, sizeof(req_buf)-1, 0);
                const char* json_hdr =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/plain; charset=utf-8\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Transfer-Encoding: chunked\r\n"
                    "\r\n";
                send(json_cli, json_hdr, strlen(json_hdr), MSG_NOSIGNAL);
                std::cout << "[INFO] JSON client kết nối\n";
            }
        }

        // ── Gửi MJPEG (full frame 1280×720) ───────────────────────────────────
        if (mjpeg_cli >= 0) {
            cv::imencode(".jpg", frame, jpeg_buf, jpeg_params);
            char phdr[96];
            int  plen = std::snprintf(phdr, sizeof(phdr),
                "--f\r\nContent-Type: image/jpeg\r\n"
                "Content-Length: %zu\r\n\r\n", jpeg_buf.size());
            int r1 = send(mjpeg_cli, phdr,             plen,             MSG_NOSIGNAL);
            int r2 = send(mjpeg_cli, jpeg_buf.data(),  jpeg_buf.size(),  MSG_NOSIGNAL);
            int r3 = send(mjpeg_cli, "\r\n",           2,                MSG_NOSIGNAL);
            if (r1 < 0 || r2 < 0 || r3 < 0) {
                std::cout << "[INFO] MJPEG client ngắt (errno=" << errno << ")\n";
                close(mjpeg_cli); mjpeg_cli = -1;
                // Xóa bbox cũ sẽ không đúng nữa – giữ nguyên last_dets
            }
        }

        // ── Gửi JSON (chunked) – chỉ khi vừa inference xong ─────────────────
        if (json_cli >= 0 && do_infer) {
            static char js_buf[2048];
            int js_len = make_json(js_buf, sizeof(js_buf), fps_infer, frame_count, dets);
            char chunk_hdr[16];
            int  hlen = std::snprintf(chunk_hdr, sizeof(chunk_hdr),
                                      "%x\r\n", js_len);
            int r1 = send(json_cli, chunk_hdr, hlen,     MSG_NOSIGNAL);
            int r2 = send(json_cli, js_buf,    js_len,   MSG_NOSIGNAL);
            int r3 = send(json_cli, "\r\n", 2,           MSG_NOSIGNAL);
            if (r1 < 0 || r2 < 0 || r3 < 0) {
                std::cout << "[INFO] JSON client ngắt\n";
                close(json_cli); json_cli = -1;
            }
        }
    }

    cap.release();
    if (mjpeg_cli >= 0) close(mjpeg_cli);
    if (json_cli  >= 0) close(json_cli);
    close(mjpeg_srv);
    close(json_srv);
    std::cout << "[INFO] Done. Frame: " << frame_count << "\n";
    return 0;
}
