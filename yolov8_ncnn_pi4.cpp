
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#include "net.h"
#include <opencv2/opencv.hpp>

// ═══════════════════════════════ CẤU HÌNH ════════════════════════════════════

static const std::string PARAM_PATH = "best_ncnn_model/model.ncnn.param";
static const std::string BIN_PATH   = "best_ncnn_model/model.ncnn.bin";
static const char* INPUT_LAYER      = "in0";
static const char* OUTPUT_LAYER     = "out0";

static const int CAM_W    = 640;
static const int CAM_H    = 360;

static const int ROI_W  = 300;
static const int ROI_H  = 200;
static const int ROI_X0 = (CAM_W - ROI_W) / 2;
static const int ROI_Y0 = (CAM_H - ROI_H) / 2;

static const int NET_SIZE = 320;

static float LB_SCALE = 1.f;
static int   LB_PAD_X = 0;
static int   LB_PAD_Y = 0;
static int   LB_NW    = 0;
static int   LB_NH    = 0;

void init_letterbox()
{
    float sw = (float)NET_SIZE / ROI_W;
    float sh = (float)NET_SIZE / ROI_H;
    LB_SCALE  = std::min(sw, sh);
    LB_NW     = (int)std::round(ROI_W * LB_SCALE);
    LB_NH     = (int)std::round(ROI_H * LB_SCALE);
    LB_PAD_X  = (NET_SIZE - LB_NW) / 2;
    LB_PAD_Y  = (NET_SIZE - LB_NH) / 2;
    std::cout << "[INFO] ROI " << ROI_W << "x" << ROI_H
              << " @ (" << ROI_X0 << "," << ROI_Y0 << ")\n"
              << "[INFO] Letterbox scale=" << LB_SCALE
              << " nw=" << LB_NW << " nh=" << LB_NH
              << " pad=(" << LB_PAD_X << "," << LB_PAD_Y << ")\n";
}

static const int   NUM_CLASSES  = 8;
static const float CONF_THRESH  = 0.8f;
static const float NMS_THRESH   = 0.45f;
static const int   SKIP_FRAMES  = 1;

static const std::vector<std::string> CLASS_NAMES = {
    "Background",
    "LapPhuong_Do",
    "LapPhuong_Vang",
    "LapPhuong_Xanh",
    "HinhTru_Do",
    "HinhTru_Vang",
    "HinhTru_Xanh",
    "SP_loi"
};

// ═══════════════════════════════ STRUCT ══════════════════════════════════════

struct Detection {
    int   class_id;
    float conf;
    int   x, y, w, h;
};

// ════════════════════ SHARED FRAME BUFFER (thread-safe) ═══════════════════════
//
// FIX CORE: Thay vì send() trực tiếp trong vòng lặp camera,
// ta lưu frame JPEG vào buffer chung. Thread MJPEG/JSON đọc ra và gửi
// độc lập. Camera không bao giờ bị block bởi network.

struct SharedState {
    // MJPEG
    std::mutex              frame_mtx;
    std::condition_variable frame_cv;
    std::vector<uchar>      jpeg_data;   // frame JPEG mới nhất
    uint64_t                frame_seq = 0; // tăng mỗi khi có frame mới

    // JSON / detections
    std::mutex              det_mtx;
    std::vector<Detection>  dets;
    float                   fps_infer = 0.f;
    int                     frame_count = 0;

    std::atomic<bool>       running{true};
} g_state;

// ═══════════════════════════════ COLOR CHECK ══════════════════════════════════

bool is_valid_color(const cv::Mat& frame, const Detection& d,
                    const std::string& class_name)
{
    if (d.w <= 10 || d.h <= 10) return true;

    int cx = d.x + d.w / 2;
    int cy = d.y + d.h / 2;
    int pw = std::max(1, (int)(d.w * 0.4));
    int ph = std::max(1, (int)(d.h * 0.4));

    cv::Rect rc(cx - pw/2, cy - ph/2, pw, ph);
    rc &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (rc.empty()) return true;

    cv::Mat hsv;
    cv::cvtColor(frame(rc), hsv, cv::COLOR_BGR2HSV);
    double avg_hue = cv::mean(hsv)[0];

    if      (class_name.find("Do")   != std::string::npos)
        return (avg_hue <= 10) || (avg_hue >= 160);
    else if (class_name.find("Vang") != std::string::npos)
        return avg_hue >= 15 && avg_hue <= 35;
    else if (class_name.find("Xanh") != std::string::npos)
        return avg_hue >= 35 && avg_hue <= 85;
    return true;
}

// ═══════════════════════════════ TIỀN XỬ LÝ ══════════════════════════════════

ncnn::Mat preprocess(const cv::Mat& frame)
{
    static cv::Mat canvas(NET_SIZE, NET_SIZE, CV_8UC3, cv::Scalar(114,114,114));

    cv::Mat roi = frame(cv::Rect(ROI_X0, ROI_Y0, ROI_W, ROI_H));
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(LB_NW, LB_NH), 0, 0, cv::INTER_LINEAR);

    canvas.setTo(cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(LB_PAD_X, LB_PAD_Y, LB_NW, LB_NH)));

    ncnn::Mat in = ncnn::Mat::from_pixels(canvas.data,
                                          ncnn::Mat::PIXEL_BGR2RGB,
                                          NET_SIZE, NET_SIZE);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean, norm);
    return in;
}

// ═══════════════════════════════ GIẢI MÃ ĐẦU RA ═══════════════════════════════

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

    const float* row0 = out.row(0);
    const float* row1 = out.row(1);
    const float* row2 = out.row(2);
    const float* row3 = out.row(3);
    const float* cls_row[NUM_CLASSES-1];
    for (int c = 0; c < NUM_CLASSES-1; c++) cls_row[c] = out.row(4 + c);

    for (int i = 0; i < out.w; i++) {
        float cx = row0[i], cy = row1[i], bw = row2[i], bh = row3[i];
        float best_score = -1.f; int best_cls = -1;
        for (int c = 0; c < NUM_CLASSES-1; c++) {
            float s = cls_row[c][i];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < CONF_THRESH || best_cls == 0) continue;

        float x1 = (cx - bw*0.5f - LB_PAD_X) / LB_SCALE + ROI_X0;
        float y1 = (cy - bh*0.5f - LB_PAD_Y) / LB_SCALE + ROI_Y0;
        float x2 = (cx + bw*0.5f - LB_PAD_X) / LB_SCALE + ROI_X0;
        float y2 = (cy + bh*0.5f - LB_PAD_Y) / LB_SCALE + ROI_Y0;

        x1 = std::max(0.f, std::min(x1, (float)CAM_W));
        y1 = std::max(0.f, std::min(y1, (float)CAM_H));
        x2 = std::max(0.f, std::min(x2, (float)CAM_W));
        y2 = std::max(0.f, std::min(y2, (float)CAM_H));
        if (x2 <= x1 || y2 <= y1) continue;

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
    int sndbuf = 2 * 1024 * 1024;  // 2MB
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[ERROR] bind port " << port << " thất bại\n";
        close(fd);
        return -1;
    }
    listen(fd, 4);  // FIX: backlog=4 để không từ chối reconnect
    fcntl(fd, F_SETFL, O_NONBLOCK);
    return fd;
}

// Gửi dữ liệu non-blocking với timeout – KHÔNG block vòng lặp camera
// Trả về false nếu client ngắt hoặc timeout
bool send_all_nb(int sock, const char* data, size_t len)
{
    size_t total = 0;
    while (total < len) {
        // Dùng select với timeout 200ms để không block mãi mãi
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(sock, &wfds);
        struct timeval tv = {0, 200000};  // 200ms timeout
        int sel = select(sock + 1, nullptr, &wfds, nullptr, &tv);
        if (sel <= 0) return false;  // timeout hoặc lỗi

        ssize_t sent = send(sock, data + total, len - total, MSG_NOSIGNAL);
        if (sent <= 0) return false;
        total += sent;
    }
    return true;
}

// ═══════════════════════════════ MJPEG SERVER THREAD ════════════════════════
//
// FIX CORE: Thread này xử lý toàn bộ MJPEG, hoàn toàn độc lập với camera.
// Mỗi client có 1 thread riêng → hỗ trợ nhiều viewer cùng lúc.
// Camera thread KHÔNG BAO GIỜ bị block bởi mạng chậm.

void mjpeg_client_thread(int cli_fd)
{
    // Đặt SO_SNDBUF và TCP_NODELAY cho client này
    int nodelay = 1, sndbuf = 1024 * 1024;
    setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    setsockopt(cli_fd, SOL_SOCKET, SO_SNDBUF,   &sndbuf,  sizeof(sndbuf));

    // Đặt non-blocking socket để send_all_nb hoạt động đúng
    int flags = fcntl(cli_fd, F_GETFL, 0);
    fcntl(cli_fd, F_SETFL, flags | O_NONBLOCK);

    // Đọc HTTP request (bỏ qua nội dung, chỉ cần drain buffer)
    {
        char req[1024] = {};
        recv(cli_fd, req, sizeof(req)-1, MSG_DONTWAIT);
    }

    // FIX: Gửi HTTP header chuẩn RFC 2046 multipart
    // boundary KHÔNG có "--" prefix ở Content-Type header
    // Mỗi part bắt đầu bằng "--BOUNDARY\r\n"
    static const char* HTTP_HDR =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=jpgboundary\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";

    if (!send_all_nb(cli_fd, HTTP_HDR, strlen(HTTP_HDR))) {
        close(cli_fd);
        return;
    }
    std::cout << "[MJPEG] client kết nối fd=" << cli_fd << "\n";

    uint64_t last_seq = 0;
    std::vector<uchar> local_jpeg;

    while (g_state.running.load()) {
        // Chờ frame mới (có timeout để kiểm tra g_state.running)
        {
            std::unique_lock<std::mutex> lk(g_state.frame_mtx);
            bool got = g_state.frame_cv.wait_for(lk,
                std::chrono::milliseconds(500),
                [&]{ return g_state.frame_seq != last_seq; });
            if (!got) continue;  // timeout → thử lại

            last_seq   = g_state.frame_seq;
            local_jpeg = g_state.jpeg_data;  // copy nhanh (vector assignment)
        }

        // FIX: Format đúng chuẩn MJPEG multipart
        // "--boundary\r\n"
        // "Content-Type: image/jpeg\r\n"
        // "Content-Length: <N>\r\n"
        // "\r\n"
        // <JPEG bytes>
        // "\r\n"
        char part_hdr[128];
        int hlen = std::snprintf(part_hdr, sizeof(part_hdr),
            "--jpgboundary\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n"
            "\r\n",
            local_jpeg.size());

        bool ok =
            send_all_nb(cli_fd, part_hdr, hlen) &&
            send_all_nb(cli_fd, (char*)local_jpeg.data(), local_jpeg.size()) &&
            send_all_nb(cli_fd, "\r\n", 2);

        if (!ok) break;  // client ngắt
    }

    std::cout << "[MJPEG] client ngắt fd=" << cli_fd << "\n";
    close(cli_fd);
}

// ═══════════════════════════════ JSON SERVER THREAD ═════════════════════════

void json_client_thread(int cli_fd)
{
    int nodelay = 1;
    setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    int flags = fcntl(cli_fd, F_GETFL, 0);
    fcntl(cli_fd, F_SETFL, flags | O_NONBLOCK);

    {
        char req[1024] = {};
        recv(cli_fd, req, sizeof(req)-1, MSG_DONTWAIT);
    }

    static const char* JSON_HDR =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";

    if (!send_all_nb(cli_fd, JSON_HDR, strlen(JSON_HDR))) {
        close(cli_fd); return;
    }
    std::cout << "[JSON] client kết nối fd=" << cli_fd << "\n";

    // Gửi JSON mỗi khi có inference mới
    // Dùng polling nhẹ thay vì condition_variable để tránh missed signal
    int last_frame = -1;

    while (g_state.running.load()) {
        std::vector<Detection> dets;
        float fps_infer;
        int   frame_count;
        {
            std::lock_guard<std::mutex> lk(g_state.det_mtx);
            frame_count = g_state.frame_count;
            fps_infer   = g_state.fps_infer;
            dets        = g_state.dets;
        }

        if (frame_count == last_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        last_frame = frame_count;

        char js_buf[2048];
        int  js_len = make_json(js_buf, sizeof(js_buf), fps_infer, frame_count, dets);

        char chunk_hdr[16];
        int  hlen = std::snprintf(chunk_hdr, sizeof(chunk_hdr), "%x\r\n", js_len);

        bool ok =
            send_all_nb(cli_fd, chunk_hdr, hlen) &&
            send_all_nb(cli_fd, js_buf,    js_len) &&
            send_all_nb(cli_fd, "\r\n",    2);

        if (!ok) break;
    }

    std::cout << "[JSON] client ngắt fd=" << cli_fd << "\n";
    close(cli_fd);
}

// ═══════════════════════════════ ACCEPT LOOP THREAD ══════════════════════════

void accept_loop(int srv_fd, bool is_mjpeg)
{
    while (g_state.running.load()) {
        struct sockaddr_in cli_addr{};
        socklen_t len = sizeof(cli_addr);
        int cli = accept(srv_fd, (struct sockaddr*)&cli_addr, &len);

        if (cli < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (!g_state.running.load()) break;
            continue;
        }

        // Spawn thread per client
        if (is_mjpeg)
            std::thread(mjpeg_client_thread, cli).detach();
        else
            std::thread(json_client_thread, cli).detach();
    }
}

// ═══════════════════════════════ MAIN ════════════════════════════════════════

int main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);  // FIX: bỏ qua SIGPIPE khi client ngắt

    std::string video_path;
    int mjpeg_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--video" && i+1 < argc) video_path = argv[++i];
        else if (a == "--port"  && i+1 < argc) mjpeg_port = std::stoi(argv[++i]);
        else if (a == "--help") {
            std::cout << "YOLOv8 NCNN Pi4\n\n"
                      << "  ./yolov8_pi4 [--port 8080] [--video <path>]\n";
            return 0;
        }
    }
    int json_port = mjpeg_port + 1;

    // ── Load model ────────────────────────────────────────────────────────────
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads        = 4;
    if (net.load_param(PARAM_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load param\n"; return -1;
    }
    if (net.load_model(BIN_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load bin\n";   return -1;
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
        std::cerr << "[ERROR] Không mở được camera.\n"; return -1;
    }

    // ── TCP servers ───────────────────────────────────────────────────────────
    int mjpeg_srv = make_server(mjpeg_port);
    int json_srv  = make_server(json_port);
    if (mjpeg_srv < 0 || json_srv < 0) return -1;

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

    // ── Khởi động accept threads ──────────────────────────────────────────────
    std::thread mjpeg_accept(accept_loop, mjpeg_srv, true);
    std::thread json_accept (accept_loop, json_srv,  false);
    mjpeg_accept.detach();
    json_accept.detach();

    // ── Vòng lặp camera chính (KHÔNG bao giờ bị block bởi network) ───────────
    cv::Mat frame;
    int frame_count = 0;
    std::vector<Detection> last_dets;

    float fps_infer = 0.f;
    auto  t_infer   = std::chrono::steady_clock::now();

    std::vector<int>   jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 55};
    std::vector<uchar> jpeg_buf;

    while (g_state.running.load()) {
        if (!cap.read(frame) || frame.empty()) break;
        if (frame.cols != CAM_W || frame.rows != CAM_H)
            cv::resize(frame, frame, cv::Size(CAM_W, CAM_H));

        bool do_infer = (frame_count % SKIP_FRAMES == 0);

        if (do_infer) {
            ncnn::Mat in = preprocess(frame);
            ncnn::Extractor ex = net.create_extractor();
            ex.input(INPUT_LAYER, in);
            ncnn::Mat out;
            ex.extract(OUTPUT_LAYER, out);

            last_dets = decode_output(out);

            for (auto& d : last_dets) {
                if (d.class_id == 0) continue;
                if (!is_valid_color(frame, d, CLASS_NAMES[d.class_id]))
                    d.class_id = 7;
            }

            auto  t_now = std::chrono::steady_clock::now();
            float dt    = std::chrono::duration<float>(t_now - t_infer).count();
            t_infer     = t_now;
            fps_infer   = fps_infer * 0.9f + (1.f / (dt + 1e-9f)) * 0.1f;

            // Cập nhật JSON state
            {
                std::lock_guard<std::mutex> lk(g_state.det_mtx);
                g_state.dets        = last_dets;
                g_state.fps_infer   = fps_infer;
                g_state.frame_count = frame_count;
            }
        }

        // FIX CORE: Encode JPEG một lần, đặt vào shared buffer
        // Tất cả MJPEG client threads sẽ tự lấy ra gửi đi
        cv::imencode(".jpg", frame, jpeg_buf, jpeg_params);
        {
            std::lock_guard<std::mutex> lk(g_state.frame_mtx);
            g_state.jpeg_data = jpeg_buf;
            g_state.frame_seq++;
        }
        g_state.frame_cv.notify_all();  // đánh thức tất cả client threads

        frame_count++;

        if (frame_count % 30 == 0) {
            std::cout << "[F" << frame_count << "]"
                      << " infer=" << (int)fps_infer << "fps"
                      << " obj="   << last_dets.size() << "\n";
            std::cout.flush();
        }
    }

    g_state.running = false;
    g_state.frame_cv.notify_all();

    cap.release();
    close(mjpeg_srv);
    close(json_srv);
    std::cout << "[INFO] Done. Frame: " << frame_count << "\n";
    return 0;
}
