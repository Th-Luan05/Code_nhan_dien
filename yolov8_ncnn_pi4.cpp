/*
 * ═══════════════════════════════════════════════════════════════════════════
 *  main_integrated.cpp
 *  Tích hợp:
 *    - test.cpp   : YOLOv8 NCNN inference, MJPEG stream, JSON stream
 *    - pi4v2.cpp  : Firebase Realtime DB, RobotDriver (UART → ESP32)
 *
 *  Luồng hoạt động:
 *    Camera → YOLO → Detection → RobotDriver (UART/ESP32)
 *                             → Firebase (counts, daily_counts)
 *                             → MJPEG stream (port 8080)
 *                             → JSON  stream (port 8081)
 *    Firebase → Pi4 (status, RobotControl) → RobotDriver
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ── Standard / System ────────────────────────────────────────────────────────
#include <termios.h>
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
#include <ctime>
#include <iomanip>
#include <sstream>
#include <map>

// ── Network ──────────────────────────────────────────────────────────────────
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

// ── Third-party ───────────────────────────────────────────────────────────────
#include "net.h"                    // ncnn
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// ── Project ───────────────────────────────────────────────────────────────────
#include "Pi4RobotDriver.h"         // RobotDriver, ProductID, SystemMode

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════════
//  CẤU HÌNH FIREBASE
// ═══════════════════════════════════════════════════════════════════════════════

static const std::string DB_URL    = "https://canhtayrobot-c4c37-default-rtdb.asia-southeast1.firebasedatabase.app";
static const std::string DB_SECRET = "Wwz72xGhJAsO9EO2GSvWHC053GsCXIvRmbDbDKLw";

// Ánh xạ ProductID ↔ tên key Firebase
static std::map<ProductID, std::string> productNames = {
    {ProductID::HINH_TRU_DO,      "HinhTru_Do"},
    {ProductID::LAP_PHUONG_DO,    "LapPhuong_Do"},
    {ProductID::HINH_TRU_XANH,   "HinhTru_Xanh"},
    {ProductID::LAP_PHUONG_XANH, "LapPhuong_Xanh"},
    {ProductID::HINH_TRU_VANG,   "HinhTru_Vang"},
    {ProductID::LAP_PHUONG_VANG, "LapPhuong_Vang"}
};

// Ánh xạ servo key Firebase → channel ESP32
static std::map<std::string, uint8_t> servoChannels = {
    {"base", 0}, {"shoulder", 3}, {"elbow", 8}, {"gripper", 15}
};

// ═══════════════════════════════════════════════════════════════════════════════
//  CẤU HÌNH CAMERA & MODEL
// ═══════════════════════════════════════════════════════════════════════════════

static const std::string PARAM_PATH = "best_ncnn_model/model.ncnn.param";
static const std::string BIN_PATH   = "best_ncnn_model/model.ncnn.bin";
static const char* INPUT_LAYER      = "in0";
static const char* OUTPUT_LAYER     = "out0";

static const int CAM_W = 640;
static const int CAM_H = 360;

static const int ROI_W  = 180;
static const int ROI_H  = 200;
static const int ROI_X0 = ((CAM_W - ROI_W) / 2)-30;
static const int ROI_Y0 = ((CAM_H - ROI_H) / 2);

static const int NET_SIZE = 320;

static float LB_SCALE = 1.f;
static int   LB_PAD_X = 0;
static int   LB_PAD_Y = 0;
static int   LB_NW    = 0;
static int   LB_NH    = 0;

static const int   NUM_CLASSES = 8;
static const float CONF_THRESH = 0.8f;
static const float NMS_THRESH  = 0.45f;
static const int   SKIP_FRAMES = 1;

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

// ═══════════════════════════════════════════════════════════════════════════════
//  STRUCT
// ═══════════════════════════════════════════════════════════════════════════════

struct Detection {
    int   class_id;
    float conf;
    int   x, y, w, h;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  SHARED STATE (camera ↔ MJPEG/JSON threads)
// ═══════════════════════════════════════════════════════════════════════════════

struct SharedState {
    // MJPEG
    std::mutex              frame_mtx;
    std::condition_variable frame_cv;
    std::vector<uchar>      jpeg_data;
    uint64_t                frame_seq = 0;

    // JSON / detections
    std::mutex              det_mtx;
    std::vector<Detection>  dets;
    float                   fps_infer  = 0.f;
    int                     frame_count = 0;

    std::atomic<bool>       running{true};
} g_state;

// ═══════════════════════════════════════════════════════════════════════════════
//  LETTERBOX INIT
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  COLOR CHECK
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  TIỀN XỬ LÝ ẢNH
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  GIẢI MÃ ĐẦU RA YOLO
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  ÁNH XẠ class_id → ProductID cho RobotDriver
// ═══════════════════════════════════════════════════════════════════════════════

// CLASS_NAMES: 0=Background,1=LapPhuong_Do,2=LapPhuong_Vang,3=LapPhuong_Xanh
//              4=HinhTru_Do,5=HinhTru_Vang,6=HinhTru_Xanh,7=SP_loi
static const std::map<int, ProductID> classIdToProductID = {
    {1, ProductID::LAP_PHUONG_DO},
    {2, ProductID::LAP_PHUONG_VANG},
    {3, ProductID::LAP_PHUONG_XANH},
    {4, ProductID::HINH_TRU_DO},
    {5, ProductID::HINH_TRU_VANG},
    {6, ProductID::HINH_TRU_XANH}
    // 0=Background, 7=SP_loi → không gửi
};

// ═══════════════════════════════════════════════════════════════════════════════
//  FIREBASE – HTTP HELPER
// ═══════════════════════════════════════════════════════════════════════════════

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Đồng bộ counts từ Firebase → ESP32 lúc khởi động
void syncInitialCounts(RobotDriver& robot)
{
    std::cout << "[SYNC] Đang tải dữ liệu counts từ Firebase...\n";
    std::string url = DB_URL + "/counts.json?auth=" + DB_SECRET;

    CURL* curl = curl_easy_init();
    if (curl) {
        std::string buf;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK && buf != "null") {
            try {
                json j = json::parse(buf);
                for (const auto& [key, val] : j.items()) {
                    for (const auto& [pid, name] : productNames) {
                        if (name == key) {
                            uint16_t countVal = val.get<uint16_t>();
                            std::cout << "  -> Đồng bộ " << name << ": " << countVal << "\n";
                            robot.setInitialCount(pid, countVal);
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            break;
                        }
                    }
                }
                std::cout << "[SYNC] Hoàn tất đồng bộ dữ liệu xuống ESP32.\n";
            } catch (...) { std::cerr << "[SYNC ERR] Lỗi Parse JSON.\n"; }
        }
        curl_easy_cleanup(curl);
    }
}

// Cập nhật counts lên Firebase (tổng + daily)
void updateFirebaseCountDB(ProductID pid, int currentCount)
{
    std::string pName = productNames[pid];

    auto t  = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    std::string todayDate = oss.str();

    json patchData;
    patchData[pName] = currentCount;
    std::string payload = patchData.dump();

    CURL* curl = curl_easy_init();
    if (curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        // /counts
        std::string urlCounts = DB_URL + "/counts.json?auth=" + DB_SECRET;
        curl_easy_setopt(curl, CURLOPT_URL, urlCounts.c_str());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_perform(curl);

        // /daily_counts/<date>
        std::string urlDaily = DB_URL + "/daily_counts/" + todayDate + ".json?auth=" + DB_SECRET;
        curl_easy_setopt(curl, CURLOPT_URL, urlDaily.c_str());
        curl_easy_perform(curl);

        std::cout << "[FIREBASE] Đã cập nhật " << pName << " = " << currentCount << "\n";
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FIREBASE STREAM CALLBACKS
// ═══════════════════════════════════════════════════════════════════════════════

static size_t StatusStreamCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    size_t realSize = size * nmemb;
    std::string chunk(ptr, realSize);
    RobotDriver* robot = static_cast<RobotDriver*>(userdata);

    size_t dataPos = chunk.find("data: ");
    if (dataPos != std::string::npos) {
        try {
            std::string jsonData = chunk.substr(dataPos + 6);
            if (jsonData.find("null") == 0) return realSize;
            json j = json::parse(jsonData);
            std::string path = j["path"];
            json data = j["data"];

            if (path == "/") {
                if (data.contains("start")) robot->setStart(data["start"]);
                if (data.contains("home"))  robot->setHome(data["home"]);
            } else if (path == "/start") {
                robot->setStart(data.get<bool>());
            } else if (path == "/home") {
                robot->setHome(data.get<bool>());
            }
        } catch (...) {}
    }
    return realSize;
}

static size_t ControlStreamCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    size_t realSize = size * nmemb;
    std::string chunk(ptr, realSize);
    RobotDriver* robot = static_cast<RobotDriver*>(userdata);

    size_t dataPos = chunk.find("data: ");
    if (dataPos != std::string::npos) {
        try {
            std::string jsonData = chunk.substr(dataPos + 6);
            if (jsonData.find("null") == 0) return realSize;
            json j = json::parse(jsonData);
            std::string path = j["path"];
            json data = j["data"];

            if (path == "/") {
                if (data.contains("mode"))
                    robot->setMode(data["mode"] == "manual" ? SystemMode::MANUAL : SystemMode::AUTO);
                if (data.contains("conveyor"))
                    robot->setConveyor(data["conveyor"]);
                for (auto const& [key, ch] : servoChannels) {
                    if (data.contains(key)) robot->setServo(ch, data[key]);
                }
            } else {
                std::string key = path.substr(1);
                if      (key == "mode")
                    robot->setMode(data == "manual" ? SystemMode::MANUAL : SystemMode::AUTO);
                else if (key == "conveyor")
                    robot->setConveyor(data);
                else if (servoChannels.count(key))
                    robot->setServo(servoChannels[key], data);
            }
        } catch (...) {}
    }
    return realSize;
}

void listenToFirebaseStream(const std::string& node, size_t (*cb)(char*, size_t, size_t, void*),
                            RobotDriver* robot)
{
    std::string url = DB_URL + node + ".json?auth=" + DB_SECRET;
    CURL* curl = curl_easy_init();
    if (curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Accept: text/event-stream");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, robot);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
        curl_easy_perform(curl);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  JSON BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

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
        const char* name = "";
        if (d.class_id > 0 && d.class_id < (int)CLASS_NAMES.size()) {
            if (CLASS_NAMES[d.class_id] != "SP_loi")
                name = CLASS_NAMES[d.class_id].c_str();
        }
        n += std::snprintf(buf + n, bufsz - n,
            "%s{\"cls\":%d,\"name\":\"%s\","
            "\"conf\":%d,\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d}",
            i ? "," : "",
            d.class_id, name,
            (int)(d.conf * 100),
            d.x, d.y, d.w, d.h);
    }
    n += std::snprintf(buf + n, bufsz - n, "]}\n");
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SOCKET HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

int make_server(int port)
{
    int fd  = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    int sndbuf = 2 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[ERROR] bind port " << port << " thất bại\n";
        close(fd); return -1;
    }
    listen(fd, 4);
    fcntl(fd, F_SETFL, O_NONBLOCK);
    return fd;
}

bool send_all_nb(int sock, const char* data, size_t len)
{
    size_t total = 0;
    while (total < len) {
        fd_set wfds; FD_ZERO(&wfds); FD_SET(sock, &wfds);
        struct timeval tv = {0, 200000};
        int sel = select(sock + 1, nullptr, &wfds, nullptr, &tv);
        if (sel <= 0) return false;
        ssize_t sent = send(sock, data + total, len - total, MSG_NOSIGNAL);
        if (sent <= 0) return false;
        total += sent;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MJPEG CLIENT THREAD
// ═══════════════════════════════════════════════════════════════════════════════

void mjpeg_client_thread(int cli_fd)
{
    int nodelay = 1, sndbuf = 1024 * 1024;
    setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    setsockopt(cli_fd, SOL_SOCKET,  SO_SNDBUF,   &sndbuf,  sizeof(sndbuf));
    int flags = fcntl(cli_fd, F_GETFL, 0);
    fcntl(cli_fd, F_SETFL, flags | O_NONBLOCK);

    { char req[1024] = {}; recv(cli_fd, req, sizeof(req)-1, MSG_DONTWAIT); }

    static const char* HTTP_HDR =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=jpgboundary\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";

    if (!send_all_nb(cli_fd, HTTP_HDR, strlen(HTTP_HDR))) { close(cli_fd); return; }
    std::cout << "[MJPEG] client kết nối fd=" << cli_fd << "\n";

    uint64_t           last_seq = 0;
    std::vector<uchar> local_jpeg;

    while (g_state.running.load()) {
        {
            std::unique_lock<std::mutex> lk(g_state.frame_mtx);
            bool got = g_state.frame_cv.wait_for(lk,
                std::chrono::milliseconds(500),
                [&]{ return g_state.frame_seq != last_seq; });
            if (!got) continue;
            last_seq   = g_state.frame_seq;
            local_jpeg = g_state.jpeg_data;
        }

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
        if (!ok) break;
    }

    std::cout << "[MJPEG] client ngắt fd=" << cli_fd << "\n";
    close(cli_fd);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  JSON CLIENT THREAD
// ═══════════════════════════════════════════════════════════════════════════════

void json_client_thread(int cli_fd)
{
    int nodelay = 1;
    setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    int flags = fcntl(cli_fd, F_GETFL, 0);
    fcntl(cli_fd, F_SETFL, flags | O_NONBLOCK);

    { char req[1024] = {}; recv(cli_fd, req, sizeof(req)-1, MSG_DONTWAIT); }

    static const char* JSON_HDR =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";

    if (!send_all_nb(cli_fd, JSON_HDR, strlen(JSON_HDR))) { close(cli_fd); return; }
    std::cout << "[JSON] client kết nối fd=" << cli_fd << "\n";

    int last_frame = -1;
    while (g_state.running.load()) {
        std::vector<Detection> dets;
        float fps_infer; int frame_count;
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

// ═══════════════════════════════════════════════════════════════════════════════
//  ACCEPT LOOP THREAD
// ═══════════════════════════════════════════════════════════════════════════════

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
        if (is_mjpeg)
            std::thread(mjpeg_client_thread, cli).detach();
        else
            std::thread(json_client_thread, cli).detach();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);

    std::string video_path;
    int mjpeg_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--video" && i+1 < argc) video_path = argv[++i];
        else if (a == "--port"  && i+1 < argc) mjpeg_port = std::stoi(argv[++i]);
        else if (a == "--help") {
            std::cout << "YOLOv8 NCNN + Firebase + RobotDriver\n\n"
                      << "  ./robot [--port 8080] [--video <path>]\n";
            return 0;
        }
    }
    int json_port = mjpeg_port + 1;

    std::cout << "--- KHOI DONG HE THONG ---\n";

    // ── 1. Khởi tạo RobotDriver (thay thế uart_init cũ) ─────────────────────
    RobotDriver robot("/dev/ttyS0", 115200);
    if (!robot.isOpen()) {
        std::cerr << "[ERR] Không thể mở UART / RobotDriver.\n";
        return -1;
    }

    // ── 2. Khởi tạo libcurl ──────────────────────────────────────────────────
    curl_global_init(CURL_GLOBAL_ALL);

    // ── 3. Đồng bộ counts Firebase → ESP32 ──────────────────────────────────
    syncInitialCounts(robot);

    // ── 4. Cài đặt callbacks nhận từ ESP32 ──────────────────────────────────
    robot.onSensor = [](bool detected) {
        if (detected)
            std::cout << "[MAIN] Phát hiện vật phẩm! Chuẩn bị phân loại...\n";
    };

    robot.onCount = [](ProductID pid, int count) {
        std::cout << "[EVENT] Sản phẩm mới! Tổng số hiện tại: " << count << "\n";
        // Đẩy lên Firebase trên thread riêng để không block camera
        std::thread(updateFirebaseCountDB, pid, count).detach();
    };

    // ── 5. Load model NCNN ───────────────────────────────────────────────────
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads        = 4;
    if (net.load_param(PARAM_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load param\n"; return -1;
    }
    if (net.load_model(BIN_PATH.c_str()) != 0) {
        std::cerr << "[ERROR] load bin\n"; return -1;
    }
    std::cout << "[INFO] Model OK\n";
    init_letterbox();

    // ── 6. Mở camera ────────────────────────────────────────────────────────
    cv::VideoCapture cap;
    if (video_path.empty()) {
        std::string gst =
            "v4l2src device=/dev/video0 ! "
            "image/jpeg,width=" + std::to_string(CAM_W) +
            ",height=" + std::to_string(CAM_H) +
            ",framerate=10/1 ! "
            "jpegdec ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false";
        cap.open(gst, cv::CAP_GSTREAMER);

        if (!cap.isOpened()) {
            std::cout << "[WARN] GStreamer lỗi → chuyển sang V4L2\n";
            cap.open("/dev/video0", cv::CAP_V4L2);
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
        std::cerr << "[ERROR] Không mở được camera.\n";
        return -1;
    }

    // ── 7. TCP servers ───────────────────────────────────────────────────────
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

    // ── 8. Luồng lắng nghe Firebase (status + RobotControl) ─────────────────
    std::thread tStatus([&robot]{
        listenToFirebaseStream("/status", StatusStreamCallback, &robot);
    });
    std::thread tControl([&robot]{
        listenToFirebaseStream("/RobotControl", ControlStreamCallback, &robot);
    });
    tStatus.detach();
    tControl.detach();

    // ── 9. Accept threads cho MJPEG / JSON ───────────────────────────────────
    std::thread mjpeg_accept(accept_loop, mjpeg_srv, true);
    std::thread json_accept (accept_loop, json_srv,  false);
    mjpeg_accept.detach();
    json_accept.detach();

    // ── 10. Vòng lặp camera chính ────────────────────────────────────────────
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
            // ── Inference ────────────────────────────────────────────────────
            ncnn::Mat in = preprocess(frame);
            ncnn::Extractor ex = net.create_extractor();
            ex.input(INPUT_LAYER, in);
            ncnn::Mat out;
            ex.extract(OUTPUT_LAYER, out);

            last_dets = decode_output(out);

            // ── Lọc màu ──────────────────────────────────────────────────────
            std::vector<Detection> filtered;
            for (auto& d : last_dets) {
                if (d.class_id == 0) continue;
                if (!is_valid_color(frame, d, CLASS_NAMES[d.class_id])) continue;
                filtered.push_back(d);
            }
            last_dets = filtered;

            // ── FPS ───────────────────────────────────────────────────────────
            auto  t_now = std::chrono::steady_clock::now();
            float dt    = std::chrono::duration<float>(t_now - t_infer).count();
            t_infer     = t_now;
            fps_infer   = fps_infer * 0.9f + (1.f / (dt + 1e-9f)) * 0.1f;

            // ── Cập nhật JSON state ───────────────────────────────────────────
            {
                std::lock_guard<std::mutex> lk(g_state.det_mtx);
                g_state.dets        = last_dets;
                g_state.fps_infer   = fps_infer;
                g_state.frame_count = frame_count;
            }

            // ── Gửi detection tốt nhất xuống ESP32 qua RobotDriver ───────────
            //    (thay thế uart_send() cũ – dùng notifyDetection của Driver)
            if (!last_dets.empty()) {
                const Detection& best = *std::max_element(
                    last_dets.begin(), last_dets.end(),
                    [](const Detection& a, const Detection& b){ return a.conf < b.conf; });

                auto it = classIdToProductID.find(best.class_id);
                if (it != classIdToProductID.end()) {
                    robot.classify(it->second);
                }
            }
            // Không có detection → không gọi classify
        }

        // ── Encode JPEG và cập nhật shared buffer ────────────────────────────
        cv::imencode(".jpg", frame, jpeg_buf, jpeg_params);
        {
            std::lock_guard<std::mutex> lk(g_state.frame_mtx);
            g_state.jpeg_data = jpeg_buf;
            g_state.frame_seq++;
        }
        g_state.frame_cv.notify_all();

        frame_count++;

        if (frame_count % 30 == 0) {
            std::cout << "[F" << frame_count << "]"
                      << " infer=" << (int)fps_infer << "fps"
                      << " obj="   << last_dets.size() << "\n";
            std::cout.flush();
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    g_state.running = false;
    g_state.frame_cv.notify_all();

    cap.release();
    close(mjpeg_srv);
    close(json_srv);
    curl_global_cleanup();

    std::cout << "[INFO] Done. Frame: " << frame_count << "\n";
    return 0;
}