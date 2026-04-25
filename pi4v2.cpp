#include <iostream>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <map>
#include <vector>

// Thư viện Driver UART đã viết
#include "Pi4RobotDriver.h"

using json = nlohmann::json;

// ─────────────────────────────────────────
// CẤU HÌNH HỆ THỐNG
// ─────────────────────────────────────────
const std::string DB_URL = "https://canhtayrobot-c4c37-default-rtdb.asia-southeast1.firebasedatabase.app";
const std::string DB_SECRET = "Wwz72xGhJAsO9EO2GSvWHC053GsCXIvRmbDbDKLw";

// Bản đồ ánh xạ giữa ProductID và tên Key trên Firebase
std::map<ProductID, std::string> productNames = {
    {ProductID::HINH_TRU_DO, "HinhTru_Do"},
    {ProductID::LAP_PHUONG_DO, "LapPhuong_Do"},
    {ProductID::HINH_TRU_XANH, "HinhTru_Xanh"},
    {ProductID::LAP_PHUONG_XANH, "LapPhuong_Xanh"},
    {ProductID::HINH_TRU_VANG, "HinhTru_Vang"},
    {ProductID::LAP_PHUONG_VANG, "LapPhuong_Vang"}
};

// Bản đồ ánh xạ Servo Firebase Key -> Channel ESP32
std::map<std::string, uint8_t> servoChannels = {
    {"base", 0}, {"shoulder", 3}, {"elbow", 8}, {"gripper", 15}
};

// ─────────────────────────────────────────
// HÀM HỖ TRỢ HTTP & FIREBASE
// ─────────────────────────────────────────

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Hàm đồng bộ số đếm từ Firebase xuống ESP32 khi khởi động
void syncInitialCounts(RobotDriver& robot) {
    std::cout << "[SYNC] Đang tải dữ liệu counts từ Firebase...\n";
    std::string url = DB_URL + "/counts.json?auth=" + DB_SECRET;
    
    CURL *curl = curl_easy_init();
    if(curl) {
        std::string readBuffer;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        CURLcode res = curl_easy_perform(curl);
        if(res == CURLE_OK && readBuffer != "null") {
            try {
                json j = json::parse(readBuffer);
                for (const auto& [key, val] : j.items()) {
                    for (const auto& [pid, name] : productNames) {
                        if (name == key) {
                            uint16_t countVal = val.get<uint16_t>();
                            std::cout << "  -> Đồng bộ " << name << ": " << countVal << "\n";
                            // Gửi lệnh 0x51 (CMD_SET_COUNT) xuống ESP32 qua UART
                            robot.setInitialCount(pid, countVal); 
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            break;
                        }
                    }
                }
                std::cout << "[SYNC] Hoàn tất đồng bộ dữ liệu xuống ESP32.\n";
            } catch(...) { std::cerr << "[SYNC ERR] Lỗi Parse JSON.\n"; }
        }
        curl_easy_cleanup(curl);
    }
}

// Hàm cập nhật số đếm lên Firebase (Tổng và Ngày)
void updateFirebaseCountDB(ProductID pid, int currentCount) {
    std::string pName = productNames[pid];
    
    // Lấy ngày hiện tại YYYY-MM-DD
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    std::string todayDate = oss.str();

    json patchData;
    patchData[pName] = currentCount;
    std::string payload = patchData.dump();

    CURL *curl = curl_easy_init();
    if(curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        // Gửi lên nhánh /counts
        std::string urlCounts = DB_URL + "/counts.json?auth=" + DB_SECRET;
        curl_easy_setopt(curl, CURLOPT_URL, urlCounts.c_str());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_perform(curl);

        // Gửi lên nhánh /daily_counts/date
        std::string urlDaily = DB_URL + "/daily_counts/" + todayDate + ".json?auth=" + DB_SECRET;
        curl_easy_setopt(curl, CURLOPT_URL, urlDaily.c_str());
        curl_easy_perform(curl);

        std::cout << "[FIREBASE] Đã cập nhật " << pName << " = " << currentCount << "\n";
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

// ─────────────────────────────────────────
// FIREBASE STREAM CALLBACKS
// ─────────────────────────────────────────

// Lắng nghe /status (start, home)
size_t StatusStreamCallback(char *ptr, size_t size, size_t nmemb, void *userdata) {
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
                if (data.contains("home")) robot->setHome(data["home"]);
            } else if (path == "/start") {
                robot->setStart(data.get<bool>());
            } else if (path == "/home") {
                robot->setHome(data.get<bool>());
            }
        } catch(...) {}
    }
    return realSize;
}

// Lắng nghe /RobotControl (mode, conveyor, servos)
size_t ControlStreamCallback(char *ptr, size_t size, size_t nmemb, void *userdata) {
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
                if (data.contains("mode")) robot->setMode(data["mode"] == "manual" ? SystemMode::MANUAL : SystemMode::AUTO);
                if (data.contains("conveyor")) robot->setConveyor(data["conveyor"]);
                for (auto const& [key, ch] : servoChannels) {
                    if (data.contains(key)) robot->setServo(ch, data[key]);
                }
            } else {
                std::string key = path.substr(1);
                if (key == "mode") robot->setMode(data == "manual" ? SystemMode::MANUAL : SystemMode::AUTO);
                else if (key == "conveyor") robot->setConveyor(data);
                else if (servoChannels.count(key)) robot->setServo(servoChannels[key], data);
            }
        } catch(...) {}
    }
    return realSize;
}

void listenToFirebaseStream(std::string node, void* cb, RobotDriver* robot) {
    std::string url = DB_URL + node + ".json?auth=" + DB_SECRET;
    CURL *curl = curl_easy_init();
    if (curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Accept: text/event-stream");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, robot);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
        curl_easy_perform(curl);
        curl_cleanup: curl_slist_free_all(headers); curl_easy_cleanup(curl);
    }
}

// ─────────────────────────────────────────
// MAIN PROGRAM
// ─────────────────────────────────────────

int main() {
    std::cout << "--- KHOI DONG HE THONG DIEU KHIEN ROBOT ARM ---\n";

    // Khởi tạo Driver UART
    RobotDriver robot("/dev/ttyS0", 115200);
    if (!robot.isOpen()) {
        std::cerr << "[ERR] Khong the mo UART.\n";
        return -1;
    }

    curl_global_init(CURL_GLOBAL_ALL);

    // 1. Đồng bộ dữ liệu ngược từ Firebase -> Pi4 -> ESP32
    syncInitialCounts(robot);

    // 2. Cài đặt Callbacks nhận từ ESP32
    robot.onSensor = [](bool detected) {
        if (detected) std::cout << "[MAIN] Phat hien vat pham! Chuan bi phan loai...\n";
    };

    robot.onCount = [](ProductID pid, int count) {
        // count là giá trị tổng đã cộng dồn (vì ESP32 đã được đồng bộ số cũ)
        std::cout << "[EVENT] San pham moi! Tong so hien tai: " << count << "\n";
        // Day len Firebase
        std::thread(updateFirebaseCountDB, pid, count).detach();
    };

    // 3. Chạy các luồng lắng nghe Firebase
    std::thread tStatus(listenToFirebaseStream, "/status", (void*)StatusStreamCallback, &robot);
    std::thread tControl(listenToFirebaseStream, "/RobotControl", (void*)ControlStreamCallback, &robot);

    std::cout << "[INFO] He thong dang chay...\n";
    
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    curl_global_cleanup();
    return 0;
}