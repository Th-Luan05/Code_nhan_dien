#ifndef PI4_ROBOT_DRIVER_H
#define PI4_ROBOT_DRIVER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>
#include <map>
#include <atomic>
#include <algorithm>  // std::find

// ─────────────────────────────────────────
// PROTOCOL CONSTANTS
// ─────────────────────────────────────────
constexpr uint8_t STX = 0x02;
constexpr uint8_t ETX = 0x03;

// CMD Pi4 -> ESP32
constexpr uint8_t CMD_SET_MODE      = 0x10;
constexpr uint8_t CMD_SET_START     = 0x11;
constexpr uint8_t CMD_SET_HOME      = 0x12;
constexpr uint8_t CMD_SET_SERVO     = 0x20;
constexpr uint8_t CMD_SET_CONVEYOR  = 0x21;
constexpr uint8_t CMD_CLASSIFY      = 0x30;
constexpr uint8_t CMD_PING          = 0x40;
constexpr uint8_t CMD_RESET_COUNT   = 0x50;
constexpr uint8_t CMD_SET_COUNT_INIT = 0x51;

// CMD ESP32 -> Pi4
constexpr uint8_t CMD_ACK           = 0x80;
constexpr uint8_t CMD_NACK          = 0x81;
constexpr uint8_t CMD_STATUS        = 0x82;
constexpr uint8_t CMD_SERVO_POS     = 0x83;
constexpr uint8_t CMD_SENSOR        = 0x84;
constexpr uint8_t CMD_COUNT         = 0x85;
constexpr uint8_t CMD_PONG          = 0x86;
constexpr uint8_t CMD_DONE          = 0x87;

enum class SystemMode { AUTO = 0, MANUAL = 1 };

enum class ProductID {
    HINH_TRU_DO    = 0,
    LAP_PHUONG_DO  = 1,
    HINH_TRU_XANH  = 2,
    LAP_PHUONG_XANH = 3,
    HINH_TRU_VANG  = 4,
    LAP_PHUONG_VANG = 5
};

struct RobotStatus {
    SystemMode mode     = SystemMode::AUTO;
    bool start          = false;
    bool home           = false;
    bool conveyor       = false;
    bool sensor         = false;
    std::map<int, int> servo_pos = {{0,0},{3,0},{8,0},{15,0}};
    std::map<ProductID, int> counts;
};

class RobotDriver {
public:
    // FIX #1: Thêm tham số port rõ ràng, mặc định /dev/ttyS0 (không phải ttyAMA0)
    RobotDriver(const std::string& port = "/dev/ttyS0", int baud = 115200);
    ~RobotDriver();

    // Public API
    bool ping();
    void setMode(SystemMode mode);
    void setStart(bool on);
    void setHome(bool on);
    void setConveyor(bool on);
    void setServo(uint8_t channel, uint8_t angle);
    void classify(ProductID pid);
    void resetCount();
    void setInitialCount(ProductID pid, uint16_t count);
    

    // Callbacks từ ESP32
    std::function<void(bool)>          onSensor;
    std::function<void(ProductID,int)> onCount;
    std::function<void(uint8_t)>       onDone;
    std::function<void(uint8_t)>       onAck;
    std::function<void(uint8_t,uint8_t)> onNack;

    RobotStatus status;
    bool isOpen() const { return fd != -1; }

private:
    int fd = -1;
    std::atomic<bool> running{false};
    std::thread rxThread;
    std::mutex txMutex;

    void rxLoop();
    void processFrame(uint8_t cmd, const std::vector<uint8_t>& payload);
    void sendRaw(uint8_t cmd, const std::vector<uint8_t>& payload = {});
};

#endif // PI4_ROBOT_DRIVER_H