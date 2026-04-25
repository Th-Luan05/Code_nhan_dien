#include "Pi4RobotDriver.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

// ─────────────────────────────────────────
// CONSTRUCTOR — Mở cổng Serial & cấu hình
// ─────────────────────────────────────────
RobotDriver::RobotDriver(const std::string& port, int baud) {
    // FIX #1: Tránh dùng /dev/ttyS0 (mini-UART bị Bluetooth chiếm trên Pi4)
    // Dùng /dev/ttyAMA0 (PL011 UART đầy đủ) hoặc /dev/ttyUSB0
    fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        std::cerr << "[ERR] Không thể mở cổng " << port << "\n"
                  << "      Thử: sudo usermod -a -G dialout $USER\n"
                  << "      Hoặc kiểm tra dtoverlay=disable-bt trong /boot/config.txt\n";
        return;
    }

    // Cấu hình UART: 115200 8N1
    struct termios options;
    if (tcgetattr(fd, &options) < 0) {
        std::cerr << "[ERR] Không thể lấy cấu hình UART\n";
        return;
    }

    // Thiết lập Baudrate
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    // Thiết lập chế độ RAW (Cực kỳ quan trọng để gửi byte binary)
    // Loại bỏ các tính năng xử lý văn bản (echo, newline, parity...)
    options.c_cflag |= (CLOCAL | CREAD);    // Cho phép đọc, bỏ qua modem lines
    options.c_cflag &= ~PARENB;             // No parity
    options.c_cflag &= ~CSTOPB;             // 1 stop bit
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;                 // 8 bits
    options.c_cflag &= ~CRTSCTS;            // No flow control

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Raw input
    // Dòng code đúng
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    options.c_oflag &= ~OPOST;              // Raw output (Không sửa byte đầu ra)

    options.c_cc[VMIN]  = 0;
    options.c_cc[VTIME] = 1; // 0.1s timeout

    tcsetattr(fd, TCSANOW, &options);
    
    // Đợi một chút để phần cứng ổn định
    usleep(100000); 
    tcflush(fd, TCIOFLUSH);

    running = true;
    rxThread = std::thread(&RobotDriver::rxLoop, this);

    std::cout << "[UART] Đã mở cổng " << port << " @ 115200\n";
}

// ─────────────────────────────────────────
// DESTRUCTOR
// ─────────────────────────────────────────
RobotDriver::~RobotDriver() {
    running = false;
    if (rxThread.joinable()) rxThread.join();
    if (fd != -1) {
        tcflush(fd, TCIOFLUSH);
        close(fd);
    }
}

// ─────────────────────────────────────────
// SEND RAW FRAME
// Frame: STX | LEN | CMD | PAYLOAD... | XOR | ETX
// ─────────────────────────────────────────
void RobotDriver::sendRaw(uint8_t cmd, const std::vector<uint8_t>& payload) {
    if (fd == -1) {
        std::cerr << "[ERR] sendRaw: cổng chưa được mở\n";
        return;
    }

    std::lock_guard<std::mutex> lock(txMutex);

    uint8_t len      = static_cast<uint8_t>(payload.size());
    uint8_t xorCalc  = cmd;   // Bắt đầu XOR từ CMD — khớp với ESP32

    std::vector<uint8_t> frame;
    frame.reserve(5 + len);
    frame.push_back(STX);
    frame.push_back(len);
    frame.push_back(cmd);
    for (uint8_t b : payload) {
        frame.push_back(b);
        xorCalc ^= b;
    }
    frame.push_back(xorCalc);
    frame.push_back(ETX);
    std::cout << "[DEBUG TX] Raw bytes: ";
    for(auto b : frame) printf("%02X ", b);
    std::cout << std::endl;
    ssize_t written = write(fd, frame.data(), frame.size());
    if (written < 0) {
        std::cerr << "[ERR] Ghi UART thất bại (cmd=0x"
                  << std::hex << (int)cmd << std::dec << ")\n";
    } else {
        std::cout << "[UART TX] CMD=0x" << std::hex << (int)cmd
                  << " | len=" << std::dec << (int)len << "\n";
    }
}

// ─────────────────────────────────────────
// RX LOOP (chạy trên luồng riêng)
// ─────────────────────────────────────────
void RobotDriver::rxLoop() {
    uint8_t buf[256];
    std::vector<uint8_t> rxBuf;

    while (running) {
        int n = read(fd, buf, sizeof(buf));
        if (n <= 0) continue; // timeout hoặc không có dữ liệu

        rxBuf.insert(rxBuf.end(), buf, buf + n);

        // Bóc tách frame liên tục từ buffer
        while (rxBuf.size() >= 5) {
            // Tìm STX
            auto it = std::find(rxBuf.begin(), rxBuf.end(), STX);
            if (it != rxBuf.begin()) {
                // Bỏ byte rác trước STX
                rxBuf.erase(rxBuf.begin(), it);
            }
            if (rxBuf.size() < 5) break;

            uint8_t len          = rxBuf[1];
            size_t  expectedSize = 5 + len; // STX+LEN+CMD+PAYLOAD+XOR+ETX = 5+len (không tính payload 2 lần)
            // Chuẩn: STX(1) + LEN(1) + CMD(1) + PAYLOAD(len) + XOR(1) + ETX(1) = len+5

            if (rxBuf.size() < expectedSize) break; // Chờ thêm dữ liệu

            // Kiểm tra ETX ở vị trí cuối
            if (rxBuf[expectedSize - 1] != ETX) {
                // Frame sai — dịch 1 byte tìm STX tiếp theo
                rxBuf.erase(rxBuf.begin());
                continue;
            }

            uint8_t cmd = rxBuf[2];
            std::vector<uint8_t> payload(rxBuf.begin() + 3, rxBuf.begin() + 3 + len);
            uint8_t rxXor = rxBuf[3 + len];

            // Verify checksum XOR (bắt đầu từ CMD)
            uint8_t calcXor = cmd;
            for (uint8_t b : payload) calcXor ^= b;

            if (calcXor == rxXor) {
                processFrame(cmd, payload);
            } else {
                std::cerr << "[WARN] Sai Checksum RX — CMD=0x" << std::hex << (int)cmd
                          << " rxXor=0x" << (int)rxXor << " calcXor=0x" << (int)calcXor
                          << std::dec << "\n";
            }

            // Xóa frame đã xử lý
            rxBuf.erase(rxBuf.begin(), rxBuf.begin() + expectedSize);
        }

        // Giới hạn kích thước buffer để tránh memory leak
        if (rxBuf.size() > 1024) {
            std::cerr << "[WARN] RX buffer tràn — xóa\n";
            rxBuf.clear();
        }
    }
}

// ─────────────────────────────────────────
// PROCESS FRAME NHẬN ĐƯỢC
// ─────────────────────────────────────────
void RobotDriver::processFrame(uint8_t cmd, const std::vector<uint8_t>& payload) {
    switch (cmd) {

        case CMD_ACK:
            if (!payload.empty()) {
                std::cout << "[UART RX] ACK cho CMD=0x" << std::hex << (int)payload[0] << std::dec << "\n";
                if (onAck) onAck(payload[0]);
            }
            break;

        case CMD_NACK:
            if (payload.size() >= 2) {
                std::cerr << "[UART RX] NACK CMD=0x" << std::hex << (int)payload[0]
                          << " err=0x" << (int)payload[1] << std::dec << "\n";
                if (onNack) onNack(payload[0], payload[1]);
            }
            break;

        case CMD_STATUS:
            if (payload.size() >= 4) {
                status.mode     = static_cast<SystemMode>(payload[0]);
                status.start    = payload[1] != 0;
                status.home     = payload[2] != 0;
                status.conveyor = payload[3] != 0;
                std::cout << "[STATUS] mode=" << (int)payload[0]
                          << " start=" << status.start
                          << " home=" << status.home
                          << " conveyor=" << status.conveyor << "\n";
            }
            break;

        case CMD_SENSOR:
            if (!payload.empty()) {
                status.sensor = payload[0] != 0;
                std::cout << "[SENSOR] detected=" << status.sensor << "\n";
                if (onSensor) onSensor(status.sensor);
            }
            break;

        case CMD_COUNT:
            if (payload.size() >= 3) {
                ProductID pid = static_cast<ProductID>(payload[0]);
                int count     = (payload[1] << 8) | payload[2];
                status.counts[pid] = count;
                std::cout << "[COUNT] productId=" << (int)payload[0] << " count=" << count << "\n";
                if (onCount) onCount(pid, count);
            }
            break;

        case CMD_PONG:
            std::cout << "[UART RX] PONG nhận — ESP32 đang sống!\n";
            break;

        case CMD_DONE:
            if (!payload.empty()) {
                std::cout << "[UART RX] DONE cho CMD=0x" << std::hex << (int)payload[0] << std::dec << "\n";
                if (onDone) onDone(payload[0]);
            }
            break;

        case CMD_SERVO_POS:
            // Payload: [ch, angle] pairs
            for (size_t i = 0; i + 1 < payload.size(); i += 2) {
                status.servo_pos[payload[i]] = payload[i+1];
            }
            break;

        default:
            std::cerr << "[WARN] CMD không xác định: 0x" << std::hex << (int)cmd << std::dec << "\n";
            break;
    }
}

// ─────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────
bool RobotDriver::ping() {
    sendRaw(CMD_PING);
    return true;
}

void RobotDriver::setMode(SystemMode mode) {
    sendRaw(CMD_SET_MODE, {static_cast<uint8_t>(mode)});
}

void RobotDriver::setStart(bool on) {
    sendRaw(CMD_SET_START, {static_cast<uint8_t>(on ? 1 : 0)});
}

void RobotDriver::setHome(bool on) {
    sendRaw(CMD_SET_HOME, {static_cast<uint8_t>(on ? 1 : 0)});
}

void RobotDriver::setConveyor(bool on) {
    sendRaw(CMD_SET_CONVEYOR, {static_cast<uint8_t>(on ? 1 : 0)});
}

void RobotDriver::classify(ProductID pid) {
    sendRaw(CMD_CLASSIFY, {static_cast<uint8_t>(pid)});
}

void RobotDriver::resetCount() {
    sendRaw(CMD_RESET_COUNT);
}
void RobotDriver::setServo(uint8_t channel, uint8_t angle) {
    // ESP32 yêu cầu payload dạng cặp [channel, angle]
    sendRaw(CMD_SET_SERVO, {channel, angle});
}
void RobotDriver::setInitialCount(ProductID pid, uint16_t count) {
    std::vector<uint8_t> payload;
    payload.push_back(static_cast<uint8_t>(pid));
    payload.push_back(static_cast<uint8_t>(count >> 8));   // Byte cao
    payload.push_back(static_cast<uint8_t>(count & 0xFF)); // Byte thấp
    sendRaw(0x51, payload);
}