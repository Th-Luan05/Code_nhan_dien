#pragma once
#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <string>

class Camera {
public:
    Camera() = default;
    ~Camera() { stop(); }

    bool start(int width = 1280, int height = 720, int fps = 30);
    void stop();

    // Lấy frame mới nhất dạng BGR cv::Mat
    bool get_frame(cv::Mat& out);

private:
    GstElement*  pipeline_  = nullptr;
    GstElement*  appsink_   = nullptr;

    int width_  = 1280;
    int height_ = 720;
    int fps_    = 30;

    cv::Mat                  latest_;
    std::mutex               mtx_;
    std::condition_variable  cv_;
    std::atomic<bool>        running_{false};
    bool                     has_frame_{false};
    std::thread              thread_;

    void capture_loop();

    // GStreamer callback — gọi mỗi khi có frame mới
    static GstFlowReturn on_new_sample(GstElement* sink, gpointer user_data);
};
