#include "camera.h"
#include <cstdio>
#include <cstring>

// ── Callback: gọi mỗi khi GStreamer có frame mới ─────────────────
GstFlowReturn Camera::on_new_sample(GstElement* sink, gpointer user_data) {
    Camera* self = static_cast<Camera*>(user_data);

    // Pull sample từ appsink
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps*   caps   = gst_sample_get_caps(sample);

    // Lấy kích thước từ caps
    GstStructure* s = gst_caps_get_structure(caps, 0);
    int w = 0, h = 0;
    gst_structure_get_int(s, "width",  &w);
    gst_structure_get_int(s, "height", &h);

    // Map buffer — zero-copy nếu có thể
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        // Wrap data vào cv::Mat BGR (không copy)
        cv::Mat frame(h, w, CV_8UC3, map.data);

        {
            std::lock_guard<std::mutex> lk(self->mtx_);
            self->latest_    = frame.clone();  // Clone để giữ data sau unmap
            self->has_frame_ = true;
        }
        self->cv_.notify_one();

        gst_buffer_unmap(buffer, &map);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// ── Start camera ─────────────────────────────────────────────────
bool Camera::start(int width, int height, int fps) {
    width_  = width;
    height_ = height;
    fps_    = fps;

    gst_init(nullptr, nullptr);

    // Pipeline: libcamerasrc → NV21 → videoconvert → BGR → appsink
    std::string desc =
        "libcamerasrc ! "
        "video/x-raw,width="    + std::to_string(width)  +
        ",height="              + std::to_string(height) +
        ",framerate="           + std::to_string(fps) + "/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=appsink0 emit-signals=true sync=false "
        "max-buffers=2 drop=true";

    printf("[CAM] Pipeline: %s\n\n", desc.c_str());

    GError* err = nullptr;
    pipeline_ = gst_parse_launch(desc.c_str(), &err);
    if (!pipeline_ || err) {
        fprintf(stderr, "[CAM] Lỗi tạo pipeline: %s\n",
                err ? err->message : "unknown");
        if (err) g_error_free(err);
        return false;
    }

    // Lấy appsink element
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink0");
    if (!appsink_) {
        fprintf(stderr, "[CAM] Không tìm thấy appsink0\n");
        return false;
    }

    // Kết nối callback new-sample
    g_signal_connect(appsink_, "new-sample",
                     G_CALLBACK(Camera::on_new_sample), this);

    // Bắt đầu pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[CAM] Không start được pipeline\n");
        return false;
    }

    // Chờ frame đầu tiên để xác nhận camera OK
    {
        std::unique_lock<std::mutex> lk(mtx_);
        bool ok = cv_.wait_for(lk, std::chrono::seconds(5),
                               [this]{ return has_frame_; });
        if (!ok) {
            fprintf(stderr, "[CAM] Timeout — không nhận được frame sau 5 giây\n");
            fprintf(stderr, "[CAM] Thử chạy: rpicam-hello để kiểm tra camera\n");
            return false;
        }
    }

    printf("[CAM] Camera OK: %dx%d @ %dfps\n", width_, height_, fps_);
    running_ = true;
    return true;
}

// ── Stop ─────────────────────────────────────────────────────────
void Camera::stop() {
    running_ = false;
    cv_.notify_all();
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }
}

// ── Get frame ────────────────────────────────────────────────────
bool Camera::get_frame(cv::Mat& out) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::milliseconds(100),
                 [this]{ return has_frame_; });
    if (!has_frame_) return false;
    out        = latest_.clone();
    has_frame_ = false;
    return true;
}
