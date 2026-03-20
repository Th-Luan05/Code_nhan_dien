/*
 * main.cpp
 * YOLOv8 NCNN inference trên Raspberry Pi 4
 * Camera: libcamera | Inference: NCNN | Stream: GStreamer WebRTC
 *
 * Build:  mkdir build && cd build && cmake .. && make -j4
 * Chạy:   ./yolo_stream --model ../ncnn_model --port 8080
 * Xem:    Mở trình duyệt → http://<IP_PI>:8080
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstdlib>

// NCNN
#include "ncnn/net.h"
#include "ncnn/mat.h"

// OpenCV
#include <opencv2/opencv.hpp>

// GStreamer
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
#include <gst/sdp/sdp.h>

// HTTP signaling server
#include "httplib.h"

// ===============================================================
// CAU HINH
// ===============================================================
struct Config {
    std::string model_dir   = "ncnn_model";
    int         cam_id      = 0;
    int         cam_w       = 640;
    int         cam_h       = 480;
    int         cam_fps     = 30;
    int         infer_w     = 640;
    int         infer_h     = 640;
    float       conf_thres  = 0.25f;
    float       nms_thres   = 0.45f;
    int         port        = 8080;
    int         num_threads = 4;
};

// ===============================================================
// STRUCT KET QUA DETECTION
// ===============================================================
struct Detection {
    cv::Rect    bbox;
    float       conf;
    int         cls_id;
    std::string label;
};

// ===============================================================
// CLASS YOLO NCNN
// ===============================================================
class YoloNcnn {
public:
    bool load(const std::string& model_dir, int num_threads = 4) {
        std::string param = model_dir + "/model.ncnn.param";
        std::string bin   = model_dir + "/model.ncnn.bin";

        net_.opt.use_vulkan_compute = false;
        net_.opt.num_threads        = num_threads;

        if (net_.load_param(param.c_str()) != 0) {
            std::cerr << "[ERROR] Khong load duoc: " << param << "\n";
            return false;
        }
        if (net_.load_model(bin.c_str()) != 0) {
            std::cerr << "[ERROR] Khong load duoc: " << bin << "\n";
            return false;
        }
        std::cout << "[INFO] NCNN model loaded: " << model_dir << "\n";
        return true;
    }

    std::vector<Detection> detect(const cv::Mat& bgr,
                                   int infer_w, int infer_h,
                                   float conf_thres, float nms_thres,
                                   const std::vector<std::string>& class_names)
    {
        int img_w = bgr.cols;
        int img_h = bgr.rows;

        float scale = std::min((float)infer_w / img_w, (float)infer_h / img_h);
        int   new_w = (int)(img_w * scale);
        int   new_h = (int)(img_h * scale);
        int   pad_x = (infer_w - new_w) / 2;
        int   pad_y = (infer_h - new_h) / 2;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
            img_w, img_h, new_w, new_h);

        ncnn::copy_make_border(in, in,
            pad_y, infer_h - new_h - pad_y,
            pad_x, infer_w - new_w - pad_x,
            ncnn::BORDER_CONSTANT, 114.f);

        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out0, out1;
        ex.extract("out0", out0);
        ex.extract("out1", out1);
        ncnn::Mat& out = out0;

        std::vector<Detection> dets;
        std::vector<cv::Rect>  boxes;
        std::vector<float>     scores;
        std::vector<int>       class_ids;

        int num_class = (int)class_names.size();
        int num_pred  = out.w;

        for (int i = 0; i < num_pred; i++) {
            float max_score = -1.f;
            int   max_cls   = 0;
            for (int c = 0; c < num_class; c++) {
                float s = out.channel(4 + c)[i];
                if (s > max_score) { max_score = s; max_cls = c; }
            }
            if (max_score < conf_thres) continue;

            float cx = out.channel(0)[i];
            float cy = out.channel(1)[i];
            float bw = out.channel(2)[i];
            float bh = out.channel(3)[i];

            float x1 = ((cx - bw/2.f) - pad_x) / scale;
            float y1 = ((cy - bh/2.f) - pad_y) / scale;
            float x2 = ((cx + bw/2.f) - pad_x) / scale;
            float y2 = ((cy + bh/2.f) - pad_y) / scale;

            x1 = std::max(0.f, std::min(x1, (float)img_w));
            y1 = std::max(0.f, std::min(y1, (float)img_h));
            x2 = std::max(0.f, std::min(x2, (float)img_w));
            y2 = std::max(0.f, std::min(y2, (float)img_h));

            boxes.push_back({(int)x1,(int)y1,(int)(x2-x1),(int)(y2-y1)});
            scores.push_back(max_score);
            class_ids.push_back(max_cls);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, nms_thres, indices);

        for (int idx : indices) {
            Detection d;
            d.bbox   = boxes[idx];
            d.conf   = scores[idx];
            d.cls_id = class_ids[idx];
            d.label  = (d.cls_id < (int)class_names.size())
                       ? class_names[d.cls_id] : "unknown";
            dets.push_back(d);
        }
        return dets;
    }

private:
    ncnn::Net net_;
};

// ===============================================================
// VE BBOX
// ===============================================================
static const std::vector<cv::Scalar> COLORS = {
    {255,56,56},{255,157,151},{255,112,31},{255,178,29},
    {207,210,49},{72,249,10},{146,204,23},{61,219,134},
    {26,147,52},{0,212,187},{44,153,168},{0,194,255},
    {52,69,147},{100,115,255},{0,24,236},{132,56,255},
    {82,0,133},{203,56,255},{255,149,200},{255,55,199},
};

void draw_detections(cv::Mat& frame,
                     const std::vector<Detection>& dets, float fps)
{
    for (const auto& d : dets) {
        auto& col = COLORS[d.cls_id % (int)COLORS.size()];
        cv::rectangle(frame,
            {d.bbox.x, d.bbox.y},
            {d.bbox.x+d.bbox.width, d.bbox.y+d.bbox.height}, col, 2);

        std::ostringstream oss;
        oss << d.label << " " << std::fixed << std::setprecision(2) << d.conf;
        std::string text = oss.str();

        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.55, 1, &baseline);
        cv::rectangle(frame,
            {d.bbox.x, d.bbox.y - ts.height - 6},
            {d.bbox.x + ts.width + 4, d.bbox.y}, col, -1);
        cv::putText(frame, text, {d.bbox.x+2, d.bbox.y-3},
            cv::FONT_HERSHEY_SIMPLEX, 0.55,
            {255,255,255}, 1, cv::LINE_AA);
    }

    std::ostringstream fps_oss;
    fps_oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, fps_oss.str(), {10,28},
        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2, cv::LINE_AA);
}

// ===============================================================
// GSTREAMER WEBRTC
// ===============================================================
struct StreamCtx {
    GstElement*           pipeline  = nullptr;
    GstElement*           appsrc    = nullptr;
    GstElement*           webrtcbin = nullptr;
    std::atomic<bool>     running{false};
    std::mutex            sdp_mutex;
    std::string           local_sdp;
    std::vector<std::string> local_ice;
    std::atomic<uint64_t> frames_pushed{0};
    std::atomic<uint64_t> frames_dropped{0};
};

static StreamCtx g_ctx;

static void on_offer_created(GstPromise* promise, gpointer) {
    std::cout << "[WEBRTC] on_offer_created fired\n";
    const GstStructure* reply = gst_promise_get_reply(promise);
    GstWebRTCSessionDescription* offer = nullptr;
    gst_structure_get(reply, "offer",
                      GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, nullptr);
    gst_promise_unref(promise);

    if (!offer) {
        std::cerr << "[WEBRTC] ERROR: offer is null!\n";
        return;
    }

    GstPromise* p = gst_promise_new();
    g_signal_emit_by_name(g_ctx.webrtcbin,
                          "set-local-description", offer, p);
    gst_promise_interrupt(p);
    gst_promise_unref(p);

    gchar* sdp_str = gst_sdp_message_as_text(offer->sdp);
    {
        std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
        g_ctx.local_sdp = sdp_str ? sdp_str : "";
    }
    std::cout << "[WEBRTC] SDP offer created, length="
              << (sdp_str ? strlen(sdp_str) : 0) << " bytes\n";
    g_free(sdp_str);
    gst_webrtc_session_description_free(offer);
}

static void on_ice_candidate(GstElement*, guint mline,
                              gchar* candidate, gpointer)
{
    std::cout << "[ICE] candidate mline=" << mline
              << " : " << candidate << "\n";
    std::ostringstream oss;
    oss << "{\"sdpMLineIndex\":" << mline
        << ",\"candidate\":\"" << candidate << "\"}";
    std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
    g_ctx.local_ice.push_back(oss.str());
}

static void on_negotiation_needed(GstElement*, gpointer) {
    std::cout << "[WEBRTC] on_negotiation_needed fired -> creating offer\n";
    GstPromise* promise = gst_promise_new_with_change_func(
        on_offer_created, nullptr, nullptr);
    g_signal_emit_by_name(g_ctx.webrtcbin, "create-offer", nullptr, promise);
}

static void on_connection_state_changed(GstElement* webrtc,
                                         GParamSpec*, gpointer) {
    GstWebRTCPeerConnectionState state;
    g_object_get(webrtc, "connection-state", &state, nullptr);
    const char* s = "unknown";
    switch(state) {
        case GST_WEBRTC_PEER_CONNECTION_STATE_NEW:          s="new"; break;
        case GST_WEBRTC_PEER_CONNECTION_STATE_CONNECTING:   s="connecting"; break;
        case GST_WEBRTC_PEER_CONNECTION_STATE_CONNECTED:    s="CONNECTED"; break;
        case GST_WEBRTC_PEER_CONNECTION_STATE_DISCONNECTED: s="disconnected"; break;
        case GST_WEBRTC_PEER_CONNECTION_STATE_FAILED:       s="FAILED"; break;
        case GST_WEBRTC_PEER_CONNECTION_STATE_CLOSED:       s="closed"; break;
        default: break;
    }
    std::cout << "[WEBRTC] connection-state -> " << s << "\n";
}

static void on_ice_connection_state_changed(GstElement* webrtc,
                                             GParamSpec*, gpointer) {
    GstWebRTCICEConnectionState state;
    g_object_get(webrtc, "ice-connection-state", &state, nullptr);
    const char* s = "unknown";
    switch(state) {
        case GST_WEBRTC_ICE_CONNECTION_STATE_NEW:          s="new"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_CHECKING:     s="checking"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_CONNECTED:    s="CONNECTED"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_COMPLETED:    s="completed"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_FAILED:       s="FAILED"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_DISCONNECTED: s="disconnected"; break;
        case GST_WEBRTC_ICE_CONNECTION_STATE_CLOSED:       s="closed"; break;
        default: break;
    }
    std::cout << "[ICE] connection-state -> " << s << "\n";
}

bool init_gstreamer_pipeline(const Config& cfg) {
    setenv("GST_DEBUG", "webrtcbin:4,vp8enc:3,appsrc:3,rtpvp8pay:3", 1);
    gst_init(nullptr, nullptr);

    std::cout << "[GST] GStreamer version: "
              << gst_version_string() << "\n";

    std::ostringstream pipe_str;
    pipe_str << "appsrc name=src is-live=true format=time block=false "
             << "caps=video/x-raw,format=BGR"
             << ",width="     << cfg.cam_w
             << ",height="    << cfg.cam_h
             << ",framerate=" << cfg.cam_fps << "/1 "
             << "! videoconvert "
             << "! video/x-raw,format=I420 "
             << "! vp8enc deadline=1 error-resilient=partitions "
             <<   "keyframe-max-dist=30 auto-alt-ref=true "
             <<   "cpu-used=8 target-bitrate=800000 "
             << "! rtpvp8pay pt=96 "
             << "! webrtcbin name=webrtc bundle-policy=max-bundle "
             <<   "stun-server=stun://stun.l.google.com:19302";

    std::cout << "[GST] Pipeline string:\n  " << pipe_str.str() << "\n";

    GError* err = nullptr;
    g_ctx.pipeline = gst_parse_launch(pipe_str.str().c_str(), &err);
    if (err) {
        std::cerr << "[GST ERROR] parse_launch: " << err->message << "\n";
        g_error_free(err);
        return false;
    }

    g_ctx.appsrc    = gst_bin_get_by_name(GST_BIN(g_ctx.pipeline), "src");
    g_ctx.webrtcbin = gst_bin_get_by_name(GST_BIN(g_ctx.pipeline), "webrtc");

    if (!g_ctx.appsrc) {
        std::cerr << "[GST ERROR] appsrc element not found!\n";
        return false;
    }
    if (!g_ctx.webrtcbin) {
        std::cerr << "[GST ERROR] webrtcbin element not found!\n";
        return false;
    }
    std::cout << "[GST] appsrc and webrtcbin found OK\n";

    g_signal_connect(g_ctx.webrtcbin, "on-negotiation-needed",
                     G_CALLBACK(on_negotiation_needed), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "on-ice-candidate",
                     G_CALLBACK(on_ice_candidate), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "notify::connection-state",
                     G_CALLBACK(on_connection_state_changed), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "notify::ice-connection-state",
                     G_CALLBACK(on_ice_connection_state_changed), nullptr);

    // Bus watch
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(g_ctx.pipeline));
    gst_bus_add_watch(bus, [](GstBus*, GstMessage* msg, gpointer) -> gboolean {
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* e; gchar* dbg;
                gst_message_parse_error(msg, &e, &dbg);
                std::cerr << "[GST ERROR] src=" << GST_MESSAGE_SRC_NAME(msg)
                          << " : " << e->message << "\n";
                std::cerr << "[GST DEBUG] " << (dbg ? dbg : "none") << "\n";
                g_error_free(e); g_free(dbg);
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* e; gchar* dbg;
                gst_message_parse_warning(msg, &e, &dbg);
                std::cerr << "[GST WARN] src=" << GST_MESSAGE_SRC_NAME(msg)
                          << " : " << e->message << "\n";
                g_error_free(e); g_free(dbg);
                break;
            }
            case GST_MESSAGE_STATE_CHANGED: {
                GstState old_s, new_s;
                gst_message_parse_state_changed(msg, &old_s, &new_s, nullptr);
                std::cout << "[GST STATE] " << GST_MESSAGE_SRC_NAME(msg)
                          << ": " << gst_element_state_get_name(old_s)
                          << " -> " << gst_element_state_get_name(new_s) << "\n";
                break;
            }
            case GST_MESSAGE_EOS:
                std::cerr << "[GST] EOS received\n";
                break;
            default: break;
        }
        return TRUE;
    }, nullptr);
    gst_object_unref(bus);

    // GLib main loop (thread rieng - can cho bus watch + webrtcbin)
    GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
    std::thread([loop]() {
        std::cout << "[GST] GLib main loop started\n";
        g_main_loop_run(loop);
    }).detach();

    GstStateChangeReturn sc =
        gst_element_set_state(g_ctx.pipeline, GST_STATE_PLAYING);
    std::cout << "[GST] set_state(PLAYING) return=" << sc
              << " (0=failure,1=success,2=async,3=no_preroll)\n";

    if (sc == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[GST ERROR] Pipeline failed to go to PLAYING!\n";
        return false;
    }

    g_ctx.running = true;
    std::cout << "[INFO] GStreamer pipeline started OK\n";
    return true;
}

// Push 1 frame BGR vao appsrc
void push_frame(const cv::Mat& bgr, int fps) {
    if (!g_ctx.appsrc || !g_ctx.running) {
        std::cerr << "[PUSH] WARN: appsrc null or not running\n";
        return;
    }

    gsize size = bgr.total() * bgr.elemSize();
    static uint64_t push_count = 0;

    if (push_count % 30 == 0) {
        std::cout << "[PUSH] frame #" << push_count
                  << " size=" << size
                  << " " << bgr.cols << "x" << bgr.rows
                  << " pushed_ok=" << g_ctx.frames_pushed.load()
                  << " dropped=" << g_ctx.frames_dropped.load() << "\n";
    }
    push_count++;

    GstBuffer* buf = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    memcpy(map.data, bgr.data, size);
    gst_buffer_unmap(buf, &map);

    static GstClockTime ts = 0;
    GST_BUFFER_PTS(buf)      = ts;
    GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(1, GST_SECOND, fps);
    ts += GST_BUFFER_DURATION(buf);

    GstFlowReturn ret;
    g_signal_emit_by_name(g_ctx.appsrc, "push-buffer", buf, &ret);
    gst_buffer_unref(buf);

    if (ret == GST_FLOW_OK) {
        g_ctx.frames_pushed++;
    } else {
        g_ctx.frames_dropped++;
        std::cerr << "[PUSH] FAILED ret=" << ret
                  << " (" << gst_flow_get_name(ret) << ")\n";
    }
}

// ===============================================================
// HTTP SIGNALING SERVER
// ===============================================================
static const char* INDEX_HTML = R"html(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>YOLOv8 Live Stream - Pi 4</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center;
         margin:0; padding:20px; }
  h2   { margin-bottom:12px; }
  video{ width:100%; max-width:800px; border:2px solid #444;
         border-radius:8px; background:#000; }
  #status { margin-top:10px; font-size:13px; color:#aaa; }
  #log    { margin-top:8px; font-size:11px; color:#666; font-family:monospace;
            max-height:150px; overflow-y:auto; width:100%; max-width:800px; }
</style>
</head>
<body>
<h2>YOLOv8 Live - Raspberry Pi 4</h2>
<video id="video" autoplay playsinline muted></video>
<div id="status">Connecting...</div>
<div id="log"></div>
<script>
function log(msg) {
  const el = document.getElementById('log');
  const t  = new Date().toISOString().slice(11,19);
  el.innerHTML += t + ' ' + msg + '<br>';
  el.scrollTop  = el.scrollHeight;
  console.log('[WebRTC]', msg);
}

const pc = new RTCPeerConnection({
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
});

pc.onconnectionstatechange = () => {
  const s = pc.connectionState;
  log('connection-state: ' + s);
  document.getElementById('status').textContent = s;
};
pc.oniceconnectionstatechange = () =>
  log('ice-connection-state: ' + pc.iceConnectionState);
pc.onsignalingstatechange = () =>
  log('signaling-state: ' + pc.signalingState);

pc.ontrack = e => {
  log('ontrack! streams=' + e.streams.length
      + ' kind=' + e.track.kind);
  const video = document.getElementById('video');
  video.srcObject = e.streams[0];
  video.play().catch(err => log('play() error: ' + err));
};

pc.onicecandidate = e => {
  if (e.candidate) {
    log('sending ICE: ' + e.candidate.candidate.slice(0, 60) + '...');
    fetch('/ice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(e.candidate)
    });
  } else {
    log('ICE gathering complete');
  }
};

async function start() {
  try {
    log('GET /offer ...');
    const r = await fetch('/offer');
    if (!r.ok) { log('ERROR /offer: status=' + r.status); return; }

    const data = await r.json();
    log('Got SDP offer, length=' + data.sdp.length);

    await pc.setRemoteDescription({ type: 'offer', sdp: data.sdp });
    log('setRemoteDescription OK');

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    log('createAnswer + setLocalDescription OK');

    const ar = await fetch('/answer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ sdp: answer.sdp })
    });
    log('POST /answer -> ' + ar.status);

    // Poll ICE candidates tu Pi
    setInterval(async () => {
      try {
        const ri = await fetch('/ice_candidates');
        const candidates = await ri.json();
        for (const c of candidates) {
          log('addIceCandidate: ' + JSON.stringify(c).slice(0, 60));
          await pc.addIceCandidate(c)
            .catch(e => log('addIceCandidate ERROR: ' + e));
        }
      } catch(e) { log('poll ICE error: ' + e); }
    }, 300);

  } catch(e) {
    log('EXCEPTION: ' + e);
  }
}
start();
</script>
</body>
</html>
)html";

void run_signaling_server(int port) {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(INDEX_HTML, "text/html");
    });

    svr.Get("/offer", [](const httplib::Request&, httplib::Response& res) {
        std::cout << "[HTTP] GET /offer (waiting for SDP...)\n";
        for (int i = 0; i < 100; i++) {
            {
                std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
                if (!g_ctx.local_sdp.empty()) {
                    std::string esc;
                    for (char c : g_ctx.local_sdp) {
                        if      (c == '\n') esc += "\\n";
                        else if (c == '\r') esc += "\\r";
                        else if (c == '"')  esc += "\\\"";
                        else if (c == '\\') esc += "\\\\";
                        else                esc += c;
                    }
                    std::cout << "[HTTP] /offer: sending SDP ("
                              << esc.size() << " bytes)\n";
                    res.set_content("{\"sdp\":\"" + esc + "\"}",
                                    "application/json");
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cerr << "[HTTP] /offer: timeout! SDP never created\n";
        res.status = 503;
    });

    svr.Post("/answer", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[HTTP] POST /answer len=" << req.body.size() << "\n";
        auto& body = req.body;

        auto pos1 = body.find("\"sdp\"");
        if (pos1 == std::string::npos) {
            std::cerr << "[HTTP] /answer: no sdp key\n";
            res.status = 400; return;
        }
        auto pos2 = body.find("\"", pos1 + 6);
        auto pos3 = body.rfind("\"");
        if (pos2 == std::string::npos || pos3 <= pos2) {
            std::cerr << "[HTTP] /answer: bad JSON\n";
            res.status = 400; return;
        }
        std::string sdp = body.substr(pos2 + 1, pos3 - pos2 - 1);

        // Unescape \n
        std::string sdp_un;
        for (size_t i = 0; i < sdp.size(); i++) {
            if (sdp[i]=='\\' && i+1<sdp.size() && sdp[i+1]=='n') {
                sdp_un += '\n'; i++;
            } else if (sdp[i]=='\\' && i+1<sdp.size() && sdp[i+1]=='r') {
                sdp_un += '\r'; i++;
            } else {
                sdp_un += sdp[i];
            }
        }
        std::cout << "[HTTP] /answer: SDP unescaped len=" << sdp_un.size() << "\n";

        GstSDPMessage* msg;
        gst_sdp_message_new(&msg);
        GstSDPResult r = gst_sdp_message_parse_buffer(
            (guint8*)sdp_un.c_str(), sdp_un.size(), msg);
        if (r != GST_SDP_OK) {
            std::cerr << "[HTTP] /answer: parse SDP failed r=" << r << "\n";
            res.status = 400; return;
        }

        GstWebRTCSessionDescription* answer =
            gst_webrtc_session_description_new(
                GST_WEBRTC_SDP_TYPE_ANSWER, msg);
        GstPromise* p = gst_promise_new();
        g_signal_emit_by_name(g_ctx.webrtcbin,
                              "set-remote-description", answer, p);
        gst_promise_interrupt(p);
        gst_promise_unref(p);
        gst_webrtc_session_description_free(answer);

        std::cout << "[HTTP] /answer: set-remote-description OK\n";
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Post("/ice", [](const httplib::Request& req, httplib::Response& res) {
        auto& body = req.body;
        std::cout << "[HTTP] POST /ice: "
                  << body.substr(0, 80) << "\n";

        auto pos1 = body.find("\"candidate\"");
        if (pos1 == std::string::npos) { res.status=400; return; }
        auto pos2 = body.find("\"", pos1 + 12);
        auto pos3 = body.find("\"", pos2 + 1);
        std::string cand = body.substr(pos2 + 1, pos3 - pos2 - 1);

        auto pm = body.find("\"sdpMLineIndex\"");
        int mline = 0;
        if (pm != std::string::npos) {
            try { mline = std::stoi(body.substr(pm + 16, 3)); }
            catch(...) {}
        }
        g_signal_emit_by_name(g_ctx.webrtcbin,
                              "add-ice-candidate", mline, cand.c_str());
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Get("/ice_candidates", [](const httplib::Request&,
                                   httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
        std::string arr = "[";
        for (size_t i = 0; i < g_ctx.local_ice.size(); i++) {
            if (i) arr += ",";
            arr += g_ctx.local_ice[i];
        }
        arr += "]";
        g_ctx.local_ice.clear();
        res.set_content(arr, "application/json");
    });

    std::cout << "[INFO] Signaling server: http://0.0.0.0:" << port << "\n";
    svr.listen("0.0.0.0", port);
}

// ===============================================================
// MAIN
// ===============================================================
int main(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--model")  cfg.model_dir  = argv[i+1];
        if (a == "--port")   cfg.port       = std::stoi(argv[i+1]);
        if (a == "--conf")   cfg.conf_thres = std::stof(argv[i+1]);
        if (a == "--width")  cfg.cam_w      = std::stoi(argv[i+1]);
        if (a == "--height") cfg.cam_h      = std::stoi(argv[i+1]);
        if (a == "--fps")    cfg.cam_fps    = std::stoi(argv[i+1]);
    }

    std::cout << "[CONFIG] model=" << cfg.model_dir
              << " cam=" << cfg.cam_w << "x" << cfg.cam_h
              << "@" << cfg.cam_fps << "fps port=" << cfg.port << "\n";

    // Load class names
    std::vector<std::string> class_names;
    std::ifstream cls_file(cfg.model_dir + "/classes.txt");
    if (cls_file.is_open()) {
        std::string line;
        while (std::getline(cls_file, line))
            if (!line.empty()) class_names.push_back(line);
        std::cout << "[INFO] Loaded " << class_names.size()
                  << " classes\n";
    } else {
        std::cout << "[WARN] classes.txt not found, using cls0..cls79\n";
        for (int i = 0; i < 80; i++)
            class_names.push_back("cls" + std::to_string(i));
    }

    // Load NCNN
    YoloNcnn yolo;
    if (!yolo.load(cfg.model_dir, cfg.num_threads)) return 1;

    // GStreamer
    if (!init_gstreamer_pipeline(cfg)) return 1;

    // Camera
    cconst std::string cam_cmd =
        "rpicam-vid -t 0 --width 640 --height 480 --framerate 30 "
        "--codec mjpeg -o - 2>/dev/null | "
        "ffmpeg -f mjpeg -i pipe:0 "
        "-f rawvideo -pix_fmt bgr24 -vf scale=640:480 pipe:1 2>/dev/null";

    FILE* cam_pipe = popen(cam_cmd.c_str(), "r");
    if (!cam_pipe) {
        std::cerr << "[ERROR] Khong mo duoc cam_pipe\n";
        return 1;
    }
    std::cout << "[INFO] Camera pipe opened OK (640x480 BGR)\n";

    // Signaling server in thread rieng
    std::thread([&]() { run_signaling_server(cfg.port); }).detach();

    std::cout << "\n[READY] Mo trinh duyet: http://<IP_PI>:"
              << cfg.port << "\n\n";

    // Main loop
    cv::Mat frame;
    auto  t_prev    = std::chrono::steady_clock::now();
    float fps_smooth = 0.f;
    int   frame_count = 0;

    const int frame_bytes = 640 * 480 * 3;
    frame = cv::Mat(480, 640, CV_8UC3);

    while (g_ctx.running) {
        size_t n = fread(frame.data, 1, frame_bytes, cam_pipe);
        if (n != (size_t)frame_bytes) {
            std::cerr << "[CAM] pipe read error n=" << n << "\n";
            break;
        }

        auto dets = yolo.detect(frame,
                                 cfg.infer_w, cfg.infer_h,
                                 cfg.conf_thres, cfg.nms_thres,
                                 class_names);

        auto t1 = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(t1 - t_prev).count();
        t_prev = t1;
        if (dt > 0) fps_smooth = fps_smooth * 0.9f + (1.f/dt) * 0.1f;

        draw_detections(frame, dets, fps_smooth);
        push_frame(frame, cfg.cam_fps);

        if (++frame_count % 60 == 0) {
            std::cout << "[MAIN] frame=" << frame_count
                      << " fps=" << std::fixed
                      << std::setprecision(1) << fps_smooth
                      << " dets=" << dets.size()
                      << " pushed=" << g_ctx.frames_pushed.load()
                      << " dropped=" << g_ctx.frames_dropped.load()
                      << "\n";
        }
    }

    gst_element_set_state(g_ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(g_ctx.pipeline);
    pclose(cam_pipe);
    return 0;
}
