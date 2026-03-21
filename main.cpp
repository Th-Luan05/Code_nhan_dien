/*
 * main.cpp
 * YOLOv8 NCNN + libcamerasrc (BGR) + GStreamer WebRTC
 *
 * Build:  mkdir build && cd build && cmake .. && make -j4
 * Chay:   ./yolo_stream --model ../ncnn_model --port 8080
 * Xem:    http://<IP_PI>:8080
 */

#include "camera.h"

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

// HTTP signaling
#include "httplib.h"

// ===============================================================
// CAU HINH
// ===============================================================
struct Config {
    std::string model_dir  = "ncnn_model";
    int         cam_w      = 1280;
    int         cam_h      = 720;
    int         cam_fps    = 30;
    int         roi_size   = 480;   // crop vuong o giua
    int         infer_size = 320;   // imgsz NCNN
    float       conf_thres = 0.5f;
    float       nms_thres  = 0.45f;
    int         skip       = 3;     // inference moi skip frame
    int         port       = 8080;
    int         threads    = 4;
};

// ===============================================================
// DETECTION STRUCT
// ===============================================================
struct Detection {
    cv::Rect    bbox;       // toa do trong ROI
    float       conf;
    int         cls_id;
    std::string label;
};

// ===============================================================
// YOLO NCNN
// ===============================================================
class YoloNcnn {
public:
    bool load(const std::string& dir, int threads = 4) {
        net_.opt.use_vulkan_compute = false;
        net_.opt.num_threads        = threads;
        if (net_.load_param((dir + "/model.ncnn.param").c_str()) != 0) return false;
        if (net_.load_model((dir + "/model.ncnn.bin").c_str())   != 0) return false;
        std::cout << "[YOLO] model loaded: " << dir << "\n";
        return true;
    }

    // input: BGR cv::Mat (tu Camera class)
    std::vector<Detection> detect(const cv::Mat& bgr,
                                   int isz,
                                   float conf_thres, float nms_thres,
                                   const std::vector<std::string>& names)
    {
        int img_w = bgr.cols, img_h = bgr.rows;
        float scale = std::min((float)isz/img_w, (float)isz/img_h);
        int nw = (int)(img_w*scale), nh = (int)(img_h*scale);
        int px = (isz-nw)/2,        py = (isz-nh)/2;

        // BGR input (camera.cpp output la BGR)
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, nw, nh);
        ncnn::copy_make_border(in, in, py, isz-nh-py, px, isz-nw-px,
                               ncnn::BORDER_CONSTANT, 114.f);

        const float mean_vals[3] = {0,0,0};
        const float norm_vals[3] = {1/255.f,1/255.f,1/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out0, out1;
        ex.extract("out0", out0);
        ex.extract("out1", out1);

        std::vector<Detection> dets;
        if (out0.empty()) return dets;

        int num_class = (int)names.size();
        if (out0.c < 4 + num_class) {
            num_class = out0.c - 4;
            if (num_class <= 0) return dets;
        }

        // Log shape lan dau
        static bool logged = false;
        if (!logged) {
            std::cout << "[YOLO] out0 w=" << out0.w
                      << " h=" << out0.h << " c=" << out0.c
                      << " num_class=" << num_class << "\n";
            logged = true;
        }

        std::vector<cv::Rect>  boxes;
        std::vector<float>     scores;
        std::vector<int>       cls_ids;

        for (int i = 0; i < out0.w; i++) {
            float max_s = -1; int max_c = 0;
            for (int c = 0; c < num_class; c++) {
                float s = out0.channel(4+c)[i];
                if (s > max_s) { max_s = s; max_c = c; }
            }
            if (max_s < conf_thres) continue;

            float cx = out0.channel(0)[i];
            float cy = out0.channel(1)[i];
            float bw = out0.channel(2)[i];
            float bh = out0.channel(3)[i];

            float x1 = ((cx-bw/2.f)-px)/scale;
            float y1 = ((cy-bh/2.f)-py)/scale;
            float x2 = ((cx+bw/2.f)-px)/scale;
            float y2 = ((cy+bh/2.f)-py)/scale;

            x1 = std::max(0.f, std::min(x1,(float)img_w));
            y1 = std::max(0.f, std::min(y1,(float)img_h));
            x2 = std::max(0.f, std::min(x2,(float)img_w));
            y2 = std::max(0.f, std::min(y2,(float)img_h));

            boxes.push_back({(int)x1,(int)y1,(int)(x2-x1),(int)(y2-y1)});
            scores.push_back(max_s);
            cls_ids.push_back(max_c);
        }

        std::vector<int> idx;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, nms_thres, idx);
        for (int i : idx) {
            Detection d;
            d.bbox   = boxes[i];
            d.conf   = scores[i];
            d.cls_id = cls_ids[i];
            d.label  = (d.cls_id < (int)names.size()) ? names[d.cls_id] : "?";
            dets.push_back(d);
        }
        return dets;
    }

private:
    ncnn::Net net_;
};

// ===============================================================
// DRAW OVERLAY (BGR — dung voi OpenCV)
// ===============================================================
static const std::vector<cv::Scalar> COLORS = {
    {0,255,0},{0,255,255},{255,100,0},{255,0,255},{0,100,255},
    {255,56,56},{255,157,151},{72,249,10},{0,194,255},{132,56,255},
};

void draw_overlay(cv::Mat& bgr_frame,
                  const std::vector<Detection>& dets,
                  int sx, int sy, int roi_size,
                  float fps, int skip,
                  const std::vector<std::string>& names)
{
    // Khung ROI — mau xanh duong BGR=(255,0,0)
    cv::rectangle(bgr_frame,
        {sx, sy}, {sx+roi_size, sy+roi_size},
        {255, 0, 0}, 2);
    cv::putText(bgr_frame, "Vung AI (480x480)",
        {sx, sy-10}, cv::FONT_HERSHEY_SIMPLEX,
        0.7, {255,0,0}, 2);

    // Bounding boxes — toa do: ROI + offset (sx,sy)
    for (const auto& d : dets) {
        auto& col = COLORS[d.cls_id % (int)COLORS.size()];
        int x1 = d.bbox.x + sx;
        int y1 = d.bbox.y + sy;
        int x2 = x1 + d.bbox.width;
        int y2 = y1 + d.bbox.height;

        cv::rectangle(bgr_frame, {x1,y1}, {x2,y2}, col, 3);

        std::ostringstream oss;
        oss << d.label << " "
            << std::fixed << std::setprecision(0)
            << d.conf*100 << "%";
        cv::putText(bgr_frame, oss.str(),
            {x1, y1-8}, cv::FONT_HERSHEY_SIMPLEX,
            0.85, col, 2, cv::LINE_AA);
    }

    // Panel ben phai — phat hien gi
    int px = bgr_frame.cols - 380, py2 = 50;
    if (!dets.empty()) {
        cv::putText(bgr_frame, "Phat hien:",
            {px, py2}, cv::FONT_HERSHEY_SIMPLEX,
            0.9, {0,255,255}, 2);
        for (int i = 0; i < (int)dets.size(); i++) {
            int ty = py2 + 40 + i*40;
            cv::circle(bgr_frame, {px+10, ty-10}, 9, {0,255,255}, -1);
            std::string lbl = dets[i].label + "  " +
                std::to_string((int)(dets[i].conf*100)) + "%";
            cv::putText(bgr_frame, lbl, {px+28, ty},
                cv::FONT_HERSHEY_SIMPLEX, 0.85, {0,255,255}, 2);
        }
    } else {
        cv::putText(bgr_frame, "Khong phat hien",
            {px, py2}, cv::FONT_HERSHEY_SIMPLEX,
            0.85, {100,100,100}, 2);
    }

    // FPS
    char buf[64];
    snprintf(buf, sizeof(buf), "FPS: %.1f  skip:%d", fps, skip);
    cv::putText(bgr_frame, buf, {10,45},
        cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,255}, 2);
}

// ===============================================================
// GSTREAMER WEBRTC
// ===============================================================
struct StreamCtx {
    GstElement*  pipeline  = nullptr;
    GstElement*  appsrc    = nullptr;
    GstElement*  webrtcbin = nullptr;
    std::atomic<bool>     running{false};
    std::mutex            sdp_mutex;
    std::string           local_sdp;
    std::vector<std::string> local_ice;
    std::atomic<uint64_t> frames_pushed{0};
    std::atomic<uint64_t> frames_dropped{0};
};
static StreamCtx g_ctx;

static void on_offer_created(GstPromise* promise, gpointer) {
    const GstStructure* reply = gst_promise_get_reply(promise);
    GstWebRTCSessionDescription* offer = nullptr;
    gst_structure_get(reply, "offer",
                      GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, nullptr);
    gst_promise_unref(promise);
    if (!offer) { std::cerr << "[WEBRTC] offer null!\n"; return; }

    GstPromise* p = gst_promise_new();
    g_signal_emit_by_name(g_ctx.webrtcbin, "set-local-description", offer, p);
    gst_promise_interrupt(p); gst_promise_unref(p);

    gchar* s = gst_sdp_message_as_text(offer->sdp);
    { std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex); g_ctx.local_sdp = s ? s : ""; }
    std::cout << "[WEBRTC] SDP offer ready (" << (s?strlen(s):0) << " bytes)\n";
    g_free(s);
    gst_webrtc_session_description_free(offer);
}

static void on_ice_candidate(GstElement*, guint mline, gchar* cand, gpointer) {
    std::ostringstream o;
    o << "{\"sdpMLineIndex\":" << mline << ",\"candidate\":\"" << cand << "\"}";
    std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
    g_ctx.local_ice.push_back(o.str());
}

static void on_negotiation_needed(GstElement*, gpointer) {
    std::cout << "[WEBRTC] negotiation needed\n";
    GstPromise* p = gst_promise_new_with_change_func(on_offer_created, nullptr, nullptr);
    g_signal_emit_by_name(g_ctx.webrtcbin, "create-offer", nullptr, p);
}

static void on_conn_state(GstElement* w, GParamSpec*, gpointer) {
    GstWebRTCPeerConnectionState st;
    g_object_get(w, "connection-state", &st, nullptr);
    const char* s[] = {"new","connecting","CONNECTED","disconnected","FAILED","closed"};
    if (st < 6) std::cout << "[WEBRTC] conn -> " << s[st] << "\n";
}

static void on_ice_state(GstElement* w, GParamSpec*, gpointer) {
    GstWebRTCICEConnectionState st;
    g_object_get(w, "ice-connection-state", &st, nullptr);
    const char* s[] = {"new","checking","CONNECTED","completed","FAILED","disconnected","closed"};
    if (st < 7) std::cout << "[ICE] state -> " << s[st] << "\n";
}

bool init_webrtc(const Config& cfg) {
    gst_init(nullptr, nullptr);

    // Input la BGR (tu Camera class) → videoconvert → I420 → vp8enc → webrtcbin
    std::ostringstream ps;
    ps << "appsrc name=src is-live=true format=time block=false "
       << "caps=video/x-raw,format=BGR"
       << ",width="     << cfg.cam_w
       << ",height="    << cfg.cam_h
       << ",framerate=" << cfg.cam_fps << "/1 "
       << "! videoconvert "
       << "! video/x-raw,format=I420 "
       << "! vp8enc deadline=1 error-resilient=partitions "
       <<   "keyframe-max-dist=30 auto-alt-ref=true cpu-used=8 "
       <<   "target-bitrate=1000000 "
       << "! rtpvp8pay pt=96 "
       << "! webrtcbin name=webrtc bundle-policy=max-bundle "
       <<   "stun-server=stun://stun.l.google.com:19302";

    GError* err = nullptr;
    g_ctx.pipeline = gst_parse_launch(ps.str().c_str(), &err);
    if (err) {
        std::cerr << "[GST] " << err->message << "\n";
        g_error_free(err); return false;
    }

    g_ctx.appsrc    = gst_bin_get_by_name(GST_BIN(g_ctx.pipeline), "src");
    g_ctx.webrtcbin = gst_bin_get_by_name(GST_BIN(g_ctx.pipeline), "webrtc");
    if (!g_ctx.appsrc || !g_ctx.webrtcbin) {
        std::cerr << "[GST] element not found\n"; return false;
    }

    g_signal_connect(g_ctx.webrtcbin, "on-negotiation-needed",
                     G_CALLBACK(on_negotiation_needed), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "on-ice-candidate",
                     G_CALLBACK(on_ice_candidate), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "notify::connection-state",
                     G_CALLBACK(on_conn_state), nullptr);
    g_signal_connect(g_ctx.webrtcbin, "notify::ice-connection-state",
                     G_CALLBACK(on_ice_state), nullptr);

    // Bus watch
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(g_ctx.pipeline));
    gst_bus_add_watch(bus, [](GstBus*, GstMessage* msg, gpointer) -> gboolean {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* e; gchar* d;
            gst_message_parse_error(msg, &e, &d);
            std::cerr << "[GST ERROR] " << e->message << " | " << (d?d:"") << "\n";
            g_error_free(e); g_free(d);
        }
        return TRUE;
    }, nullptr);
    gst_object_unref(bus);

    // GLib loop
    GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
    std::thread([loop]{ g_main_loop_run(loop); }).detach();

    gst_element_set_state(g_ctx.pipeline, GST_STATE_PLAYING);
    g_ctx.running = true;
    std::cout << "[GST] pipeline PLAYING\n";
    return true;
}

void push_frame(const cv::Mat& bgr, int fps) {
    if (!g_ctx.appsrc || !g_ctx.running) return;

    gsize size = bgr.total() * bgr.elemSize();
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

    if (ret == GST_FLOW_OK) g_ctx.frames_pushed++;
    else {
        g_ctx.frames_dropped++;
        std::cerr << "[PUSH] failed: " << gst_flow_get_name(ret) << "\n";
    }
}

// ===============================================================
// HTTP SIGNALING
// ===============================================================
static const char* INDEX_HTML = R"html(
<!DOCTYPE html><html>
<head><meta charset="utf-8"><title>YOLOv8 Pi4</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;
     display:flex;flex-direction:column;align-items:center;margin:0;padding:20px}
video{width:100%;max-width:900px;border:2px solid #444;border-radius:8px;background:#000}
#status{margin:8px 0;font-size:13px;color:#aaa}
#log{font-size:11px;color:#555;font-family:monospace;max-height:120px;
     overflow-y:auto;width:100%;max-width:900px}
</style></head>
<body>
<h2>YOLOv8 Live — Raspberry Pi 4</h2>
<video id="v" autoplay playsinline muted></video>
<div id="status">Connecting...</div>
<div id="log"></div>
<script>
function log(m){
  const e=document.getElementById('log');
  e.innerHTML+=new Date().toISOString().slice(11,19)+' '+m+'<br>';
  e.scrollTop=e.scrollHeight;
}
const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
pc.onconnectionstatechange=()=>{
  log('conn: '+pc.connectionState);
  document.getElementById('status').textContent=pc.connectionState;
};
pc.oniceconnectionstatechange=()=>log('ice: '+pc.iceConnectionState);
pc.ontrack=e=>{
  log('track! kind='+e.track.kind);
  document.getElementById('v').srcObject=e.streams[0];
};
pc.onicecandidate=e=>{
  if(e.candidate){
    log('send ICE');
    fetch('/ice',{method:'POST',headers:{'Content-Type':'application/json'},
                  body:JSON.stringify(e.candidate)});
  }
};
async function start(){
  try{
    log('GET /offer...');
    const r=await fetch('/offer');
    if(!r.ok){log('offer err '+r.status);return;}
    const d=await r.json();
    log('got SDP len='+d.sdp.length);
    await pc.setRemoteDescription({type:'offer',sdp:d.sdp});
    const ans=await pc.createAnswer();
    await pc.setLocalDescription(ans);
    log('send answer');
    await fetch('/answer',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sdp:ans.sdp})});
    setInterval(async()=>{
      const ri=await fetch('/ice_candidates');
      const cs=await ri.json();
      for(const c of cs) await pc.addIceCandidate(c).catch(e=>log('ice err:'+e));
    },300);
  }catch(e){log('EX: '+e);}
}
start();
</script></body></html>
)html";

void run_http(int port) {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& r){
        r.set_content(INDEX_HTML,"text/html");
    });

    svr.Get("/offer", [](const httplib::Request&, httplib::Response& r){
        for (int i=0; i<100; i++) {
            {
                std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
                if (!g_ctx.local_sdp.empty()) {
                    std::string esc;
                    for (char c : g_ctx.local_sdp) {
                        if      (c=='\n') esc+="\\n";
                        else if (c=='\r') esc+="\\r";
                        else if (c=='"')  esc+="\\\"";
                        else if (c=='\\') esc+="\\\\";
                        else              esc+=c;
                    }
                    r.set_content("{\"sdp\":\""+esc+"\"}","application/json");
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        r.status=503;
    });

    svr.Post("/answer", [](const httplib::Request& req, httplib::Response& r){
        auto& b=req.body;
        auto p1=b.find("\"sdp\""); if(p1==std::string::npos){r.status=400;return;}
        auto p2=b.find("\"",p1+6);
        auto p3=b.rfind("\"");
        if(p2==std::string::npos||p3<=p2){r.status=400;return;}
        std::string sdp=b.substr(p2+1,p3-p2-1);

        // Unescape
        std::string un;
        for(size_t i=0;i<sdp.size();i++){
            if(sdp[i]=='\\'&&i+1<sdp.size()&&sdp[i+1]=='n'){un+='\n';i++;}
            else if(sdp[i]=='\\'&&i+1<sdp.size()&&sdp[i+1]=='r'){un+='\r';i++;}
            else un+=sdp[i];
        }

        GstSDPMessage* msg; gst_sdp_message_new(&msg);
        if(gst_sdp_message_parse_buffer((guint8*)un.c_str(),un.size(),msg)!=GST_SDP_OK){
            r.status=400; return;
        }
        GstWebRTCSessionDescription* ans=
            gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER,msg);
        GstPromise* p=gst_promise_new();
        g_signal_emit_by_name(g_ctx.webrtcbin,"set-remote-description",ans,p);
        gst_promise_interrupt(p); gst_promise_unref(p);
        gst_webrtc_session_description_free(ans);
        r.set_content("{\"ok\":true}","application/json");
    });

    svr.Post("/ice", [](const httplib::Request& req, httplib::Response& r){
        auto& b=req.body;
        auto p1=b.find("\"candidate\""); if(p1==std::string::npos){r.status=400;return;}
        auto p2=b.find("\"",p1+12);
        auto p3=b.find("\"",p2+1);
        std::string cand=b.substr(p2+1,p3-p2-1);
        auto pm=b.find("\"sdpMLineIndex\"");
        int mline=0;
        if(pm!=std::string::npos) try{mline=std::stoi(b.substr(pm+16,3));}catch(...){}
        g_signal_emit_by_name(g_ctx.webrtcbin,"add-ice-candidate",mline,cand.c_str());
        r.set_content("{\"ok\":true}","application/json");
    });

    svr.Get("/ice_candidates", [](const httplib::Request&, httplib::Response& r){
        std::lock_guard<std::mutex> lk(g_ctx.sdp_mutex);
        std::string arr="[";
        for(size_t i=0;i<g_ctx.local_ice.size();i++){
            if(i) arr+=",";
            arr+=g_ctx.local_ice[i];
        }
        arr+="]";
        g_ctx.local_ice.clear();
        r.set_content(arr,"application/json");
    });

    std::cout << "[HTTP] Signaling: http://0.0.0.0:" << port << "\n";
    svr.listen("0.0.0.0", port);
}

// ===============================================================
// MAIN
// ===============================================================
int main(int argc, char* argv[]) {
    Config cfg;
    for (int i=1; i<argc-1; i++) {
        std::string a=argv[i];
        if(a=="--model") cfg.model_dir =argv[i+1];
        if(a=="--port")  cfg.port      =std::stoi(argv[i+1]);
        if(a=="--conf")  cfg.conf_thres=std::stof(argv[i+1]);
        if(a=="--skip")  cfg.skip      =std::stoi(argv[i+1]);
        if(a=="--w")     cfg.cam_w     =std::stoi(argv[i+1]);
        if(a=="--h")     cfg.cam_h     =std::stoi(argv[i+1]);
        if(a=="--fps")   cfg.cam_fps   =std::stoi(argv[i+1]);
    }

    std::cout << "[CONFIG] model=" << cfg.model_dir
              << " " << cfg.cam_w << "x" << cfg.cam_h
              << "@" << cfg.cam_fps << " roi=" << cfg.roi_size
              << " infer=" << cfg.infer_size
              << " conf=" << cfg.conf_thres
              << " skip=" << cfg.skip
              << " port=" << cfg.port << "\n";

    // Load class names
    std::vector<std::string> class_names;
    std::ifstream f(cfg.model_dir + "/classes.txt");
    if (f.is_open()) {
        std::string line;
        while(std::getline(f,line))
            if(!line.empty()) class_names.push_back(line);
        std::cout << "[INFO] " << class_names.size() << " classes\n";
    } else {
        std::cerr << "[WARN] classes.txt not found!\n";
        // Fallback 7 class cu the
        class_names = {"Background","HinhLapPhuong_Do","HinhLapPhuong_Vang",
                       "HinhLapPhuong_Xanh","HinhTru_Do","HinhTru_Vang","HinhTru_Xanh"};
    }

    // Load NCNN
    YoloNcnn yolo;
    if (!yolo.load(cfg.model_dir, cfg.threads)) return 1;

    // Khoi dong Camera (dung camera.h/cpp — BGR output)
    Camera cam;
    if (!cam.start(cfg.cam_w, cfg.cam_h, cfg.cam_fps)) return 1;

    // Khoi dong GStreamer WebRTC
    if (!init_webrtc(cfg)) return 1;

    // HTTP signaling server
    std::thread([&]{ run_http(cfg.port); }).detach();

    std::cout << "\n[READY] Mo trinh duyet: http://<IP_PI>:"
              << cfg.port << "\n\n";

    // ROI offset
    int sx = (cfg.cam_w  - cfg.roi_size) / 2;
    int sy = (cfg.cam_h - cfg.roi_size) / 2;

    // Adaptive skip
    int   skip      = cfg.skip;
    float fps_smooth = (float)cfg.cam_fps;
    int   frame_cnt  = 0;
    auto  t_prev     = std::chrono::steady_clock::now();
    std::vector<Detection> last_dets;

    cv::Mat frame;
    while (g_ctx.running) {
        // Lay frame BGR tu Camera (libcamerasrc → BGR)
        if (!cam.get_frame(frame)) continue;

        // Crop ROI vuong o giua
        cv::Mat roi = frame(cv::Rect(sx, sy, cfg.roi_size, cfg.roi_size));

        // Inference moi skip frame
        if (frame_cnt % skip == 0) {
            try {
                last_dets = yolo.detect(roi,
                                        cfg.infer_size,
                                        cfg.conf_thres,
                                        cfg.nms_thres,
                                        class_names);
            } catch (...) {
                std::cerr << "[YOLO] exception!\n";
            }
        }
        frame_cnt++;

        // Ve overlay len frame goc BGR
        // bbox trong last_dets la toa do trong ROI
        // draw_overlay tu dong offset +sx,+sy
        draw_overlay(frame, last_dets,
                     sx, sy, cfg.roi_size,
                     fps_smooth, skip, class_names);

        // Push BGR frame vao GStreamer (caps=BGR)
        push_frame(frame, cfg.cam_fps);

        // FPS + adaptive skip
        auto t1 = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(t1-t_prev).count();
        t_prev = t1;
        if (dt > 0) fps_smooth = fps_smooth*0.9f + (1.f/dt)*0.1f;

        if      (fps_smooth < 20.f) skip = std::min(skip+1, 6);
        else if (fps_smooth > 28.f) skip = std::max(skip-1, 2);

        if (frame_cnt % 60 == 0) {
            std::cout << "[MAIN] frame=" << frame_cnt
                      << " fps=" << std::fixed << std::setprecision(1) << fps_smooth
                      << " dets=" << last_dets.size()
                      << " skip=" << skip
                      << " pushed=" << g_ctx.frames_pushed.load()
                      << " dropped=" << g_ctx.frames_dropped.load() << "\n";
        }
    }

    cam.stop();
    gst_element_set_state(g_ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(g_ctx.pipeline);
    return 0;
}
