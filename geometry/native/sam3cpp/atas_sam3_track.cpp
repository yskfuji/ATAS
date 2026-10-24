// Batch SAM3 video tracking for the ATAS geometry service (sam3.cpp, Metal or CPU).
//
//   atas_sam3_track --model M.ggml --frames DIR --out DIR (--prompt TEXT)... [--no-gpu] [--threads N]
//   atas_sam3_track --model M.ggml --frames DIR --out DIR --init-boxes boxes.txt [--no-gpu]
//
// DIR holds frames named 00000.png, 00001.png, ... in temporal order.
// Text mode: one tracker per prompt over all frames (SAM3 PCS detection + tracking).
// Box mode (semi-supervised evaluation): each line of boxes.txt is "object_id x0 y0 x1 y1"
// [px py] on frame 0 (optional positive point); instances are added with PVS and propagated.
// Output: DIR/masks/<prompt>_<frame>_<instance>.png and DIR/tracks.jsonl, one JSON object
// per detection, plus a final {"summary": ...} line with timings and the peak footprint.
#include "sam3.h"

#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static uint64_t peak_footprint() {
    rusage_info_v4 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&info)) != 0) return 0;
    return info.ri_lifetime_max_phys_footprint;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
        else out += c;
    }
    return out;
}

static void write_detection(std::ofstream & log, const fs::path & out, int prompt_index, const std::string & prompt,
                            int frame, const sam3_detection & det) {
    char name[128];
    snprintf(name, sizeof name, "p%02d_f%05d_i%04d.png", prompt_index, frame, det.instance_id);
    fs::path mask_path = out / "masks" / name;
    if (!sam3_save_mask(det.mask, mask_path.string())) {
        fprintf(stderr, "failed to save %s\n", mask_path.c_str());
        std::exit(3);
    }
    log << "{\"prompt_index\":" << prompt_index << ",\"prompt\":\"" << json_escape(prompt) << "\",\"frame\":" << frame
        << ",\"instance_id\":" << det.instance_id << ",\"score\":" << det.score << ",\"iou_score\":" << det.iou_score
        << ",\"box_xyxy\":[" << det.box.x0 << "," << det.box.y0 << "," << det.box.x1 << "," << det.box.y1 << "]"
        << ",\"mask\":\"masks/" << name << "\"}\n";
}

int main(int argc, char ** argv) {
    sam3_params params;
    std::string frames_dir, out_dir, boxes_file;
    std::vector<std::string> prompts;
    float score_threshold = 0.5f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--model") params.model_path = next();
        else if (a == "--frames") frames_dir = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--prompt") prompts.push_back(next());
        else if (a == "--init-boxes") boxes_file = next();
        else if (a == "--threads") params.n_threads = std::atoi(next().c_str());
        else if (a == "--score-threshold") score_threshold = std::atof(next().c_str());
        else if (a == "--no-gpu") params.use_gpu = false;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (params.model_path.empty() || frames_dir.empty() || out_dir.empty() || (prompts.empty() == boxes_file.empty())) {
        fprintf(stderr, "need --model --frames --out and exactly one of --prompt... / --init-boxes\n");
        return 2;
    }
    std::vector<fs::path> frames;
    for (const auto & e : fs::directory_iterator(frames_dir))
        if (e.path().extension() == ".png") frames.push_back(e.path());
    std::sort(frames.begin(), frames.end());
    if (frames.empty()) { fprintf(stderr, "no frames in %s\n", frames_dir.c_str()); return 2; }
    fs::path out(out_dir);
    fs::create_directories(out / "masks");

    using clock = std::chrono::steady_clock;
    auto t_load = clock::now();
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 4; }
    double load_s = std::chrono::duration<double>(clock::now() - t_load).count();

    std::ofstream log(out / "tracks.jsonl");
    std::vector<double> frame_seconds;
    size_t detections = 0;

    auto run = [&](int prompt_index, const std::string & prompt, bool box_mode) {
        auto state = sam3_create_state(*model, params);
        sam3_video_params vp;
        vp.text_prompt = box_mode ? "" : prompt;
        vp.score_threshold = score_threshold;
        auto tracker = sam3_create_tracker(*model, vp);
        if (!state || !tracker) { fprintf(stderr, "failed to create tracker\n"); std::exit(4); }
        for (size_t f = 0; f < frames.size(); ++f) {
            sam3_image image = sam3_load_image(frames[f].string());
            if (image.data.empty()) { fprintf(stderr, "failed to read %s\n", frames[f].c_str()); std::exit(3); }
            auto t0 = clock::now();
            sam3_result result = sam3_track_frame(*tracker, *state, *model, image);
            if (box_mode && f == 0) {
                // Instances are added on the encoded first frame; the tracker index has already advanced.
                std::ifstream boxes(boxes_file);
                std::string line;
                while (std::getline(boxes, line)) {
                    std::istringstream row(line);
                    int object_id; sam3_pvs_params pvs;
                    if (!(row >> object_id >> pvs.box.x0 >> pvs.box.y0 >> pvs.box.x1 >> pvs.box.y1)) continue;
                    sam3_point point;
                    if (row >> point.x >> point.y) pvs.pos_points.push_back(point);  // optional interior point
                    pvs.use_box = true; pvs.multimask = false;
                    int id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
                    auto shown = sam3_segment_pvs(*state, *model, pvs);
                    if (id < 0 || shown.detections.empty()) { fprintf(stderr, "failed to add object %d\n", object_id); std::exit(5); }
                    sam3_detection det = shown.detections[0];
                    det.instance_id = id; det.mask.instance_id = id;
                    log << "{\"object_id\":" << object_id << ",\"instance_id\":" << id << ",\"frame\":0,\"init\":true}\n";
                    result.detections.push_back(std::move(det));
                }
            }
            frame_seconds.push_back(std::chrono::duration<double>(clock::now() - t0).count());
            for (const auto & det : result.detections) { write_detection(log, out, prompt_index, prompt, (int)f, det); ++detections; }
        }
    };
    if (boxes_file.empty()) {
        for (size_t p = 0; p < prompts.size(); ++p) run((int)p, prompts[p], false);
    } else {
        run(0, "", true);
    }
    double total = 0, worst = 0;
    for (double s : frame_seconds) { total += s; worst = std::max(worst, s); }
    log << "{\"summary\":{\"frames\":" << frames.size() << ",\"passes\":" << (boxes_file.empty() ? prompts.size() : 1)
        << ",\"detections\":" << detections << ",\"load_s\":" << load_s << ",\"track_s\":" << total
        << ",\"sec_per_frame_max\":" << worst << ",\"use_gpu\":" << (params.use_gpu ? "true" : "false")
        << ",\"peak_footprint_bytes\":" << peak_footprint() << "}}\n";
    return 0;
}
