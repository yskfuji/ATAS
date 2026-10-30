// Batch SAM3 video tracking for the ATAS geometry service (sam3.cpp, Metal or CPU).
//
//   atas_sam3_track --model M.ggml --frames DIR --out DIR (--prompt TEXT)... [--no-gpu] [--threads N]
//   atas_sam3_track --model M.ggml --frames DIR --out DIR --init-boxes boxes.txt [--no-gpu]
//   atas_sam3_track --model M.ggml --frames DIR --out DIR --init-masks masks.txt [--no-gpu]   (patched sam3.cpp only)
//   ... --hybrid-k K --edgetam-model E.ggml   (patched sam3.cpp only) with --prompt or --init-boxes:
//       SAM3 runs only on frames 0, K, 2K, ...; after each such frame EdgeTAM is re-initialised from SAM3's masks
//       (mask hand-off) and propagates the K-1 frames in between. Output files and schema are unchanged.
//       Needs sam3.cpp with patches/mask_handoff.patch and patches/visual_no_object_gating.patch. Hand-off options:
//       --handoff-min-coverage C (default -1 = off; e.g. 0.001), text prompts only unless --handoff-f0-box: never hand over an
//         empty SAM3 mask; a mask covering <= C of the frame only if it is not a fragment within --handoff-fragment-px (default 40)
//         of a larger mask on the same keyframe.
//       --handoff-empty-absent, --gate-no-object (default off): SAM 2's no-object semantics for the hand-off and EdgeTAM.
//       Hand-off decisions are logged as {"handoff":true,...} rows; propagated detections carry EdgeTAM's obj_score, empty ones are not written.
//
// DIR holds frames named 00000.png, 00001.png, ... in temporal order.
// Text mode: one tracker per prompt over all frames (SAM3 PCS detection + tracking).
// Box mode (semi-supervised evaluation): each line of boxes.txt is "object_id x0 y0 x1 y1"
// [px py] on frame 0 (optional positive point); instances are added with PVS and propagated.
// Visual-only models (EdgeTAM, SAM2) have no detector: box mode uses the visual tracker instead (encode frame 0,
// add the instances with PVS, then propagate each later frame by memory attention); text mode is refused.
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
#include <map>
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
                            int frame, const sam3_detection & det, bool with_obj_score = false) {
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
        << ",\"mask\":\"masks/" << name << "\"";
    if (with_obj_score) log << ",\"obj_score\":" << det.mask.obj_score;
    log << "}\n";
}

int main(int argc, char ** argv) {
    sam3_params params;
    std::string frames_dir, out_dir, boxes_file, masks_file, edgetam_path;
    int hybrid_k = 0;
    // Hand-off fixes F0/F0'/F1 (docs/macos/HYBRID_LEAK_DIAGNOSIS.md) are opt-in: the registered evaluation rejected them as
    // defaults (F0 drops true small objects whose box-prompted keyframe mask is empty or tiny). All off = the stage-2 hybrid.
    // F0 acts on text prompts only (the empty/tiny masks it targets are the text tracker's hot-start duplicates); a box-prompted
    // object is distinct, so its empty or tiny keyframe mask is still handed over unless --handoff-f0-box is given.
    double handoff_min_coverage = -1.0;   // e.g. 0.001 enables F0; < 0 disables it
    bool handoff_f0_box = false;
    int handoff_fragment_px = 40;
    bool handoff_empty_absent = false, gate_no_object = false;
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
        else if (a == "--init-masks") masks_file = next();
        else if (a == "--hybrid-k") hybrid_k = std::atoi(next().c_str());
        else if (a == "--edgetam-model") edgetam_path = next();
        else if (a == "--handoff-min-coverage") handoff_min_coverage = std::atof(next().c_str());
        else if (a == "--handoff-fragment-px") handoff_fragment_px = std::atoi(next().c_str());
        else if (a == "--handoff-f0-box") handoff_f0_box = true;
        else if (a == "--handoff-empty-absent") handoff_empty_absent = true;
        else if (a == "--no-handoff-empty-absent") handoff_empty_absent = false;
        else if (a == "--gate-no-object") gate_no_object = true;
        else if (a == "--no-gate-no-object") gate_no_object = false;
        else if (a == "--threads") params.n_threads = std::atoi(next().c_str());
        else if (a == "--score-threshold") score_threshold = std::atof(next().c_str());
        else if (a == "--no-gpu") params.use_gpu = false;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!masks_file.empty()) {
#ifndef ATAS_MASK_HANDOFF
        fprintf(stderr, "--init-masks needs sam3.cpp patched with mask_handoff.patch (build with -DATAS_MASK_HANDOFF=ON)\n");
        return 2;
#endif
        boxes_file = masks_file;   // one file of initial objects; its lines name masks instead of boxes
    }
    if (hybrid_k > 0) {
#ifndef ATAS_MASK_HANDOFF
        fprintf(stderr, "--hybrid-k needs sam3.cpp patched with mask_handoff.patch (build with -DATAS_MASK_HANDOFF=ON)\n");
        return 2;
#endif
        if (hybrid_k < 2 || edgetam_path.empty() || !masks_file.empty()) {
            fprintf(stderr, "--hybrid-k K (K >= 2) needs --edgetam-model and cannot be combined with --init-masks\n");
            return 2;
        }
    }
    if (params.model_path.empty() || frames_dir.empty() || out_dir.empty() || (prompts.empty() == boxes_file.empty())) {
        fprintf(stderr, "need --model --frames --out and exactly one of --prompt... / --init-boxes / --init-masks\n");
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
    std::shared_ptr<sam3_model> model2;
    sam3_params params2 = params;
    double load2_s = 0;
    size_t keyframes = 0, handoff_instances = 0, handoff_failed = 0, empty_keyframes = 0, handoff_dropped = 0;
    if (hybrid_k > 0) {
        auto t2 = clock::now();
        params2.model_path = edgetam_path;
        model2 = sam3_load_model(params2);
        if (!model2 || !sam3_is_visual_only(*model2)) { fprintf(stderr, "--edgetam-model must load a visual-only model\n"); return 4; }
        if (sam3_is_visual_only(*model)) { fprintf(stderr, "--model must be a SAM3 model with a detector in hybrid mode\n"); return 4; }
        load2_s = std::chrono::duration<double>(clock::now() - t2).count();
    }

    std::ofstream log(out / "tracks.jsonl");
    if (!log) { fprintf(stderr, "cannot open %s\n", (out / "tracks.jsonl").c_str()); return 4; }
    std::vector<double> frame_seconds;
    size_t detections = 0;

    auto run = [&](int prompt_index, const std::string & prompt, bool box_mode) {
        auto state = sam3_create_state(*model, params);
        const bool visual = sam3_is_visual_only(*model);
        if (visual && !box_mode) { fprintf(stderr, "a visual-only model cannot take text prompts\n"); std::exit(2); }
        sam3_tracker_ptr tracker;
        if (visual) {
            sam3_visual_track_params vtp;
            tracker = sam3_create_visual_tracker(*model, vtp);
        } else {
            sam3_video_params vp;
            vp.text_prompt = box_mode ? "" : prompt;
            vp.score_threshold = score_threshold;
            if (hybrid_k > 0) {
                // The text tracker counts tracker steps; with SAM3 on every K-th frame one step spans K video frames.
                auto per_step = [&](int frames_count) { return std::max(1, (frames_count + hybrid_k - 1) / hybrid_k); };
                vp.hotstart_delay = per_step(vp.hotstart_delay);
                vp.max_keep_alive = per_step(vp.max_keep_alive);
                vp.recondition_every = per_step(vp.recondition_every);
            }
            tracker = sam3_create_tracker(*model, vp);
        }
        // EdgeTAM side of the hybrid (patched sam3.cpp only): a fresh state and visual tracker at every keyframe.
        decltype(sam3_create_state(*model, params)) state2;
        sam3_tracker_ptr tracker2;
        std::map<int, int> to_sam3;        // EdgeTAM instance id -> SAM3 instance id
        std::map<int, float> last_score;   // SAM3 score at the last keyframe, reported for propagated masks
        std::vector<std::pair<int, sam3_pvs_params>> init_prompts;  // box mode: (SAM3 id, prompt) on frame 0
        if (!state || !tracker) { fprintf(stderr, "failed to create tracker\n"); std::exit(4); }
        for (size_t f = 0; f < frames.size(); ++f) {
            sam3_image image = sam3_load_image(frames[f].string());
            if (image.data.empty()) { fprintf(stderr, "failed to read %s\n", frames[f].c_str()); std::exit(3); }
            auto t0 = clock::now();
            sam3_result result;
            const bool keyframe = hybrid_k == 0 || f % (size_t)hybrid_k == 0;
            if (!keyframe) {
#ifdef ATAS_MASK_HANDOFF
                if (tracker2 && !to_sam3.empty()) {
                    result = sam3_propagate_frame(*tracker2, *state2, *model2, image);
                    std::vector<sam3_detection> kept;
                    for (auto det : result.detections) {
                        auto it = to_sam3.find(det.instance_id);
                        if (it == to_sam3.end()) continue;
                        bool any = false;
                        for (auto v : det.mask.data) if (v > 127) { any = true; break; }
                        if (!any) continue;   // "no object" (or an empty propagation) is not written
                        det.instance_id = it->second; det.mask.instance_id = it->second; det.score = last_score[it->second];
                        kept.push_back(std::move(det));
                    }
                    result.detections.swap(kept);
                }
#endif
            } else if (visual && f == 0) {
                if (!sam3_encode_image(*state, *model, image)) { fprintf(stderr, "failed to encode frame 0\n"); std::exit(4); }
            } else if (visual) {
                result = sam3_propagate_frame(*tracker, *state, *model, image);
            } else {
                result = sam3_track_frame(*tracker, *state, *model, image);
            }
            if (box_mode && f == 0 && keyframe) {
                // Instances are added on the encoded first frame (for the text tracker its index has already advanced).
                std::ifstream boxes(boxes_file);
                std::string line;
                while (std::getline(boxes, line)) {
                    std::istringstream row(line);
                    int object_id; int id = -1; sam3_detection det;
                    if (!masks_file.empty()) {
#ifdef ATAS_MASK_HANDOFF
                        std::string mask_path;
                        if (!(row >> object_id >> mask_path)) continue;
                        sam3_image gray = sam3_load_image(mask_path);   // 3-channel decode of a grey PNG: red channel carries the mask
                        if (gray.data.empty()) { fprintf(stderr, "failed to read mask %s\n", mask_path.c_str()); std::exit(3); }
                        sam3_mask given; given.width = gray.width; given.height = gray.height;
                        given.data.resize((size_t)gray.width * gray.height);
                        for (size_t p = 0; p < given.data.size(); ++p) given.data[p] = gray.data[3 * p] > 127 ? 255 : 0;
                        id = sam3_tracker_add_instance_from_mask(*tracker, *state, *model, given);
                        det.mask = given; det.score = 1.0f; det.iou_score = 1.0f;
#endif
                    } else {
                        sam3_pvs_params pvs;
                        if (!(row >> object_id >> pvs.box.x0 >> pvs.box.y0 >> pvs.box.x1 >> pvs.box.y1)) continue;
                        sam3_point point;
                        if (row >> point.x >> point.y) pvs.pos_points.push_back(point);  // optional interior point
                        pvs.use_box = true; pvs.multimask = false;
                        id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
                        auto shown = sam3_segment_pvs(*state, *model, pvs);
                        if (id >= 0 && !shown.detections.empty()) det = shown.detections[0]; else id = -1;
                        if (id >= 0 && hybrid_k > 0) { init_prompts.emplace_back(id, pvs); last_score[id] = det.score; }
                    }
                    if (id < 0) { fprintf(stderr, "failed to add object %d\n", object_id); std::exit(5); }
                    det.instance_id = id; det.mask.instance_id = id;
                    log << "{\"object_id\":" << object_id << ",\"instance_id\":" << id << ",\"frame\":0,\"init\":true}\n";
                    result.detections.push_back(std::move(det));
                }
            }
#ifdef ATAS_MASK_HANDOFF
            if (hybrid_k > 0 && keyframe) {
                state2 = sam3_create_state(*model2, params2);
                sam3_visual_track_params vtp2;
                vtp2.gate_no_object = gate_no_object;
                tracker2 = sam3_create_visual_tracker(*model2, vtp2);
                to_sam3.clear();
                if (!state2 || !tracker2 || !sam3_encode_image(*state2, *model2, image)) { fprintf(stderr, "failed to start EdgeTAM at frame %zu\n", f); std::exit(4); }
                if (box_mode && f == 0) {
                    for (auto & ip : init_prompts) {   // the first frame is prompted like AC15: box + interior point
                        int id2 = sam3_tracker_add_instance(*tracker2, *state2, *model2, ip.second);
                        if (id2 < 0) { fprintf(stderr, "EdgeTAM failed to add object\n"); std::exit(5); }
                        to_sam3[id2] = ip.first;
                    }
                } else {
                    std::vector<long> areas;
                    for (const auto & det : result.detections) { long a = 0; for (auto v : det.mask.data) a += v > 127; areas.push_back(a); }
                    for (size_t di = 0; di < result.detections.size(); ++di) {   // mask hand-off from SAM3, in SAM3's output order
                        const auto & det = result.detections[di];
                        const long area = areas[di];
                        std::string drop;
                        bool isolated_small = false;
                        if (handoff_min_coverage >= 0.0 && (!box_mode || handoff_f0_box)) {
                            const int W = det.mask.width, Hh = det.mask.height, r = handoff_fragment_px;
                            if (area == 0) drop = "empty";
                            else if (area <= handoff_min_coverage * (double)W * Hh) {
                                bool fragment = false;   // within r px of a larger mask on this keyframe
                                for (size_t dj = 0; dj < result.detections.size() && !fragment; ++dj) {
                                    if (dj == di || areas[dj] <= area) continue;
                                    const auto & big = result.detections[dj].mask.data;
                                    for (int y = 0; y < Hh && !fragment; ++y)
                                        for (int x = 0; x < W && !fragment; ++x) {
                                            if (det.mask.data[(size_t)y * W + x] <= 127) continue;
                                            for (int yy = std::max(0, y - r); yy <= std::min(Hh - 1, y + r) && !fragment; ++yy)
                                                for (int xx = std::max(0, x - r); xx <= std::min(W - 1, x + r); ++xx)
                                                    if (big[(size_t)yy * W + xx] > 127 && (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r) { fragment = true; break; }
                                        }
                                }
                                if (fragment) drop = "fragment"; else isolated_small = true;
                            }
                        }
                        int id2 = -2;
                        if (drop.empty()) {
                            id2 = sam3_tracker_add_instance_from_mask(*tracker2, *state2, *model2, det.mask, handoff_empty_absent);
                            if (id2 >= 0) { to_sam3[id2] = det.instance_id; last_score[det.instance_id] = det.score; ++handoff_instances; }
                            else ++handoff_failed;   // a SAM3 instance EdgeTAM could not take over is lost until the next keyframe
                        } else {
                            ++handoff_dropped;
                        }
                        log << "{\"handoff\":true,\"frame\":" << f << ",\"sam3_instance\":" << det.instance_id << ",\"edgetam_instance\":" << id2
                            << ",\"score\":" << det.score << ",\"area\":" << area << ",\"dropped\":\"" << drop << "\",\"isolated_small\":"
                            << (isolated_small ? "true" : "false") << "}\n";
                    }
                    if (to_sam3.empty()) ++empty_keyframes;
                }
                ++keyframes;
            }
#endif
            frame_seconds.push_back(std::chrono::duration<double>(clock::now() - t0).count());
            for (const auto & det : result.detections) { write_detection(log, out, prompt_index, prompt, (int)f, det, hybrid_k > 0 && !keyframe); ++detections; }
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
        << ",\"hybrid_k\":" << hybrid_k << ",\"keyframes\":" << keyframes << ",\"handoff_instances\":" << handoff_instances
        << ",\"handoff_failed\":" << handoff_failed << ",\"handoff_dropped\":" << handoff_dropped << ",\"empty_keyframes\":" << empty_keyframes << ",\"edgetam_load_s\":" << load2_s
        << ",\"peak_footprint_bytes\":" << peak_footprint() << "}}\n";
    log.close();   // a failed write must not look like a finished run (callers parse tracks.jsonl)
    if (log.fail()) { fprintf(stderr, "writing %s failed\n", (out / "tracks.jsonl").c_str()); return 4; }
    return 0;
}
