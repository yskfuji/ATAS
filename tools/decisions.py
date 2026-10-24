"""AC26 decisions and AC12 anchored-path parity: does the device change any world-state decision?

    python tools/decisions.py initial <clip-dir> <eval-out> --device cpu|mps
    python tools/decisions.py anchored <pair-new-clip> <eval-out> --device cpu|mps
    python tools/decisions.py summarise <run-id> <eval-out>

`initial` runs the first-observation check (prepare_initial_world) on the device's own reconstruction
of an analysis clip. `anchored` runs anchored registration (register.anchored_fusion) of a pair's new
clip against its history clip; both devices receive the same CPU-fp32 inputs, so the device is the
only difference. `summarise` counts flips and writes decisions.json and anchored_parity_<dtype>.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
PYTHON = REPO / ".venv-macos/bin/python"
PREREG = REPO / "evidence/prereg"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_for(device):
    sys.path.insert(0, str(REPO / "geometry"))
    env = {**os.environ, "GEOMETRY_DEVICE": device,
           "PYTHONPATH": f"{REPO / 'geometry'}{os.pathsep}{REPO / 'geometry/backbones'}"}
    if device == "mps":
        os.environ["GEOMETRY_DEVICE"] = "mps"
        from atas_geometry import device as accel
        env.update(accel.worker_env(0))
    return env


def initial(clip: Path, eval_out: Path, device: str, mode: str = "crn", k: int = 0):
    """First-observation check on the device's own reconstruction (CRN seed k, or native-RNG CPU repeat k)."""
    l0 = eval_out / clip.name / f"{device}_float32_{mode}{k}" / "l0"
    previous = eval_out / clip.name / f"initial_{device}_{mode}{k}.json"
    if device == "cpu" and previous.is_file() and json.loads(previous.read_text()).get("status") != "error":
        print(previous.read_text()[:200]); return
    folder = eval_out / clip.name / f"initial_world_{device}_{mode}{k}"
    prologue = (f"import os,sys; os.environ['GEOMETRY_EVAL_CRN_SEED']='{20261019 + k}'; "
                f"sys.path.insert(0, {str(REPO / 'tools')!r}); import crn_launch; crn_launch.install();") if mode == "crn" else ""
    extra = " | {'crn': crn_launch.draw_summary()}" if mode == "crn" else ""
    code = (prologue +
            "import json,sys; from pathlib import Path; from atas_geometry.reconstruct.initial_world import prepare_initial_world;"
            "import traceback\n"
            "try:\n"
            " r=prepare_initial_world(Path(sys.argv[1]),Path(sys.argv[2]),vista_repo=sys.argv[3],model_id=sys.argv[4],first_window=24,pixel_limit=160000)\n"
            "except Exception as exc:\n"
            " r={'status':'error','reason':traceback.format_exc()[-1500:]}\n"  # an upstream crash is a result; keep the draw log
            "print('RESULT '+json.dumps({k:r.get(k) for k in ('status','reason','assessment')}" + extra + "))")
    completed = subprocess.run([str(PYTHON), "-c", code, str(l0), str(folder), str(REPO / "geometry/backbones"),
                                str(CACHE / "weights/pi3x")], env=env_for(device), capture_output=True, text=True,
                               cwd=REPO / "geometry")
    line = next((l for l in completed.stdout.splitlines() if l.startswith("RESULT ")), None)
    result = json.loads(line[7:]) if line else {"status": "error", "reason": completed.stderr[-1500:]}
    (eval_out / clip.name / f"initial_{device}_{mode}{k}.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"clip": clip.name, "device": device, "mode": mode, "k": k, "status": result.get("status"), "reason": result.get("reason")}))


def anchored(clip: Path, eval_out: Path, device: str):
    history = eval_out / f"{clip.name}_history" / "cpu_float32_crn0" / "l0"
    new = eval_out / clip.name / "cpu_float32_crn0" / "l0"
    base = eval_out / clip.name / f"anchored_{device}"
    base.mkdir(parents=True, exist_ok=True)
    l1 = eval_out / f"{clip.name}_history" / "home_l1.npz"
    if not l1.is_file():  # history L1 is built once on CPU and shared by both devices
        fuse = ("import sys; from pathlib import Path; from atas_geometry.memory.fuse import fuse_l0_to_l1;"
                "fuse_l0_to_l1(Path(sys.argv[1]),Path(sys.argv[2]),duration=3.2,shot_id='history',gpu_index=0)")
        subprocess.run([str(PYTHON), "-c", fuse, str(history), str(l1)], env=env_for("cpu"), check=True, cwd=REPO / "geometry")
    command = [str(PYTHON), str(REPO / "tools/crn_launch.py"), "atas_geometry.register.anchored_fusion", "--reference-l0", str(history),
               "--new-l0", str(new), "--l1", str(l1), "--output", str(base / "out"),
               "--vista-repo", str(REPO / "geometry/backbones"), "--model-id", str(CACHE / "weights/pi3x")]
    env = {**env_for(device), "GEOMETRY_EVAL_CRN_SEED": "20261019", "GEOMETRY_EVAL_CRN_LOG": str(base / "crn.json")}
    env.pop("GEOMETRY_EVAL_NATIVE_RNG", None)
    with (base / "anchored.log").open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO / "geometry")
    report = base / "out" / "fusion_report.json"
    data = json.loads(report.read_text()) if report.is_file() else {}
    result = {"clip": clip.name, "device": device, "exit_code": completed.returncode,
              "registered": bool(data.get("cross_shot_validation", {}).get("accepted")) and bool(
                  data.get("reconstruction", {}).get("geometry_accepted")),
              "cross_shot_accepted": data.get("cross_shot_validation", {}).get("accepted"),
              "geometry_accepted": data.get("reconstruction", {}).get("geometry_accepted"),
              "peak_mib": data.get("gpu_reserved_peak_mib"),
              "crn": json.loads((base / "crn.json").read_text()) if (base / "crn.json").is_file() else None}
    (base / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def summarise(run_id: str, eval_out: Path):
    sys.path.insert(0, str(REPO / "tools"))
    from geometry_eval import parity
    analysis = [c["clip"] for c in json.loads((PREREG / "analysis_clips.json").read_text())["clips"]]
    pairs = [p["new"] for p in json.loads((PREREG / "e2e_pairs.json").read_text())["pairs"]]
    initial_rows, flips = {}, {"initial_world_status": 0, "registered": 0}
    draws_equal, native_flips, native_clips = True, 0, 0
    for name in analysis:
        cpu, mps = (json.loads((eval_out / name / f"initial_{d}_crn0.json").read_text()) for d in ("cpu", "mps"))
        initial_rows[name] = {"cpu": cpu.get("status"), "mps": mps.get("status")}
        flips["initial_world_status"] += int(cpu.get("status") != mps.get("status"))
        draws_equal &= (cpu.get("crn") or {}).get("draw_sequence_sha256") == (mps.get("crn") or {}).get("draw_sequence_sha256") is not None
        native = [json.loads(p.read_text()).get("status") for p in sorted((eval_out / name).glob("initial_cpu_native*.json"))]
        if len(native) >= 2:
            native_clips += 1
            native_flips += int(len(set(native)) > 1)
            initial_rows[name]["cpu_native_statuses"] = native
    anchored_rows, per_pair = {}, {}
    for name in pairs:
        cpu, mps = (json.loads((eval_out / name / f"anchored_{d}" / "result.json").read_text()) for d in ("cpu", "mps"))
        anchored_rows[name] = {"cpu": cpu["registered"], "mps": mps["registered"]}
        draws_equal &= (cpu.get("crn") or {}).get("draw_sequence_sha256") == (mps.get("crn") or {}).get("draw_sequence_sha256") is not None
        flips["registered"] += int(cpu["registered"] != mps["registered"])
        ref, test = (eval_out / name / f"anchored_{d}" / "out" / "conditioned_l0" for d in ("cpu", "mps"))
        if (ref / "cameras.npz").is_file() and (test / "cameras.npz").is_file():
            p = parity(ref, test, eval_out / name / "masks.npz")
            per_pair[name] = {k: v for k, v in p.items() if k != "per_frame_depth_absrel"}
    decisions = {"n_clips": len(analysis), "n_pairs": len(pairs), "clips_sha256": sha256(PREREG / "analysis_clips.json"),
                 "pairs_sha256": sha256(PREREG / "e2e_pairs.json"), "flips": flips, "random_numbers": "common",
                 "crn_draw_sequences_equal": bool(draws_equal),
                 "production_noise": {"cpu_native_initial_world_clips_with_differing_status": native_flips,
                                      "clips_with_native_repeats": native_clips,
                                      "note": "native-RNG CPU repeats: upstream inference is random (scale_aug); informative"},
                 "initial_world": initial_rows, "registration": anchored_rows}
    (REPO / "evidence" / run_id / "decisions.json").write_text(json.dumps(decisions, indent=2))
    (eval_out / "anchored_parity_float32.json").write_text(json.dumps({"per_pair": per_pair}, indent=2))
    print(json.dumps({"flips": flips, "pairs_with_parity": len(per_pair)}))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "summarise":
        summarise(sys.argv[2], Path(sys.argv[3]))
    else:
        device = sys.argv[sys.argv.index("--device") + 1]
        if action == "initial":
            mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "crn"
            k = int(sys.argv[sys.argv.index("--seed-index") + 1]) if "--seed-index" in sys.argv else 0
            initial(Path(sys.argv[2]), Path(sys.argv[3]), device, mode, k)
        else:
            anchored(Path(sys.argv[2]), Path(sys.argv[3]), device)
