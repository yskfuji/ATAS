import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest

from atas_geometry.memory import timebase
from tests.mac.conftest import GEOMETRY

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
LEVEL = 6  # frame k is filled with grey level k * LEVEL, so a decoded frame names its source


def _frames(count, size=(64, 96)):
    return [np.full((*size, 3), k * LEVEL, np.uint8) for k in range(count)]


def make_cfr(path, count=37, fps=30):
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, macro_block_size=1,
                            output_params=["-crf", "0", "-pix_fmt", "yuv444p"]) as writer:
        for frame in _frames(count):
            writer.append_data(frame)


def make_vfr(folder, durations):
    """Concat-demuxer clip whose frame k is shown for durations[k] seconds."""
    lines = []
    for k, frame in enumerate(_frames(len(durations))):
        imageio.imwrite(folder / f"{k:03d}.png", frame)
        lines += [f"file '{folder / f'{k:03d}.png'}'", f"duration {durations[k]:.6f}"]
    lines.append(f"file '{folder / f'{len(durations) - 1:03d}.png'}'")
    (folder / "list.txt").write_text("\n".join(lines) + "\n")
    out = folder / "vfr.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(folder / "list.txt"),
                    "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(out)], check=True)
    return out


def ffprobe_count(path):
    ffprobe = FFMPEG.replace("ffmpeg", "ffprobe")
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def decode(path):
    """Every frame exactly once, in presentation order (imageio's reader resamples VFR)."""
    import av
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in sorted(container.decode(video=0), key=lambda f: f.pts)]


def source_index(frame):
    return int(round(float(frame.mean()) / LEVEL))


def test_frame_count_matches_ffprobe(tmp_path):
    clip = tmp_path / "cfr.mp4"
    make_cfr(clip)
    expected = ffprobe_count(clip)
    info = timebase.probe(clip)
    assert info["frames"] == expected == 37 and info["cfr"] is True and abs(info["fps"] - 30) < 1e-6
    from atas_geometry.reconstruct.vista4d import probe_frame_count
    assert probe_frame_count(sys.executable, clip) == expected
    sys.path.insert(0, str(GEOMETRY / "backbones"))
    from utils.media import load_video
    video, fps = load_video(str(clip))
    assert len(video) == expected and [source_index(f) for f in video] == list(range(expected))
    path, record = timebase.ensure_cfr(clip, tmp_path / "job")
    assert path == clip and record["normalized"] is False


@pytest.mark.parametrize("pattern", ["alternating", "stall"])
def test_vfr_clip_flagged_and_aligned(tmp_path, pattern):
    base = 1 / 30
    durations = ([base, 2 * base] * 12) if pattern == "alternating" else ([base] * 10 + [5 * base] + [base] * 13)
    clip = make_vfr(tmp_path, durations)
    info = timebase.probe(clip)
    # Ground truth from the VFR file itself: decoded content and presentation times.
    shown = [source_index(f) for f in decode(clip)]
    assert info["cfr"] is False and info["frames"] == len(shown)
    path, record = timebase.ensure_cfr(clip, tmp_path / "job")
    assert record["normalized"] is True and timebase.probe(path)["cfr"] is True
    fps, starts = record["normalized_fps"], np.asarray(info["pts_seconds"])
    errors = []
    for i, frame in enumerate(decode(path)):
        expected = shown[int(np.searchsorted(starts, i / fps + 1e-6, side="right") - 1)]
        errors.append(abs(source_index(frame) - expected))
    assert errors and max(errors) == 0
    assert abs(len(errors) / fps - info["duration"]) <= 1.5 / fps
