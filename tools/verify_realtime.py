"""Real-device verification for audiospace -- deliberately inaudible.

Runs the full high-level stack (device selection, WASAPI stream, engine,
convolution reverb) on the actual sound card, but every source is rendered
at zero gain so nothing can be heard.  What we are measuring is callback
timing, dropout count and the selected latency.

    python -X utf8 tools\\verify_realtime.py
    python -X utf8 tools\\verify_realtime.py --audible   (only if you want to listen)

Out: tools/verify_realtime_report.txt (UTF-8)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import (BufferSource, ConvolutionReverb,  # noqa: E402
                        SpatialPlayer, VectorFdnReverb, describe_devices,
                        load_all)

REPORT = Path(__file__).with_name("verify_realtime_report.txt")
_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def noise(seconds: float, rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)).astype(np.float32) * 0.2)


def run(n_sources: int, seconds: float, audible: bool,
        reverb_kind: str) -> str:
    player = SpatialPlayer.open(block=512, max_sources=max(64, n_sources),
                                reverb=None)
    rate = player.engine.sample_rate
    if reverb_kind == "convolution":
        rooms = load_all(rate)
        if rooms:
            name = "hall" if "hall" in rooms else sorted(rooms)[0]
            player.engine.set_reverb(
                ConvolutionReverb(rooms[name].ir, block=player.engine.block))
            reverb_label = f"convolution ({name}, {rooms[name].ir.shape[1]} taps)"
        else:
            player.engine.set_reverb(VectorFdnReverb(rate, rt60=1.6))
            reverb_label = "FDN (no baked IR found for this rate)"
    else:
        player.engine.set_reverb(VectorFdnReverb(rate, rt60=1.6))
        reverb_label = "FDN"

    gain = 1.0 if audible else 0.0
    handles = []
    rng = np.random.default_rng(4)
    for i in range(n_sources):
        handles.append(player.play(
            noise(1.0, rate, i), loop=True, gain=gain, reverb_send=0.3,
            position=(float(rng.uniform(-8, 8)),
                      float(rng.uniform(-8, 8)), 0.0)))

    # move every source continuously, which is the expensive case because
    # the convolver has to crossfade between HRIR directions each block
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        t = time.perf_counter() - started
        for k, h in enumerate(handles):
            angle = t * 1.4 + k * (2 * np.pi / max(1, n_sources))
            player.engine.set_source_position(
                h, (6.0 * float(np.cos(angle)), 6.0 * float(np.sin(angle)),
                    0.0))
        player.engine.set_listener(yaw_deg=(t * 40.0) % 360.0)
        time.sleep(0.01)

    budget = player.budget_ms
    line = (f"{n_sources:>4} sources / {reverb_label:<44} "
            f"drops={player.underruns:>3} "
            f"mean={player.mean_ms:5.2f} ms ({player.mean_ms / budget * 100:4.1f}%) "
            f"worst={player.worst_ms:5.2f} ms "
            f"({player.worst_ms / budget * 100:5.1f}%) "
            f"first={player.first_ms:5.2f} ms")
    player.stop()
    return line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audible", action="store_true",
                        help="actually make sound (default: silent)")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    log("audiospace real-device verification")
    log(f"output is {'AUDIBLE' if args.audible else 'MUTED (gain=0)'}")
    log("=" * 78)
    log("lowest-latency outputs available:")
    log("\n".join(describe_devices().splitlines()[:5]))
    log("")

    for kind in ("fdn", "convolution"):
        for n in (8, 32, 64):
            try:
                log(run(n, args.seconds, args.audible, kind))
            except Exception as exc:                    # noqa: BLE001
                log(f"[エラー] {n} sources / {kind}: {exc!r}")
            log("")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
