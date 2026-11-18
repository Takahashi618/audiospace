"""材質つきの空間（set_space）の処理時間を測る（デバイスは開かない）。

比べるのは hall の IR 残響と、空間 2種（コンクリートの部屋、建物の間の道）。
全音源が動き、首振りあり、聞き手も少し動く（反射の遅れが毎ブロック変わる）。

性能コアと効率コアのある CPU では、効率コアで約2倍遅くなることがあるので、
コアを固定して測ること。例: PowerShell で起動直後に
$p.ProcessorAffinity = [IntPtr]0xFFFF（性能コア）／0xF0000（効率コア）。

Run: python -X utf8 tools\\bench_space.py [<repo> [<ラベル>]]

<repo> は測るライブラリのあるフォルダ（省くと、この道具のあるフォルダの親）。
別の版を展開したフォルダを渡せば、版どうしを比べられる。<ラベル> は結果の
見出し（省くと repo のフォルダの名前）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"

SR, BLOCK = 48000, 512


def run(lib, full, n: int, space: str | None) -> tuple[float, float]:
    e = lib.SpatialEngine(full, block=BLOCK, max_sources=n)
    e.set_head_wobble(lib.WOBBLE_PRESETS["clear"])
    p = lib.SpatialPlayer(e)
    if space:
        p.set_space(space)
    else:
        p.use_room_ir("hall")
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(SR) * 0.02).astype(np.float32)
    hs = [e.add_source(lib.BufferSource(noise, loop=True), reverb_send=0.3,
                       position=(float(rng.uniform(-8, 8)),
                                 float(rng.uniform(-8, 8)),
                                 float(rng.uniform(-1, 1))))
          for _ in range(n)]
    t = []
    for i in range(110):
        a = i * 0.05
        for k, h in enumerate(hs):
            e.set_source_position(h, (6 * np.cos(a + k), 6 * np.sin(a + k),
                                      0.5 * np.sin(a + k)))
        # 聞き手も少し動く（反射の遅れが変わる）
        e.set_listener((0.3 * np.sin(i * 0.02), 0.0, 0.0))
        t0 = time.perf_counter()
        e.render()
        t.append((time.perf_counter() - t0) * 1000)
    arr = np.array(t[10:])
    return float(arr.mean()), float(np.percentile(arr, 99))


def main(argv: list[str]) -> int:
    if len(argv) > 2 or any(a in ("-h", "--help") for a in argv):
        print("使い方: python -X utf8 tools\\bench_space.py [<repo> [<ラベル>]]")
        return 0 if len(argv) <= 2 else 2
    repo = Path(argv[0]) if argv else Path(__file__).resolve().parents[1]
    label = argv[1] if len(argv) > 1 else repo.resolve().name
    sofa = repo / "assets" / "hrtf" / "viking" / "subj_E.sofa"
    if not sofa.is_file():
        print(f"[エラー] {sofa} がありません（<repo> はライブラリのあるフォルダ）")
        return 2
    sys.path.insert(0, str(repo))
    import audiospace as lib

    full = lib.HrirSet.from_sofa(sofa, sample_rate=SR)
    print(f"[{label}]")
    for n in (16, 128, 256):
        for sp in (None, "concrete_room", "street"):
            m, q = run(lib, full, n, sp)
            name = "hall IR" if sp is None else sp
            print(f"  {n:>3}音源 {name:<14} 平均 {m:5.2f} / 99% {q:5.2f} ms",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
