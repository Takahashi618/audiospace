"""材質つきの空間（set_space）の処理時間を測る（デバイスは開かない）。

比べるのは hall の IR 残響と、空間 2種（コンクリートの部屋、建物の間の道）。
全音源が動き、首振りあり、聞き手も少し動く（反射の遅れが毎ブロック変わる）。

性能コアと効率コアのある CPU では、効率コアで約2倍遅くなることがあるので、
コアを固定して測ること。例: PowerShell で起動直後に
$p.ProcessorAffinity = [IntPtr]0xFFFF（性能コア）／0xF0000（効率コア）。

Run: python -X utf8 tools\\bench_space.py <repo> <ラベル>
"""
import os, sys, time
from pathlib import Path
import numpy as np
os.environ["AUDIOSPACE_SILENT"] = "1"
repo = sys.argv[1]; label = sys.argv[2]
sys.path.insert(0, repo)
import audiospace as lib
SR, BLOCK = 48000, 512
full = lib.HrirSet.from_sofa(Path(repo) / "assets" / "hrtf" / "viking" / "subj_E.sofa", sample_rate=SR)
def run(n, space):
    e = lib.SpatialEngine(full, block=BLOCK, max_sources=n)
    e.set_head_wobble(lib.WOBBLE_PRESETS["clear"])
    p = lib.SpatialPlayer(e)
    if space: p.set_space(space)
    else: p.use_room_ir("hall")
    rng = np.random.default_rng(3); noise = (rng.standard_normal(SR) * 0.02).astype(np.float32)
    hs = [e.add_source(lib.BufferSource(noise, loop=True), reverb_send=0.3,
          position=(float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8)), float(rng.uniform(-1, 1)))) for _ in range(n)]
    t = []
    for i in range(110):
        a = i * 0.05
        for k, h in enumerate(hs):
            e.set_source_position(h, (6 * np.cos(a + k), 6 * np.sin(a + k), 0.5 * np.sin(a + k)))
        e.set_listener((0.3 * np.sin(i * 0.02), 0.0, 0.0))   # 聞き手も少し動く（反射の遅れが変わる）
        t0 = time.perf_counter(); e.render(); t.append((time.perf_counter() - t0) * 1000)
    a = np.array(t[10:]); return a.mean(), np.percentile(a, 99)
print(f"[{label}]")
for n in (16, 128, 256):
    for sp in (None, "concrete_room", "street"):
        m, q = run(n, sp)
        print(f"  {n:>3}音源 {('hall IR' if sp is None else sp):<14} 平均 {m:5.2f} / 99% {q:5.2f} ms", flush=True)
