"""材質つきの空間の後期残響を、精密な計算で作る（焼き込みとキャッシュ）。

`Space` の面ごとの材質（7帯域の吸音率）と拡散の度合いを pyroomacoustics
（MIT）に渡し、像の方法（3次まで）と光線追跡を組み合わせて部屋の
インパルス応答（IR）を計算する。耳の位置に 17cm 離した2つのマイクを置くので、
IR は左右で少しずつ違う（響きが広がって聞こえる）。

使うのは IR の「尾」だけ。反射が混ざり合って拡散した後（部屋の大きさに応じて
50〜150ms 以降）を切り出し、頭を滑らかに立ち上げる。それより前の、方向の
はっきりした反射はエンジンの初期反射（`early.EarlyReflections`）が受け持つ
（聞き手の位置と向きに合わせて方向つきで鳴らせるため）。尾の大きさは、速い版
（`BandFdnReverb`）と同じエネルギーにそろえる（切り替えても音量が変わらない）。

計算には1部屋あたり数秒〜十数秒かかる（光線2万本、2マイク。5m×4m の
コンクリートの部屋で約13秒）。ゲーム中には回せないので:

- 用意した空間は `tools/build_space_irs.py` で焼き込み、`assets/space_ir/` に置く
- それ以外の空間は、初めて使うときに計算して `cache_dir` に保存する
  （エンジンは計算中は速い版で鳴らし、できたらつなぎ替える）

焼き込みのファイル名は空間の中身（大きさ・材質・拡散）から作る鍵で決まる。
置き場所（origin）は IR に関係しないので鍵に含めない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .acoustics import FACES, Space, absorption, band_centers
from .i18n import tr

#: 焼き込み済みの IR の置き場所（リポジトリの構成）。wheel には無い
SPACE_IR_DIR = Path(__file__).resolve().parent.parent / "assets" / "space_ir"
#: 頭の立ち上げの長さ（秒）
TAIL_FADE_S = 0.02
#: 終わりのフェードの長さ（秒）
TAIL_END_FADE_S = 0.1


def space_key(space: Space) -> str:
    """空間の中身から作る鍵（焼き込みのファイル名に使う）。置き場所は含めない。"""
    payload = {
        "size": [round(float(v), 4) for v in space.size],
        "materials": {f: [round(float(a), 5)
                          for a in absorption(space.materials[f])]
                      for f in FACES},
        "scattering": round(float(space.scattering), 4),
    }
    # 置いてあるものは、あるときだけ鍵に入れる（入れると、焼き込み済みの
    # 尾のファイル名が変わって見つからなくなる）
    if space.contents:
        payload["contents"] = {k: int(v)
                               for k, v in sorted(space.contents.items())}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def supports_precise(space: Space) -> bool:
    """精密版の尾を作れる（使ってよい）空間か。

    閉じた部屋で、拡散した残響があり、置いてあるもの（家具や人）が無い
    こと。開いた面があると光線が抜けて尾が途切れる。家具や人の吸音は
    pyroomacoustics に渡せないので、物がある空間は速い版で鳴らす（速い版は
    物の吸音を含む）。エンジンの準備・選択と、計算そのものが同じ判定を使う。
    """
    return (space.late_level() > 0.0 and space.open_fraction == 0.0
            and not space.contents)


def tail_start_s(space: Space) -> float:
    """尾を切り出し始める時刻（秒）。部屋が大きいほど遅い。

    反射が混ざり合って方向が分からなくなるまでの時間（混合時間）の目安として、
    50ms に体積 1m³ あたり 0.01ms を足し、150ms で打ち切る。
    """
    return min(0.150, 0.050 + 0.00001 * space.volume)


@dataclass
class SpaceIr:
    """精密に計算した空間の尾（後期残響の IR）。"""

    key: str
    name: str
    tail: np.ndarray                    # (2, taps) float32
    sample_rate: int
    rt60_bands: np.ndarray = field(default_factory=lambda: np.zeros(7))
    meta: dict = field(default_factory=dict)

    def resampled(self, sample_rate: int) -> "SpaceIr":
        if sample_rate == self.sample_rate:
            return self
        from math import gcd

        from scipy import signal

        g = gcd(int(self.sample_rate), int(sample_rate))
        out = signal.resample_poly(self.tail, sample_rate // g,
                                   self.sample_rate // g,
                                   axis=-1).astype(np.float32)
        return SpaceIr(self.key, self.name, out, int(sample_rate),
                       self.rt60_bands,
                       dict(self.meta, resampled_from=self.sample_rate))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tail=self.tail, key=self.key,
                            name=self.name, sample_rate=self.sample_rate,
                            rt60_bands=self.rt60_bands,
                            meta=json.dumps(self.meta, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SpaceIr":
        data = np.load(Path(path), allow_pickle=False)
        return cls(str(data["key"]), str(data["name"]),
                   data["tail"].astype(np.float32), int(data["sample_rate"]),
                   np.asarray(data["rt60_bands"], dtype=np.float64),
                   json.loads(str(data["meta"])))


def render_space_ir(space: Space, sample_rate: int = 48000,
                    max_seconds: float = 6.0, n_rays: int = 20000,
                    match_level: bool = True) -> SpaceIr | None:
    """空間の後期残響の尾を、pyroomacoustics で精密に計算する。

    閉じた部屋だけを計算する。開いた面のある空間（建物の間の道など）、
    拡散した残響が無い空間、家具や人を置いた空間は None を返す（エンジンは
    速い版で鳴らす。`supports_precise`）。
    開いた面があると光線の多くが外へ抜け、残った光線が向かい合う壁の間を
    往復し続けて、計算の打ち切りで尾が途切れる。建物の間の道で試すと、
    0.9秒までほとんど減衰せず、そこで10ms のうちに30dB 落ちた（響きが
    急に止まる）。pyroomacoustics が無ければ ImportError。
    """
    if not supports_precise(space):
        return None
    try:
        import pyroomacoustics as pra
    except ImportError as exc:            # pragma: no cover
        raise ImportError(
            "precise room simulation needs pyroomacoustics: "
            "python -m pip install pyroomacoustics") from exc

    centers = list(band_centers())
    mats = {f: pra.Material(
        {"coeffs": [float(a) for a in absorption(space.materials[f])],
         "center_freqs": centers}, float(space.scattering))
        for f in FACES}
    w, d, h = (float(v) for v in space.size)
    room = pra.ShoeBox([w, d, h], fs=int(sample_rate), materials=mats,
                       max_order=3, air_absorption=True, ray_tracing=True)
    room.set_ray_tracing(receiver_radius=0.5, n_rays=int(n_rays),
                         energy_thres=1e-7)
    # 聞き手と音源は、真ん中を避けて斜めに置く（箱の真ん中では左右の反射が
    # そろって不自然になる）。presets.render_room_ir と同じ考え方
    listener = np.array([w * 0.42, d * 0.38, min(1.65, h * 0.55)])
    half = 0.17 / 2.0
    ears = np.c_[listener, listener]
    ears[0, 0] -= half
    ears[0, 1] += half
    room.add_microphone_array(pra.MicrophoneArray(ears, int(sample_rate)))
    source = np.array([w * 0.68, d * 0.72, min(1.6, h * 0.5)])
    source = np.clip(source, 0.2, np.array([w, d, h]) - 0.2)
    room.add_source(source.tolist())
    room.compute_rir()

    limit = int(max_seconds * sample_rate)
    chans = [np.asarray(room.rir[m][0], dtype=np.float64)[:limit]
             for m in range(2)]
    n = max(len(c) for c in chans)
    ir = np.zeros((2, n))
    for m, c in enumerate(chans):
        ir[m, :len(c)] = c

    # 尾の切り出し: 混合時間の手前から 20ms で滑らかに立ち上げる
    start = int(tail_start_s(space) * sample_rate)
    fade = int(TAIL_FADE_S * sample_rate)
    lead = max(0, start - fade)
    window = np.zeros(n)
    window[start:] = 1.0
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, start - lead,
                                          endpoint=False))
    window[lead:start] = ramp
    tail = ir * window[None, :]
    tail = tail[:, lead:]               # 頭の無音は詰める（遅れは lead 秒）
    # 終わりも滑らかに消す。計算の打ち切りで尾が角張って終わると、
    # 小さな音でも「ぷつっ」と止まって聞こえる
    end = min(int(TAIL_END_FADE_S * sample_rate), tail.shape[1] // 4)
    if end > 0:
        tail[:, -end:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, end))

    if match_level:
        target = _fast_tail_energy(space, int(sample_rate))
        energy = float(np.sum(tail ** 2))
        if energy > 0.0 and target > 0.0:
            tail *= np.sqrt(target / energy)

    rt = _band_rt60(ir.mean(axis=0), int(sample_rate))
    return SpaceIr(space_key(space), space.name, tail.astype(np.float32),
                   int(sample_rate), rt,
                   {"tail_start_s": tail_start_s(space),
                    "lead_s": lead / sample_rate, "n_rays": int(n_rays),
                    "listener": listener.tolist(), "source": source.tolist(),
                    "pyroomacoustics": pra.__version__})


def _fast_tail_energy(space: Space, sample_rate: int) -> float:
    """速い版（BandFdnReverb）の後期残響の、インパルスへの応答のエネルギー。

    精密版の尾をこれにそろえれば、速い版と切り替えても音量が変わらない。
    """
    from .reverb import band_reverb_for

    block = 512
    rev = band_reverb_for(space, sample_rate, block)
    seconds = float(min(12.0, max(1.0, space.rt60_bands().max() * 1.5)))
    x = np.zeros((int(seconds * sample_rate / block) + 1, block),
                 dtype=np.float32)
    x[0, 0] = 1.0
    y = np.concatenate([rev.process(b) for b in x])
    return float(np.sum(y.astype(np.float64) ** 2))


def _band_rt60(ir: np.ndarray, sample_rate: int) -> np.ndarray:
    """IR の帯域ごとの残響時間（7帯域、T20 から外挿）。測れない帯域は 0。"""
    from scipy import signal

    out = []
    for fc in band_centers():
        hi = min(fc * np.sqrt(2), sample_rate / 2 * 0.95)
        lo = fc / np.sqrt(2)
        if lo >= hi:
            out.append(0.0)
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=sample_rate,
                            output="sos")
        y = signal.sosfilt(sos, ir)
        e = np.cumsum(y[::-1] ** 2)[::-1]
        if e[0] <= 0:
            out.append(0.0)
            continue
        db = 10 * np.log10(np.maximum(e / e[0], 1e-12))
        s, t = int(np.argmax(db <= -5)), int(np.argmax(db <= -25))
        if t <= s:
            out.append(0.0)
            continue
        slope = np.polyfit(np.arange(s, t) / sample_rate, db[s:t], 1)[0]
        out.append(float(-60 / slope) if slope < 0 else 0.0)
    return np.asarray(out)


_MEMORY: dict[tuple[str, int], SpaceIr] = {}


def find_space_ir(space: Space, sample_rate: int,
                  cache_dir: str | Path | None = None) -> SpaceIr | None:
    """焼き込み済み・保存済みの尾を探す（計算はしない）。無ければ None。

    探す順: メモリ → cache_dir（無ければ default_cache_dir()）→ assets/space_ir。
    レートが違えば変換する。
    """
    key = space_key(space)
    hit = _MEMORY.get((key, int(sample_rate)))
    if hit is not None:
        return hit
    # cache_dir が無いときは、start_render_process が書く既定の場所を探す
    # （探さないと、計算した尾が次に起動したとき見つからず、毎回計算し直す）
    folders = [Path(cache_dir) if cache_dir else default_cache_dir(),
               SPACE_IR_DIR]
    for folder in folders:
        if not folder.exists():
            continue
        exact = folder / f"{key}_{int(sample_rate)}.npz"
        candidates = ([exact] if exact.exists()
                      else sorted(folder.glob(f"{key}_*.npz")))
        if candidates:
            found = SpaceIr.load(candidates[0]).resampled(int(sample_rate))
            _MEMORY[(key, int(sample_rate))] = found
            return found
    return None


def get_space_ir(space: Space, sample_rate: int,
                 cache_dir: str | Path | None = None,
                 render: bool = True) -> SpaceIr | None:
    """尾を返す。無ければ計算して cache_dir に保存する（render=True のとき）。

    計算には数秒〜十数秒かかる。音声スレッドから呼ばないこと（エンジンは
    別のスレッドで呼ぶ）。
    """
    found = find_space_ir(space, sample_rate, cache_dir)
    if found is not None or not render:
        return found
    made = render_space_ir(space, sample_rate)
    if made is None:
        return None
    _MEMORY[(made.key, int(sample_rate))] = made
    if cache_dir:
        made.save(Path(cache_dir) / f"{made.key}_{int(sample_rate)}.npz")
    return made


# ---------------------------------------------------------------------------
# 別のプロセスでの計算
# ---------------------------------------------------------------------------
# pyroomacoustics は計算中 Python の GIL を握り続ける。同じプロセスの別スレッドで
# 計算させると、その間エンジンの描画が止まる（実測で最大12.8秒、音が途切れる）。
# そこで計算は別のプロセス（python -m audiospace.precise）で行い、結果を
# ファイルで受け取る。


def default_cache_dir() -> Path:
    """計算した尾の既定の保存先（一時フォルダの下）。"""
    import tempfile
    return Path(tempfile.gettempdir()) / "audiospace_space_ir"


def can_render_in_background() -> bool:
    """別のプロセスで計算できるか（pyroomacoustics があり、凍結した exe でない）。

    PyInstaller などで固めた exe では `python -m` で自分を呼べないので、
    焼き込み済みの尾しか使えない。
    """
    import importlib.util
    import sys
    if getattr(sys, "frozen", False):
        return False
    return importlib.util.find_spec("pyroomacoustics") is not None


def start_render_process(space: Space, sample_rate: int,
                         cache_dir: str | Path | None = None):
    """尾の計算を別のプロセスで始める。(Popen, 出力ファイルのパス) を返す。

    出力は cache_dir（無ければ一時フォルダの下）の「鍵_レート.npz」。
    """
    import os
    import subprocess
    import sys
    import tempfile

    folder = Path(cache_dir) if cache_dir else default_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{space_key(space)}_{int(sample_rate)}.npz"
    # mkstemp が開いたままの手続き番号で書いて閉じる。閉じないと漏れるうえ、
    # Windows では開いたままのファイルを子が消せず、一時フォルダに残り続ける
    fd, name = tempfile.mkstemp(suffix=".json", prefix="space_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(space.to_dict(), ensure_ascii=False))
    spec = Path(name)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "audiospace.precise",
         "--space", str(spec), "--rate", str(int(sample_rate)),
         "--out", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=flags,
        cwd=str(Path(__file__).resolve().parent.parent))
    return proc, out


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=tr("空間の後期残響の尾を精密に計算して保存する",
                       "Compute the late reverb tail of a space precisely "
                       "and save it"))
    parser.add_argument("--space", required=True,
                        help=tr("Space.to_dict の JSON",
                                "JSON from Space.to_dict"))
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = Path(args.space)
    space = Space.from_dict(json.loads(spec.read_text(encoding="utf-8")))
    try:
        spec.unlink()
    except OSError:
        pass
    made = render_space_ir(space, args.rate)
    if made is None:
        return 3                         # 拡散した残響の無い空間
    # 書きかけのファイルを読まれないよう、別名で書いてから置き換える
    out = Path(args.out)
    tmp = out.with_name(out.stem + ".part.npz")
    made.save(tmp)
    tmp.replace(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
