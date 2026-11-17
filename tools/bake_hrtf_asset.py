"""選定した耳介を、配布用の軽量アセットに焼き込む。

SOFA は 1 ファイル 2.7MB で、読み込みに sofar と netCDF が要る。
水平面に絞って .npz にすれば 200KB 台になり、numpy だけで読めるので、
PyInstaller で固めるゲームに同梱しやすい。全天球のまま .npz にすると
約1.4MB で、上下も鳴らせる（default_hrir(full_sphere=True)）。

    python -X utf8 tools\\bake_hrtf_asset.py                   # 水平面（耳介E）
    python -X utf8 tools\\bake_hrtf_asset.py --keep-elevation  # 全天球（耳介E）
    python -X utf8 tools\\bake_hrtf_asset.py --pinna F --out 別の場所.npz

出力: 水平面は audiospace/data/default_horizontal.npz、
全天球（--keep-elevation）は audiospace/data/default_sphere.npz

表示義務の文面（npz の attribution）の「改変」は、実際にした変換だけを書く
（元のレートと受信点の数は SOFA から読む）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import HrirSet  # noqa: E402
from audiospace.defaults import (ASSET_DIR, DEFAULT_ASSET,  # noqa: E402
                                 DEFAULT_PINNA, DEFAULT_SPHERE_ASSET)

VIKING_DIR = ASSET_DIR / "viking"

# v2 の著者と、データセットが併せて引用を求める初版論文の著者は異なる
# （documentation.pdf）。CC BY の表示を確実に満たすため両方を載せる。
# 改変の内容は焼き込み条件で変わるので、文面は main() で組み立てる。
ATTRIBUTION_EN = (
    "This software includes HRTF data from the Viking HRTF Dataset v2 "
    "by Spagnol, Miccini and Unnthorsson (2020, "
    "https://doi.org/10.5281/zenodo.4160401), based on The Viking HRTF "
    "dataset by Spagnol, Purkhus, Bjornsson and Unnthorsson (SMC 2019), "
    "used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Modified: {mod_en}."
)
ATTRIBUTION_JA = (
    "本ソフトウェアは Viking HRTF Dataset v2"
    "（Spagnol, Miccini, Unnthorsson 作、2020年、"
    "https://doi.org/10.5281/zenodo.4160401）の HRTF データを含みます。"
    "元は The Viking HRTF dataset（Spagnol, Purkhus, Bjornsson, "
    "Unnthorsson 作、SMC 2019）です。CC BY 4.0"
    "（https://creativecommons.org/licenses/by/4.0/）のもとで使用しています。"
    "改変: {mod_ja}。"
)


def sofa_facts(path: Path) -> tuple[int, int]:
    """SOFA の元のサンプルレートと受信点の数。"""
    import contextlib
    import io

    import sofar
    with contextlib.redirect_stdout(io.StringIO()):
        sofa = sofar.read_sofa(str(path))
    return (int(np.ravel(sofa.Data_SamplingRate)[0]),
            int(np.asarray(sofa.Data_IR).shape[1]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="選定した耳介を配布用アセットに焼き込む")
    parser.add_argument("--pinna", default=DEFAULT_PINNA,
                        help=f"耳介の記号 A-T または Z (既定 {DEFAULT_PINNA})")
    parser.add_argument("--keep-elevation", action="store_true",
                        help="仰角を残す（3Dゲーム用。表が約7倍になる）")
    parser.add_argument("--decimate", type=float, default=0.0,
                        help="方位をこの刻み（度）に間引く。0で間引かない")
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--out", default=None,
                        help="出力先。既定は水平面なら default_horizontal.npz、"
                             "--keep-elevation なら default_sphere.npz")
    args = parser.parse_args()
    if args.out is None:
        args.out = str(DEFAULT_SPHERE_ASSET if args.keep_elevation
                       else DEFAULT_ASSET)

    source = VIKING_DIR / f"subj_{args.pinna}.sofa"
    if not source.exists():
        print(f"[エラー] {source} がありません。"
              "tools\\fetch_viking_hrtf.py で取得してください")
        return 2

    full = HrirSet.from_sofa(source, sample_rate=args.rate)
    print(f"読み込み: {source.name} "
          f"{full.n_directions}方向 {full.taps}タップ {full.sample_rate}Hz "
          f"(表 {full.table_bytes / 1024 / 1024:.1f} MB)")

    hrir = full if args.keep_elevation else full.horizontal()
    if not args.keep_elevation:
        print(f"水平面に絞る: {hrir.n_directions}方向 "
              f"(表 {hrir.table_bytes / 1024:.0f} KB)")
    if args.decimate > 0:
        hrir = hrir.decimated(args.decimate)
        print(f"{args.decimate:g}度刻みに間引く: {hrir.n_directions}方向")

    native_rate, receivers = sofa_facts(source)
    mod_en: list[str] = []
    mod_ja: list[str] = []
    if native_rate != args.rate:
        mod_en.append(f"resampled to {args.rate} Hz")
        mod_ja.append(f"{args.rate}Hz にリサンプリング")
    if receivers > 2:
        mod_en.append("truncated to two receivers")
        mod_ja.append("受信点を2つに削減")
    if not args.keep_elevation:
        mod_en.append("restricted to the horizontal plane")
        mod_ja.append("水平面のみに限定")
    if args.decimate > 0:
        mod_en.append(f"azimuth decimated to {args.decimate:g} deg")
        mod_ja.append(f"方位を{args.decimate:g}度刻みに間引き")
    if mod_en:
        mod_en.append("converted from SOFA to the NumPy npz format")
        mod_ja.append("SOFA から NumPy の npz 形式に変換")
    else:
        mod_en.append("converted from SOFA to the NumPy npz format "
                      "(the measured values are unchanged)")
        mod_ja.append("SOFA から NumPy の npz 形式に変換（測定値は変えていない）")

    out = Path(args.out)
    path = hrir.save_npz(
        out,
        source=source.name,
        pinna=args.pinna,
        dataset="Viking HRTF Dataset v2",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        doi="https://doi.org/10.5281/zenodo.4160401",
        attribution=ATTRIBUTION_EN.format(mod_en=", ".join(mod_en)),
        attribution_ja=ATTRIBUTION_JA.format(mod_ja="・".join(mod_ja)),
        modifications="; ".join(mod_en),
        full_sphere=bool(args.keep_elevation),
        selection_note=("聴取テスト（12方向・首振りあり）で選定。"
                        "上位3種は同着だったが、この耳介のみ"
                        "判別可能な難易度72問で前後の取り違えゼロ。"),
    )

    size_kb = path.stat().st_size / 1024
    print(f"\n書き出し: {path}")
    print(f"サイズ: {size_kb:.0f} KB "
          f"(元の SOFA は {source.stat().st_size / 1024:.0f} KB)")

    # 読み直して健全性を確認する
    back = HrirSet.load_npz(path)
    assert back.n_directions == hrir.n_directions
    assert np.allclose(back.data, hrir.data)
    ir = back.data[back.index_for(90.0, 0.0)]
    lead_l = int(np.argmax(np.abs(ir[0]) > 0.15 * np.max(np.abs(ir[0]))))
    lead_r = int(np.argmax(np.abs(ir[1]) > 0.15 * np.max(np.abs(ir[1]))))
    itd_us = (lead_r - lead_l) / back.sample_rate * 1e6
    print(f"読み直し確認: {back.n_directions}方向、"
          f"90度でのITD {itd_us:+.0f}マイクロ秒（左耳が先行）")
    if itd_us <= 0:
        print("[警告] ITD が負です。左右が入れ替わっている可能性があります")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
