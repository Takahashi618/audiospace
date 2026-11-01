"""材質の吸音率の表を、pyroomacoustics から audiospace のデータへ書き出す。

audiospace は実行時に pyroomacoustics を要らないようにしているので、材質表は
パッケージのデータ（audiospace/data/materials.json）として持つ。表を更新する
ときだけこのスクリプトを走らせる（pyroomacoustics が要る）。

- 帯域は 125 / 250 / 500 / 1000 / 2000 / 4000 / 8000 Hz の7つ
- 表には帯域が6つしか無い材質がある（8000Hz が無い）。最後の値を延ばして
  7つにそろえ、そろえたことを `padded` に残す
- 「open」（吸音率1、何も返らない）を足す。屋外の空や、道の先の開いた面に使う

表の出典は pyroomacoustics（MIT、Copyright (c) 2014-2017 EPFL-LCAV）。
表示は ATTRIBUTION.md。

Run: python -X utf8 tools\\build_materials.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = (Path(__file__).resolve().parents[1] / "audiospace" / "data"
       / "materials.json")


def main() -> int:
    import pyroomacoustics as pra

    data = pra.materials_data
    centers = [int(f) for f in data["center_freqs"]]
    out: dict[str, dict] = {}
    for category, table in data["absorption"].items():
        for name, entry in table.items():
            coeffs = [float(c) for c in entry["coeffs"]]
            padded = len(coeffs) < len(centers)
            coeffs = coeffs + [coeffs[-1]] * (len(centers) - len(coeffs))
            out[name] = {"category": category,
                         "description": entry.get("description", ""),
                         "absorption": coeffs[:len(centers)],
                         "padded": padded}
    out["open"] = {"category": "audiospace",
                   "description": "Open side: sky, or a street running on "
                                  "(nothing comes back)",
                   "absorption": [1.0] * len(centers), "padded": False}
    payload = {
        "source": f"pyroomacoustics {pra.__version__} materials database "
                  "(MIT, Copyright (c) 2014-2017 EPFL-LCAV), plus 'open'",
        "center_freqs": centers,
        "materials": out,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"{len(out)} materials -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
