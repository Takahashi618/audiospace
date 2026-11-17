"""定位確認ツールの設定欄の選択を、次の起動に引き継ぐ。

GUI に依存しない。選択は並び順ではなく中身の名前で持つ（HRTF はファイルの
名前、残響と空間は種類と名前、難易度は名前）。並び順で持つと、耳介の
ファイルや自分で作った残響が増えたり消えたりしたときに、別のものを指して
しまう。

    {"version": 1,
     "settings": {"hrtf": "subj_E.sofa", "sound": "noise_bursts",
                  "wobble": "clear", "reverb": ["preset", "small_room"],
                  "space": null, "furnish": "empty", "people": 0,
                  "precise": false, "difficulty": "FOUR", "front_back": 35}}

保存しないもの（利用者が起動時の状態を決めたもの。2026-10-15・16）:

- 首かしげ・上下の強調・床の反射（上下の手がかり。起動時はオフ）
- 繰り返し再生と基準音（テンキーで切り替える。起動時はオン）
- 向き（いつも正面・水平から始める）

ファイルが壊れていても起動は止めない。壊れたファイルは .broken を付けて
残し、起動時の状態から始めて、そのことを `load_error` に書く。書き込みは
一時ファイルに書いてから置き換える（途中で落ちても元のファイルが壊れない）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from audiospace.i18n import tr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "settings" / "app_settings.json"

#: 保存する項目
KEYS = ("hrtf", "sound", "wobble", "reverb", "space", "furnish", "people",
        "precise", "difficulty", "front_back")


class AppSettingsStore:
    """設定欄の選択の保存先。値の確かめと当てはめはアプリが行う。"""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.values: dict = {}
        self.load_error = ""
        self.load()

    def load(self) -> None:
        self.values, self.load_error = {}, ""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            values = dict(data.get("settings", {}))
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            broken = self.path.with_suffix(self.path.suffix + ".broken")
            try:
                os.replace(self.path, broken)
            except OSError:
                pass
            self.load_error = tr(
                f"設定のファイルが読めないため、{broken.name} に移して"
                f"起動時の状態から始めます（{exc}）",
                f"The settings file cannot be read, so it was moved to "
                f"{broken.name} and the tool starts with its usual "
                f"settings ({exc}).")
            return
        self.values = {k: values[k] for k in KEYS if k in values}

    def save(self, values: dict) -> bool:
        """変わっていれば書く（書いたら True）。失敗は OSError のまま上げる。"""
        values = {k: values[k] for k in KEYS if k in values}
        if values == self.values and self.path.exists():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "settings": values}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)
        self.values = values
        return True
