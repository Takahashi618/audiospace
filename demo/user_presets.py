"""自分で作った空間と残響を、名前を付けて保存する。

GUI に依存しない。中身は audiospace の設計図（`space_from_spec` と
`reverb_from_spec` が受け取る辞書）をそのまま JSON に書いたもの。

    {"version": 1,
     "spaces":  {"名前": {空間の設計図}, ...},
     "reverbs": {"名前": {残響の設計図}, ...}}

ファイルが壊れていても起動は止めない。壊れたファイルは .broken を付けて
残し（消すと作ったものが失われる）、空の状態から始めて、そのことを
`load_error` に書く。書き込みは一時ファイルに書いてから置き換える
（途中で落ちても元のファイルが壊れない）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from audiospace import reverb_spec_problems, space_spec_problems
from audiospace.i18n import list_sep, tr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "settings" / "user_presets.json"

#: 名前の長さの上限（読み上げで長すぎないように）
NAME_MAX = 40


def name_problems(name: str) -> list[str]:
    """名前の問題点。空・長すぎる・前後の空白だけ、を断る。"""
    text = name.strip()
    if not text:
        return [tr("名前を入れてください", "Enter a name")]
    if len(text) > NAME_MAX:
        return [tr(f"名前は{NAME_MAX}文字以内にしてください",
                   f"Use a name of {NAME_MAX} characters or fewer")]
    return []


class UserPresetStore:
    """自分で作った空間と残響の保存先。"""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.spaces: dict[str, dict] = {}
        self.reverbs: dict[str, dict] = {}
        self.load_error = ""
        self.load()

    # -- 読み書き -------------------------------------------------------
    def load(self) -> None:
        self.spaces, self.reverbs, self.load_error = {}, {}, ""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            spaces = dict(data.get("spaces", {}))
            reverbs = dict(data.get("reverbs", {}))
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            broken = self.path.with_suffix(self.path.suffix + ".broken")
            try:
                os.replace(self.path, broken)
            except OSError:
                pass
            self.load_error = tr(
                f"自分で作った空間と残響のファイルが読めないため、"
                f"{broken.name} に移して空から始めます（{exc}）",
                f"The file of your own spaces and reverbs cannot be read, "
                f"so it was moved to {broken.name} and you start with none "
                f"({exc}).")
            return
        # 1つずつ確かめ、使えないものだけ外す（全部を捨てない）。検査は
        # 例外を出さない作りだが、手で編集したファイルでも起動を止めない
        # よう、1件ごとに守る
        skipped = []
        for table, store, check in ((spaces, self.spaces, space_spec_problems),
                                    (reverbs, self.reverbs,
                                     reverb_spec_problems)):
            for name, spec in table.items():
                try:
                    usable = (isinstance(name, str) and isinstance(spec, dict)
                              and not check(spec))
                except Exception:        # noqa: BLE001
                    usable = False
                if usable:
                    store[name] = spec
                else:
                    skipped.append(str(name))
        if skipped:
            self.load_error = (tr("使えない設定があったので読み込みませんでした: ",
                                  "Some settings could not be used and were "
                                  "not loaded: ")
                               + list_sep().join(skipped))

    def save(self) -> None:
        """一時ファイルに書いてから置き換える。失敗は OSError のまま上げる。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "spaces": self.spaces, "reverbs": self.reverbs}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # -- 空間 ---------------------------------------------------------
    def put_space(self, name: str, spec: dict) -> bool:
        """空間を保存する（同じ名前は上書き）。上書きなら True。

        問題があれば ValueError（問題点を日本語で）。保存には失敗しても、
        メモリ上の内容は変えない。
        """
        name = name.strip()
        problems = name_problems(name) + space_spec_problems(spec)
        if problems:
            raise ValueError("、".join(problems))
        replaced = name in self.spaces
        before = dict(self.spaces)
        self.spaces[name] = dict(spec, name=name)
        try:
            self.save()
        except OSError:
            self.spaces = before
            raise
        return replaced

    def delete_space(self, name: str) -> None:
        before = dict(self.spaces)
        self.spaces.pop(name)
        try:
            self.save()
        except OSError:
            self.spaces = before
            raise

    # -- 残響 ---------------------------------------------------------
    def put_reverb(self, name: str, spec: dict) -> bool:
        """残響を保存する（同じ名前は上書き）。上書きなら True。"""
        name = name.strip()
        problems = name_problems(name) + reverb_spec_problems(spec)
        if problems:
            raise ValueError("、".join(problems))
        replaced = name in self.reverbs
        before = dict(self.reverbs)
        self.reverbs[name] = dict(spec, name=name)
        try:
            self.save()
        except OSError:
            self.reverbs = before
            raise
        return replaced

    def delete_reverb(self, name: str) -> None:
        before = dict(self.reverbs)
        self.reverbs.pop(name)
        try:
            self.save()
        except OSError:
            self.reverbs = before
            raise
