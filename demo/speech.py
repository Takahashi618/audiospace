"""speech.py -- スクリーンリーダー読み上げのラッパー

accessible_output2 経由で PC-Talker / NVDA / SAPI に出力する。

方針 (全プロジェクト共通):
- 接頭語は [エラー] と [警告] のみ。情報・成功は本文をそのまま読む。
- 1 つの動作から複数の文が出る場合は必ず 1 回の呼び出しにまとめる。
  逐次呼び出しはスクリーンリーダーが前の発話を打ち切るため、
  結果として最後の 1 文しか聞こえなくなる。
- 環境変数 AUDIOSPACE_SILENT=1 で全出力を抑止する (自動テスト用)。
"""
from __future__ import annotations

import os

from audiospace.i18n import tr


class Speech:
    """読み上げ出力。初期化に失敗しても例外を投げず、無音で動作を続ける。"""

    def __init__(self, silent: bool | None = None) -> None:
        if silent is None:
            silent = os.environ.get("AUDIOSPACE_SILENT", "").strip() \
                not in ("", "0", "false")
        self.silent = silent
        self.available = False
        self.backend_name = tr("なし", "none")
        self.log: list[str] = []  # テスト時の検証用に発話を保持する
        self._speaker = None
        if silent:
            return
        try:
            import accessible_output2.outputs.auto

            self._speaker = accessible_output2.outputs.auto.Auto()
            self.available = True
            self.backend_name = type(self._speaker).__name__
        except Exception as exc:  # 読み上げ環境がなくても動かす
            self.log.append(tr(f"[警告] 読み上げを初期化できません: {exc}",
                               f"[Warning] Cannot start speech: {exc}"))

    def say(self, *parts: str, interrupt: bool = True) -> None:
        """複数の文をまとめて 1 回で読み上げる。空文字は無視する。"""
        text = " ".join(p.strip() for p in parts if p and p.strip())
        if not text:
            return
        self.log.append(text)
        if self.silent or not self._speaker:
            return
        try:
            self._speaker.speak(text, interrupt=interrupt)
        except Exception:
            pass

    def error(self, *parts: str) -> None:
        self.say(tr("[エラー]", "[Error]"), *parts)

    def warn(self, *parts: str) -> None:
        self.say(tr("[警告]", "[Warning]"), *parts)

    def stop(self) -> None:
        if self._speaker is not None:
            try:
                self._speaker.silence()
            except Exception:
                pass
