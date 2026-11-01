"""表示の言語（日本語か英語）。

利用者に見せる文（読み上げ名、状態の説明、誤りの説明）は、日本語と英語を
`tr("日本語", "English")` のように並べて書き、いまの言語のほうを使う。

言語は最初に使われたときに1回だけ決める:

1. 環境変数 AUDIOSPACE_LANG が ja か en なら、それ
2. 無ければ OS の表示言語。日本語なら ja、それ以外は en

Windows は表示言語（GetUserDefaultUILanguage）、macOS は「優先する言語」の
先頭（AppleLanguages）、それ以外は LC_ALL・LC_MESSAGES・LANG を見る。

一覧や名前の表（FURNISH_LEVELS など）は import のときに作るので、言語を
変えたいときは、プログラムを起動する前に AUDIOSPACE_LANG を設定する。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

#: 使える言語
LANGUAGES = ("ja", "en")
#: 言語を決める環境変数（ja か en）
ENV_VAR = "AUDIOSPACE_LANG"

_current: str | None = None


def _normalize(value: str | None) -> str | None:
    """「ja_JP.UTF-8」「en-US」などを ja / en にする。どちらでもなければ None。"""
    if not value:
        return None
    head = value.strip().lower()
    if head.startswith("ja"):
        return "ja"
    if head.startswith("en"):
        return "en"
    return None


def _windows_language() -> str | None:
    try:
        import ctypes
        lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
    except Exception:  # noqa: BLE001
        return None
    # 下位10ビットが主言語。0x11 が日本語
    return "ja" if (int(lang_id) & 0x3FF) == 0x11 else "en"


def _mac_language() -> str | None:
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return None
    first = re.search(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]+)*", out)
    if first is None:
        return None
    return "ja" if first.group(0).lower().startswith("ja") else "en"


def _posix_language() -> str | None:
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name)
        if value:
            return "ja" if _normalize(value) == "ja" else "en"
    return None


def system_language() -> str:
    """OS の表示言語（ja か en）。日本語でなければ en。"""
    if sys.platform == "win32":
        found = _windows_language()
    elif sys.platform == "darwin":
        found = _mac_language() or _posix_language()
    else:
        found = _posix_language()
    return found or "en"


def language() -> str:
    """いまの表示の言語（ja か en）。"""
    global _current
    if _current is None:
        _current = _normalize(os.environ.get(ENV_VAR)) or system_language()
    return _current


def is_japanese() -> bool:
    return language() == "ja"


def tr(ja: str, en: str) -> str:
    """いまの言語のほうの文を返す。"""
    return ja if language() == "ja" else en


def join_sentences(*parts: str) -> str:
    """文をつなぐ。日本語は詰め、英語は空白を1つ入れる。空の文は飛ばす。"""
    items = [p.strip() if language() == "en" else p for p in parts if p]
    return ("" if language() == "ja" else " ").join(i for i in items if i)


def list_sep() -> str:
    """並べるときの区切り（日本語は「、」、英語は「, 」）。"""
    return tr("、", ", ")
