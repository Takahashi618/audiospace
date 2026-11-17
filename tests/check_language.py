"""OS の表示言語の判定（audiospace.i18n.system_language）を確かめる。

    python -X utf8 tests/check_language.py               判定した言語を表示する
    python -X utf8 tests/check_language.py --expect ja   ja でなければ失敗（終了コード 1）

自動テスト（.github/workflows/tests.yml）が、Windows・macOS・Linux それぞれで
使う。環境変数 AUDIOSPACE_LANG は見ない（OS の判定だけを確かめるため）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect", choices=("ja", "en"))
    args = parser.parse_args(argv)
    os.environ.pop("AUDIOSPACE_LANG", None)
    from audiospace import i18n
    found = i18n.system_language()
    print(f"platform={sys.platform} system_language={found}")
    if args.expect and found != args.expect:
        print(f"[FAIL] expected {args.expect}, got {found}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
