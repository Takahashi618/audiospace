"""全テストスイートを順に実行して合計を報告する。

AUDIOSPACE_SILENT=1 は内部で設定するので、環境変数の指定を覚えて
おかなくてよい。音は出ない。表示の言語は日本語に固定する
（AUDIOSPACE_LANG=ja。英語の表示は GUI構成のテストが別のプロセスで調べる）。

    python -X utf8 tests\\run_all.py
    python -X utf8 tests\\run_all.py --skip-gui   GUI構成のテストを省く

`--skip-gui` は wxPython の無い環境（自動テストの macOS と Linux）のため。
省いたことは結果の行に書く。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SUITES = [
    ("run_tests.py", "ドメイン"),
    ("run_motion_tests.py", "首振り"),
    ("run_quiz_tests.py", "クイズ"),
    ("run_records_tests.py", "記録"),
    ("run_app_tests.py", "GUI構成"),
]


GUI_SUITE = "run_app_tests.py"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    unknown = [a for a in argv if a != "--skip-gui"]
    if unknown:
        print(f"使い方: run_all.py [--skip-gui]（知らない指定: {unknown}）"
              f" (usage: run_all.py [--skip-gui])")
        return 2
    skip_gui = "--skip-gui" in argv
    env = dict(os.environ, AUDIOSPACE_SILENT="1", AUDIOSPACE_LANG="ja")
    total_passed = 0
    total_all = 0
    failed_suites: list[str] = []
    skipped: list[str] = []

    for name, label in SUITES:
        if skip_gui and name == GUI_SUITE:
            print(f"{label} ({name}): 省いた (skipped, --skip-gui)")
            skipped.append(name)
            continue
        path = Path(__file__).with_name(name)
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(path)],
            env=env, capture_output=True, text=True, encoding="utf-8")
        line = ""
        if result.stdout:
            lines = [x for x in result.stdout.strip().splitlines() if x]
            line = lines[-1] if lines else ""
        print(f"{label} ({name}): {line}")
        m = re.search(r"passed (\d+)/(\d+)", line)
        if m:
            total_passed += int(m.group(1))
            total_all += int(m.group(2))
        if result.returncode != 0 or not m:
            failed_suites.append(name)
            if result.stderr:
                print(result.stderr.strip()[-2000:])

    if failed_suites:
        print(f"合計 {total_passed}/{total_all}、"
              f"失敗のあったスイート: {', '.join(failed_suites)}"
              f" (total {total_passed}/{total_all}, FAILED)")
        return 1
    # 英語の利用者のために英語も添える（マニュアルの英語版がこの行を説明する）
    note = (f"（{', '.join(skipped)} は省いた）" if skipped else "")
    note_en = (f", skipped {', '.join(skipped)}" if skipped else "")
    print(f"合計 {total_passed}/{total_all}、全件合格{note}"
          f" (total {total_passed}/{total_all}, all passed{note_en})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
