"""logs の内容と通算成績をまとめて読み上げ用に整形する。

セッションが増えたときの確認用。ログの一覧、耳介ごとの通算、
順位、残りの未検証を一度に出す。

    python -X utf8 tools\\report_records.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ["AUDIOSPACE_SILENT"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.localization_app import list_hrtf_choices  # noqa: E402
from demo.records import MIN_DIRECTIONS_FOR_RANKING, RecordStore  # noqa: E402

LOG_DIR = ROOT / "logs"
OUT = Path(__file__).with_name("records_report.txt")
_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def session_brief(path: Path) -> str:
    """1 セッション分の1行要約。"""
    text = path.read_text(encoding="utf-8")
    diff = "?"
    m = re.search(r"難易度は([^、]+)", text)
    if m:
        diff = m.group(1)
    hrtf = "?"
    m = re.search(r"HRTF は([^、]+)", text)
    if m:
        hrtf = m.group(1)
    wob = "?"
    m = re.search(r"首振りは([^、]+)", text)
    if m:
        wob = m.group(1)
    # 詳細行から総問数を数える
    totals = re.findall(r"\((\d+)/(\d+)\)", text)
    if totals:
        correct = sum(int(a) for a, _ in totals)
        answered = sum(int(b) for _, b in totals)
        score = f"{correct}/{answered}問 ({correct / answered * 100:.0f}%)"
    else:
        m = re.search(r"(\d+)問中(\d+)問正解", text)
        score = f"{m.group(2)}/{m.group(1)}問" if m else "集計不明"
    fb = "前後不明"
    m = re.search(r"前後を試せた(\d+)問", text)
    if m:
        conf = re.search(r"のうち(\d+)問で取り違え", text)
        fb = (f"前後 {conf.group(1)}/{m.group(1)}問" if conf
              else f"前後 0/{m.group(1)}問")
    return (f"{path.stem[-6:]}  {diff:<6} {score:<16} {fb:<14} "
            f"{hrtf}／{wob}")


def main() -> int:
    pool = [(str(p), label) for label, p in list_hrtf_choices() if p]
    store = RecordStore()

    log("logs の内容")
    log("=" * 78)
    for kind in ("quiz", "training", "calib"):
        files = sorted(LOG_DIR.glob(f"{kind}_*.txt"))
        if not files:
            continue
        log("")
        log(f"--- {kind} ({len(files)}件) ---")
        for path in files:
            try:
                log("  " + session_brief(path))
            except Exception as exc:  # noqa: BLE001
                log(f"  {path.name}: 読めません ({exc!r})")

    imported = LOG_DIR / "imported_calib.txt"
    calib_files = sorted(LOG_DIR.glob("calib_*.txt"))
    if calib_files:
        names = set()
        if imported.exists():
            names = {x.strip()
                     for x in imported.read_text(encoding="utf-8").splitlines()
                     if x.strip()}
        pending = [p.name for p in calib_files if p.name not in names]
        log("")
        log(f"通算に反映済み: {len(names)}件 / calib ログ {len(calib_files)}件")
        if pending:
            # アプリは終了時に自分で反映して印も付ける。印のないログは
            # 印付け機構より前に作られたものだけのはず。
            log(f"[警告] 印のない calib ログが {len(pending)}件 あります。")
            log("  アプリが記録した分なら既に反映済みなので、"
                "取り込むと二重集計になります。")
            log("  反映済みのログは取り込まないでください。")
            for name in pending:
                log(f"    {name}")

    log("")
    log("=" * 78)
    log("耳介の通算成績")
    log("=" * 78)
    log("")
    log(store.summary(pool))
    log("")
    for i, rec in enumerate(store.ranking(), start=1):
        log(f"{i:>2}. {rec.describe()}")
    pending_rec = [r for r in store.records.values() if not r.ranked]
    if pending_rec:
        log("")
        log(f"順位保留（{MIN_DIRECTIONS_FOR_RANKING}方向以上での試行が不足）:")
        for rec in pending_rec:
            log(f"    {rec.label}: 通算{rec.n_answered}問、"
                f"うち判別可能な難易度は{rec.n_answered_hard}問")

    done = store.tested_tags()
    untested = [label for tag, label in pool if tag not in done]
    log("")
    log(f"未検証 {len(untested)}種類: "
        + ("、".join(x.replace("実測 Viking ", "") for x in untested[:10])
           + ("ほか" if len(untested) > 10 else "")))

    OUT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
