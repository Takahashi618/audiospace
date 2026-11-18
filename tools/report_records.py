"""logs の内容と通算成績をまとめて読み上げ用に整形する。

セッションが増えたときの確認用。結果のファイルの一覧、耳介ごとの通算、
順位、残りの未検証を一度に出す。

    python -X utf8 tools\\report_records.py

結果のファイルは、定位確認ツールを動かしたときの表示の言語（日本語か英語）で
書かれている。どちらの言語のファイルも読む。この道具の出力は、いまの表示の
言語（AUDIOSPACE_LANG か OS の表示言語）で出す。
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
from audiospace.i18n import list_sep, tr  # noqa: E402
from audiospace.records import MIN_DIRECTIONS_FOR_RANKING, RecordStore  # noqa: E402

LOG_DIR = ROOT / "logs"
OUT = Path(__file__).with_name("records_report.txt")
_log: list[str] = []

#: 設定の文から拾う項目。(日本語の形, 英語の形)。結果のファイルの「設定:」の
#: 行は demo/localization_app.py の settings_text が作る
SETTING_PATTERNS = {
    "difficulty": (r"難易度は([^、]+)", r"Difficulty: (.+?)\. Front/back"),
    "hrtf": (r"HRTF は([^、]+)", r"HRTF: (.+?)\. Sound:"),
    "wobble": (r"首振りは([^、]+)", r"Head wobble: (.+?)\. Head tilt:"),
}
#: 耳介ごとの成績の行の「(正解/回答)」
PER_PINNA = (r"\((\d+)/(\d+)\)", r"\((\d+) of (\d+)\)")
#: 全体の成績。日本語は「回答、正解」、英語は「正解、回答」の順
TOTAL_JA = r"(\d+)問中(\d+)問正解"
TOTAL_EN = r"(\d+) of (\d+) correct"
#: 前後を試せた数（取り違えの有無にかかわらず出る）
FB_TESTABLE = (r"前後を試せた(\d+)問",
               r"(\d+) questions that tested (?:front and back|it)")
#: 前後の取り違えの数（無ければ 0）
FB_CONFUSED = (r"前後を試せた\d+問のうち、?(?:取り違えが)?(\d+)問",
               r"front and back, (\d+) were confused",
               r"Overall, (\d+) of the \d+ questions")


def mode_logs(mode: str) -> list[Path]:
    """そのモードの結果のファイル（「モード_年月日_時刻.txt」）を名前の順に。

    「calib_*.txt」で探すと、上下用の選定（calib_elev_…）まで拾ってしまう。
    """
    pattern = re.compile(rf"^{re.escape(mode)}_\d{{8}}_\d{{6}}\.txt$")
    return sorted(p for p in LOG_DIR.glob(f"{mode}_*.txt")
                  if pattern.match(p.name))


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def _first(patterns: tuple[str, ...], text: str) -> re.Match | None:
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m
    return None


def brief_from_text(stem: str, text: str) -> str:
    """1 セッション分の1行要約（結果のファイルの中身から）。"""
    found = {}
    for key, patterns in SETTING_PATTERNS.items():
        m = _first(patterns, text)
        found[key] = m.group(1) if m else "?"
    # 耳介ごとの成績の行があれば、それを足し合わせる
    totals = [pair for pattern in PER_PINNA
              for pair in re.findall(pattern, text)]
    correct = answered = None
    if totals:
        correct = sum(int(a) for a, _ in totals)
        answered = sum(int(b) for _, b in totals)
    else:
        m = re.search(TOTAL_JA, text)
        if m:
            answered, correct = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(TOTAL_EN, text)
            if m:
                correct, answered = int(m.group(1)), int(m.group(2))
    if answered:
        score = tr(f"{correct}/{answered}問 ({correct / answered * 100:.0f}%)",
                   f"{correct} of {answered} ({correct / answered * 100:.0f}%)")
    else:
        score = tr("集計不明", "no totals")
    m = _first(FB_TESTABLE, text)
    if m:
        conf = _first(FB_CONFUSED, text)
        n = conf.group(1) if conf else "0"
        fb = tr(f"前後 {n}/{m.group(1)}問",
                f"front/back {n} of {m.group(1)}")
    else:
        fb = tr("前後不明", "front/back unknown")
    return (f"{stem[-6:]}  {found['difficulty']:<6} {score:<16} {fb:<14} "
            f"{found['hrtf']} / {found['wobble']}")


def session_brief(path: Path) -> str:
    return brief_from_text(path.stem, path.read_text(encoding="utf-8"))


def main() -> int:
    pool = [(str(p), label) for label, p in list_hrtf_choices() if p]
    store = RecordStore(LOG_DIR / "pinna_records.json")

    log(tr("logs の内容", "Contents of logs"))
    log("=" * 78)
    for kind in ("quiz", "training", "calib", "calib_elev"):
        files = mode_logs(kind)
        if not files:
            continue
        log("")
        log(tr(f"--- {kind} ({len(files)}件) ---",
               f"--- {kind} ({len(files)}) ---"))
        for path in files:
            try:
                log("  " + session_brief(path))
            except Exception as exc:  # noqa: BLE001
                log(tr(f"  {path.name}: 読めません ({exc!r})",
                       f"  {path.name}: cannot be read ({exc!r})"))

    imported = LOG_DIR / "imported_calib.txt"
    calib_files = mode_logs("calib")
    if calib_files:
        names = set()
        if imported.exists():
            names = {x.strip()
                     for x in imported.read_text(encoding="utf-8").splitlines()
                     if x.strip()}
        pending = [p.name for p in calib_files if p.name not in names]
        log("")
        log(tr(f"通算に反映済みの印: {len(names)}件 / "
               f"耳介の選定の結果 {len(calib_files)}件",
               f"Marked as counted in the totals: {len(names)} / "
               f"pinna results: {len(calib_files)}"))
        if pending:
            # ツールは終了時に自分で通算に反映して印も付ける。印の無い
            # 結果は、印を付けるようになる前に作られたものだけのはず
            log(tr(f"[警告] 印の無い結果が {len(pending)}件 あります。",
                   f"[Warning] {len(pending)} result(s) have no mark."))
            log(tr("  ツールが記録した結果なら、通算には反映済みです。",
                   "  If the tool recorded them, they are already in the "
                   "totals."))
            for name in pending:
                log(f"    {name}")

    log("")
    log("=" * 78)
    log(tr("耳介の通算成績", "Pinna totals"))
    log("=" * 78)
    log("")
    log(store.summary(pool))
    log("")
    for i, rec in enumerate(store.ranking(), start=1):
        log(f"{i:>2}. {rec.describe()}")
    pending_rec = [r for r in store.records.values() if not r.ranked]
    if pending_rec:
        log("")
        log(tr(f"順位保留（{MIN_DIRECTIONS_FOR_RANKING}方向以上での試行が不足）:",
               f"Not ranked yet (too few questions with "
               f"{MIN_DIRECTIONS_FOR_RANKING} directions or more):"))
        for rec in pending_rec:
            log(tr(f"    {rec.label}: 通算{rec.n_answered}問、"
                   f"うち判別可能な難易度は{rec.n_answered_hard}問",
                   f"    {rec.label}: {rec.n_answered} questions in all, "
                   f"{rec.n_answered_hard} at difficulties that can tell "
                   "them apart"))

    done = store.tested_tags()
    # 記録のタグはファイル名（subj_E.sofa）、候補はパス。ファイル名で比べる
    untested = [label for tag, label in pool if Path(tag).name not in done]
    shown = list_sep().join(untested[:10])
    more = tr("ほか", " and more") if len(untested) > 10 else ""
    log("")
    log(tr(f"未検証 {len(untested)}種類: {shown}{more}",
           f"Not tested yet ({len(untested)}): {shown}{more}"))

    OUT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
