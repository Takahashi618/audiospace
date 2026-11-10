"""耳介の通算戦績（demo/records.py）のテスト。

要点は「聞き手の上達による交絡を、基準との差で打ち消せているか」。
ここが効いていないと、後から試した耳介ほど有利になってしまう。

Run: set AUDIOSPACE_SILENT=1 && python -X utf8 tests\\run_records_tests.py
Out: tests/records_test_result.txt (UTF-8)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ["AUDIOSPACE_SILENT"] = "1"
# 表示の言語を日本語に固定する（テストが期待する文は日本語）
os.environ["AUDIOSPACE_LANG"] = "ja"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.records import (MIN_TRIALS_FOR_RANKING, PinnaRecord,  # noqa: E402
                          RecordStore)

OUT = Path(__file__).with_name("records_test_result.txt")
_log: list[str] = []
_passed = 0
_failed = 0


def log(msg: str = "") -> None:
    _log.append(msg)


def test(name: str):
    def deco(fn):
        def run() -> None:
            global _passed, _failed
            try:
                detail = fn()
                _passed += 1
                log(f"[PASS] {name}")
                for line in str(detail or "").splitlines():
                    if line.strip():
                        log(f"         {line}")
            except AssertionError as exc:
                _failed += 1
                log(f"[FAIL] {name}")
                log(f"         {exc}")
            except Exception as exc:  # noqa: BLE001
                _failed += 1
                log(f"[ERROR] {name}: {exc!r}")
                for line in traceback.format_exc().splitlines()[-4:]:
                    log(f"         {line}")
        return run
    return deco


@dataclass
class FakeScore:
    """CalibrationSession.scores() の戻り値を模したもの。"""

    tag: str
    label: str
    n_answered: int
    n_correct: int
    n_front_back: int = 0
    n_fb_testable: int = 0
    total_error_deg: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_answered if self.n_answered else 0.0


def fresh_store() -> RecordStore:
    tmp = Path(tempfile.mkdtemp()) / "records.json"
    return RecordStore(tmp)


def score(tag: str, correct: int, total: int = 16) -> FakeScore:
    return FakeScore(tag=tag, label=f"耳介{tag}", n_answered=total,
                     n_correct=correct, n_fb_testable=total * 3 // 4,
                     total_error_deg=(total - correct) * 45.0)


POOL = [(t, f"耳介{t}") for t in "ABCDE"]


@test("【核心】上達しても基準との差で正しく比較できる")
def t_skill_drift() -> str:
    """後のセッションほど全体の成績が上がる状況を作る。

    生の正答率だけで順位を付けると、後から試した耳介が不当に有利になる。
    基準との差を使えば、その影響が消えるはずである。
    """
    store = fresh_store()
    # 第1回: 基準A=50%、挑戦者B=75%（Bのほうが25ポイント良い）
    store.update([score("A", 8), score("B", 12)], anchor_tag="A", n_directions=8)
    # 第2回: 上達して全体が底上げされ、基準A=75%、挑戦者C=81%
    #        Cの生の正答率(81%)はBの生の正答率(75%)より高いが、
    #        基準との差は C:+6 < B:+25 なので B が上のままであるべき
    store.update([score("A", 12), score("C", 13)], anchor_tag="A", n_directions=8)

    b = store.records["B"]
    c = store.records["C"]
    assert c.accuracy > b.accuracy, "前提が崩れています（Cの生の正答率が上）"
    ranked = [r.tag for r in store.ranking()]
    assert ranked[0] == "B", (
        f"生の正答率に引きずられています: 順位={ranked}, "
        f"B={b.accuracy:.2f}(差{b.mean_delta:+.2f}) "
        f"C={c.accuracy:.2f}(差{c.mean_delta:+.2f})")
    return (f"生の正答率は B={b.accuracy * 100:.0f}% < C={c.accuracy * 100:.0f}%\n"
            f"基準との差は B={b.mean_delta * 100:+.0f} > C={c.mean_delta * 100:+.0f}\n"
            f"順位={ranked} → 上達の影響を打ち消せている")


@test("基準自身の差は0になる")
def t_anchor_zero() -> str:
    store = fresh_store()
    store.update([score("A", 8), score("B", 12)], anchor_tag="A", n_directions=8)
    store.update([score("A", 14), score("C", 10)], anchor_tag="A", n_directions=8)
    a = store.records["A"]
    assert a.mean_delta == 0.0, a.mean_delta
    assert len(a.relative_deltas) == 2, a.relative_deltas
    return f"基準Aの差={a.mean_delta:+.2f}（{len(a.relative_deltas)}回分）"


@test("通算が積み上がる")
def t_accumulate() -> str:
    store = fresh_store()
    store.update([score("A", 8), score("B", 12)], anchor_tag="A", n_directions=8)
    store.update([score("A", 10), score("B", 14)], anchor_tag="A", n_directions=8)
    b = store.records["B"]
    assert b.n_answered == 32 and b.n_correct == 26, (b.n_answered, b.n_correct)
    assert b.n_sessions == 2
    return (f"B: {b.n_correct}/{b.n_answered}問 "
            f"({b.accuracy * 100:.0f}パーセント)、{b.n_sessions}セッション")


@test("試行数が足りない耳介は順位に出さない")
def t_min_trials() -> str:
    store = fresh_store()
    store.update([score("A", 8), score("B", 4, total=4)], anchor_tag="A", n_directions=8)
    ranked = [r.tag for r in store.ranking()]
    assert "B" not in ranked, f"4問しかないBが順位に入っています: {ranked}"
    assert "A" in ranked
    lines = store.detail_lines()
    assert any("試行不足" in x for x in lines), lines
    return (f"最低{MIN_TRIALS_FOR_RANKING}問。順位={ranked}、"
            f"不足分は別途表示: {[x for x in lines if '試行不足' in x]}")


@test("次の顔ぶれに必ずチャンピオンが入る")
def t_lineup_anchor() -> str:
    store = fresh_store()
    store.update([score("A", 8), score("B", 14)], anchor_tag="A", n_directions=8)
    lineup, anchor = store.build_lineup(POOL, slots=3)
    tags = [t for t, _ in lineup]
    assert anchor == "B", f"チャンピオンはBのはず: {anchor}"
    assert tags[0] == "B", tags
    assert "C" in tags and "D" in tags, f"新顔が入っていません: {tags}"
    assert "A" not in tags, "検証済みのAが再登板しています"
    return f"顔ぶれ={tags}、基準={anchor}"


@test("初回はチャンピオンがいないので新顔だけで組む")
def t_lineup_first() -> str:
    store = fresh_store()
    lineup, anchor = store.build_lineup(POOL, slots=3)
    assert anchor is None, anchor
    assert [t for t, _ in lineup] == ["A", "B", "C"], lineup
    return f"顔ぶれ={[t for t, _ in lineup]}、基準なし"


@test("新顔が尽きたら下位を再検証に回す")
def t_lineup_exhausted() -> str:
    store = fresh_store()
    for tag, correct in (("A", 8), ("B", 14), ("C", 6), ("D", 10), ("E", 9)):
        store.update([score(tag, correct)], anchor_tag=tag, n_directions=8)
    lineup, anchor = store.build_lineup(POOL, slots=3)
    tags = [t for t, _ in lineup]
    assert len(tags) == 3, tags
    assert len(set(tags)) == 3, f"重複しています: {tags}"
    return f"全種検証済みの状態で顔ぶれ={tags}"


@test("2枠未満は拒否する")
def t_lineup_too_small() -> str:
    store = fresh_store()
    try:
        store.build_lineup(POOL, slots=1)
    except ValueError as exc:
        return f"想定どおり例外: {exc}"
    raise AssertionError("1枠が通ってしまいました")


@test("保存して読み直せる")
def t_roundtrip() -> str:
    store = fresh_store()
    store.update([score("A", 8), score("B", 14)], anchor_tag="A", n_directions=8)
    store.save()
    again = RecordStore(store.path)
    assert set(again.records) == {"A", "B"}, again.records.keys()
    b = again.records["B"]
    assert b.n_correct == 14 and b.relative_deltas
    assert again.champion().tag == "B"
    return (f"{store.path.name} に保存し読み直し、"
            f"チャンピオン={again.champion().label}")


@test("壊れたファイルでも起動を止めない")
def t_corrupt() -> str:
    tmp = Path(tempfile.mkdtemp()) / "records.json"
    tmp.write_text("{ これはJSONではない", encoding="utf-8")
    store = RecordStore(tmp)
    assert store.records == {}, store.records
    store.update([score("A", 8)], anchor_tag="A", n_directions=8)
    store.save()
    assert RecordStore(tmp).records, "保存し直せていません"
    return "壊れたJSONを読み飛ばし、空の状態で起動して復旧できた"


@test("僅差なら偶然の可能性に言及する")
def t_close_call() -> str:
    store = fresh_store()
    # BはAより1問だけ良い（16問中1問 = 6ポイント）
    store.update([score("A", 12), score("B", 13)], anchor_tag="A", n_directions=8)
    text = store.summary(POOL)
    assert "偶然" in text, f"僅差の注意がありません: {text}"

    store2 = fresh_store()
    store2.update([score("A", 6), score("B", 15)], anchor_tag="A", n_directions=8)
    text2 = store2.summary(POOL)
    assert "偶然" not in text2, f"大差なのに偶然扱いです: {text2}"
    return f"僅差: {text}\n\n大差: {text2}"


@test("進捗を伝える")
def t_progress() -> str:
    store = fresh_store()
    assert "残り5種類" in store.progress_text(POOL)
    store.update([score("A", 8), score("B", 12)], anchor_tag="A", n_directions=8)
    mid = store.progress_text(POOL)
    assert "2種類が検証済み" in mid and "残り3種類" in mid, mid
    for tag in "CDE":
        store.update([score(tag, 8)], anchor_tag=tag, n_directions=8)
    assert "全5種類の検証が終わりました" in store.progress_text(POOL)
    return mid


@test("前後の取り違え率は試せた問題を分母にする")
def t_fb_rate() -> str:
    store = fresh_store()
    s = FakeScore("A", "耳介A", n_answered=16, n_correct=10,
                  n_front_back=3, n_fb_testable=12, total_error_deg=270.0)
    store.update([s], anchor_tag="A", n_directions=8)
    rec = store.records["A"]
    assert abs(rec.fb_rate - 0.25) < 1e-9, rec.fb_rate
    assert "前後3/12問" in rec.describe(), rec.describe()
    return rec.describe()


@test("【重要】易しい難易度の満点は順位に使わない")
def t_easy_excluded() -> str:
    """4方向は全員が満点に張り付いて差が出ない。実際、4方向の回では
    5種類中4種類が100パーセントで並んだ。その回の満点を根拠に
    チャンピオンを決めると、8方向で本当に良かった耳介が負ける。"""
    store = fresh_store()
    # 4方向の易しい回: E も B も満点
    store.update([score("B", 8, total=8), score("E", 8, total=8)],
                 anchor_tag="B", n_directions=4)
    # 8方向の回: B が good、A が悪い。E は参加していない
    store.update([score("B", 14), score("A", 11)],
                 anchor_tag="B", n_directions=8)

    e = store.records["E"]
    assert not e.ranked, (
        f"4方向でしか試していないEが順位に入っています "
        f"(hard={e.n_answered_hard})")
    ranked = [r.tag for r in store.ranking()]
    assert ranked and ranked[0] == "B", f"順位={ranked}"
    assert "E" not in store.tested_tags(), \
        "Eが検証済み扱いになり、再出題されなくなります"
    return (f"Eは通算{e.n_answered}問(正答率{e.accuracy * 100:.0f}%)だが"
            f"判別可能な難易度は{e.n_answered_hard}問なので順位外\n"
            f"順位={ranked}、Eは未検証として再出題の対象")


@test("易しい回の差は平均に混ざらない")
def t_easy_delta_excluded() -> str:
    store = fresh_store()
    # 4方向で C が基準より大きく上（易しいので信用できない）
    store.update([score("B", 4, total=8), score("C", 8, total=8)],
                 anchor_tag="B", n_directions=4)
    # 8方向で C は基準より下
    store.update([score("B", 14), score("C", 10)],
                 anchor_tag="B", n_directions=8)
    c = store.records["C"]
    assert len(c.relative_deltas) == 2, c.relative_deltas
    assert len(c.hard_deltas) == 1, c.hard_deltas
    assert c.mean_delta < 0, (
        f"易しい回の+50%が混ざっています: 全差={c.relative_deltas} "
        f"採用={c.hard_deltas} 平均={c.mean_delta}")
    return (f"記録された差={[f'{d:+.2f}' for d in c.relative_deltas]}"
            f"（方向数{c.delta_directions}）\n"
            f"順位に使う差={[f'{d:+.2f}' for d in c.hard_deltas]}、"
            f"平均={c.mean_delta:+.2f}")


@test("旧形式（絶対パスのタグ）を読み込み時にファイル名へ移行する")
def t_path_migration() -> str:
    """以前はタグが SOFA の絶対パスだったため、フォルダ移動や別の PC との行き来で
    通算成績が全部「別の耳介」になった。読み込み時にファイル名へ正規化し、
    旧新両形式が並んでいたら1つに畳む。"""
    import json

    tmp = Path(tempfile.mkdtemp()) / "records.json"
    old_tag = r"C:\old\place\assets\hrtf\viking\subj_B.sofa"
    new_tag = "subj_B.sofa"
    base = dict(label="耳介B", n_fb_testable=0, n_fb_confusions=0,
                total_error_deg=0.0, n_sessions=1,
                relative_deltas=[0.1], delta_directions=[8],
                n_answered_hard=16, n_correct_hard=12, last_seen="2026-08-10")
    tmp.write_text(json.dumps({"records": [
        dict(base, tag=old_tag, n_answered=16, n_correct=12),
        dict(base, tag=new_tag, n_answered=16, n_correct=14),
    ]}, ensure_ascii=False), encoding="utf-8")

    store = RecordStore(tmp)
    assert set(store.records) == {"subj_B.sofa"}, store.records.keys()
    rec = store.records["subj_B.sofa"]
    assert rec.n_answered == 32 and rec.n_correct == 26, \
        (rec.n_answered, rec.n_correct)
    assert len(rec.relative_deltas) == 2
    store.save()
    again = RecordStore(tmp)
    assert set(again.records) == {"subj_B.sofa"}
    return (f"旧パスと新名が {rec.tag} に統合され "
            f"{rec.n_correct}/{rec.n_answered}問。保存後も名前のまま")


@test("候補にパスを渡しても、記録はファイル名で照合される")
def t_path_candidates() -> str:
    """アプリは (SOFAのフルパス, 表示名) を渡してくる。記録側は名前で
    積み上げつつ、build_lineup はアプリが読み込みに使える形（パス）の
    まま返す必要がある。"""
    store = fresh_store()
    pool = [(rf"C:\somewhere\viking\subj_{t}.sofa", f"耳介{t}")
            for t in "ABC"]
    store.update([score(pool[0][0], 8), score(pool[1][0], 14)],
                 anchor_tag=pool[0][0], n_directions=8)
    assert set(store.records) == {"subj_A.sofa", "subj_B.sofa"}, \
        store.records.keys()

    lineup, anchor = store.build_lineup(pool, slots=2)
    assert anchor == pool[1][0], f"基準はBのフルパスのはず: {anchor}"
    assert lineup[0] == pool[1], lineup
    assert lineup[1] == pool[2], f"新顔はCのはず: {lineup}"

    moved = [(rf"D:\moved\subj_{t}.sofa", f"耳介{t}") for t in "ABC"]
    text = store.progress_text(moved)
    assert "2種類が検証済み" in text, f"移動後に通算が切れています: {text}"
    return (f"記録キー={sorted(store.records)}、"
            f"基準はパスのまま返る、移動後の進捗: {text}")


@test("自作の空間と残響: 保存・読み込み・上書き・削除")
def t_user_presets() -> str:
    from demo.user_presets import UserPresetStore
    room = {"kind": "room", "size": [4.0, 3.0, 2.5],
            "materials": {f: "rough_concrete" for f in
                          ("west", "east", "south", "north", "floor",
                           "ceiling")},
            "at": [0.4, 0.4]}
    reverb = {"rt60": 2.5, "room_scale": 1.2, "damping": 0.3,
              "width": 1.0, "level": 1.0}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "user_presets.json"
        store = UserPresetStore(path)
        assert store.spaces == {} and store.load_error == ""
        assert store.put_space("  灰色の部屋 ", room) is False
        assert store.put_space("灰色の部屋", dict(room, late_gain=2.0)) is True
        assert store.put_reverb("長い残響", reverb) is False
        again = UserPresetStore(path)
        assert list(again.spaces) == ["灰色の部屋"]
        assert again.spaces["灰色の部屋"]["late_gain"] == 2.0
        assert again.spaces["灰色の部屋"]["name"] == "灰色の部屋"
        assert again.reverbs["長い残響"]["rt60"] == 2.5
        # 名前と値の検査
        for name, spec in (("", room), ("x" * 41, room),
                           ("壊れ", dict(room, kind="cave"))):
            try:
                again.put_space(name, spec)
            except ValueError:
                pass
            else:
                raise AssertionError(f"受け付けました: {name!r}")
        again.delete_space("灰色の部屋")
        assert UserPresetStore(path).spaces == {}
        assert not path.with_suffix(".json.tmp").exists(), "一時ファイルが残った"
    return "前後の空白を除いて保存、上書きを報告、読み直して同じ、削除も残る"


@test("自作の空間と残響: 壊れたファイルは .broken に移して空から始める")
def t_user_presets_broken() -> str:
    from demo.user_presets import UserPresetStore
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "user_presets.json"
        path.write_text("{ これは JSON ではない", encoding="utf-8")
        store = UserPresetStore(path)
        assert store.spaces == {} and "読めない" in store.load_error
        assert path.with_suffix(".json.broken").exists(), "元のファイルを消した"
        # 一部だけ使えないときは、その分だけ外す
        path.write_text(json.dumps({"spaces": {
            "良い": {"kind": "open", "materials": {"floor": "carpet_thin"}},
            "悪い": {"kind": "room"}}}), encoding="utf-8")
        store = UserPresetStore(path)
        assert list(store.spaces) == ["良い"], store.spaces
        assert "悪い" in store.load_error
        # 書けないときはメモリ上の内容も変えない
        store.path = Path(tmp)                 # フォルダには書けない
        try:
            store.put_space("新しい", {"kind": "open",
                                      "materials": {"floor": "carpet_thin"}})
        except OSError:
            pass
        else:
            raise AssertionError("書けないのに成功しました")
        assert list(store.spaces) == ["良い"], store.spaces
    return "壊れたファイルは残して空から、使えない項目だけ外す、書けなければ元のまま"


@test("review: 手で壊した値の入った保存ファイルでも、起動を止めずに外す")
def t_review_presets_malformed() -> str:
    from demo.user_presets import UserPresetStore
    good = {"kind": "open", "materials": {"floor": "carpet_thin"}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "user_presets.json"
        path.write_text(json.dumps({
            "spaces": {
                "良い": good,
                "大きさが文字": {"kind": "room", "size": "abc",
                                 "materials": {}},
                "材質が一覧": {"kind": "room", "size": [4, 3, 2.5],
                               "materials": ["wood_16mm"]},
                "数が文字": dict(good, contents={"person": "many"}),
                "回数が無効": dict(good, early_order=None),
            },
            "reverbs": {"残響が文字": {"rt60": "long"}},
        }), encoding="utf-8")
        store = UserPresetStore(path)               # ここで例外を出さない
    assert list(store.spaces) == ["良い"], store.spaces
    assert store.reverbs == {}
    for name in ("大きさが文字", "材質が一覧", "数が文字", "回数が無効",
                 "残響が文字"):
        assert name in store.load_error, (name, store.load_error)
    return store.load_error


@test("通算成績のファイルが壊れていたら .broken に残して知らせ、形が違っても起動する")
def t_review2_records_broken() -> str:
    for text in ("[]", '{"records": [', '{"records": 5}'):
        folder = Path(tempfile.mkdtemp())
        path = folder / "records.json"
        path.write_text(text, encoding="utf-8")
        store = RecordStore(path)                 # 例外を出さない
        assert store.records == {} and store.load_error, text
        assert not path.exists(), "元のファイルが残っていて、上書きされうる"
        assert (folder / "records.json.broken").read_text(
            encoding="utf-8") == text, "退避したファイルが元と違う"
    # 保存は一時ファイル経由（途中の .tmp が残らない）
    store.update([score("A", 8)], anchor_tag="A", n_directions=8)
    store.save()
    assert not list(folder.glob("*.tmp")) and RecordStore(path).records
    return "3通りの壊れ方で、退避して知らせ、保存し直せた"


TESTS = [t_skill_drift, t_easy_excluded, t_easy_delta_excluded,
         t_anchor_zero, t_accumulate, t_min_trials,
         t_lineup_anchor, t_lineup_first, t_lineup_exhausted,
         t_lineup_too_small, t_roundtrip, t_corrupt, t_close_call,
         t_progress, t_fb_rate, t_path_migration, t_path_candidates,
         t_user_presets, t_user_presets_broken, t_review_presets_malformed,
         t_review2_records_broken]


def main() -> int:
    log(f"records tests — Python {sys.version.split()[0]}")
    log("=" * 74)
    for fn in TESTS:
        fn()
        log()
    log("=" * 74)
    log(f"passed {_passed} / {_passed + _failed}"
        + (f", FAILED {_failed}" if _failed else ""))
    OUT.write_text("\n".join(_log), encoding="utf-8")
    print(f"passed {_passed}/{_passed + _failed}; written: {OUT}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
