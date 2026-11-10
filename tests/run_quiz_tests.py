"""demo/quiz.py のドメインテスト。GUI も音も不要。

Run: set AUDIOSPACE_SILENT=1 && python -X utf8 tests\\run_quiz_tests.py
Out: tests/quiz_test_result.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
# 表示の言語を日本語に固定する（テストが期待する文は日本語）
os.environ["AUDIOSPACE_LANG"] = "ja"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import HrirSet, Listener, SpatialEngine  # noqa: E402
from audiospace.geometry import relative_spherical  # noqa: E402
from audiospace.sources import BufferSource  # noqa: E402
from demo.quiz import (ELEVATION_LEVELS, ABWobbleSession,  # noqa: E402
                       CalibrationSession, Difficulty, ElevationSession,
                       FrontElevationSession,
                       QuizSession, TrainingSession,
                       angular_error, direction_name, elevation_name,
                       is_front_back_confusion, is_front_back_testable,
                       position_for)
from demo.speech import Speech  # noqa: E402

OUT = Path(__file__).with_name("quiz_test_result.txt")
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


@test("position_for: 方位から作った座標が元の方位に戻る")
def t_roundtrip() -> str:
    """符号を1つ間違えると定位が左右反転する。往復で検算する。"""
    listener = Listener((0.0, 0.0, 0.0), yaw_deg=0.0)
    worst = 0.0
    for az in range(0, 360, 15):
        pos = position_for(float(az), 4.0)
        back, el, dist = relative_spherical(listener, pos)
        err = angular_error(back, float(az))
        worst = max(worst, err)
        assert err < 1e-6, f"方位{az}度が{back:.3f}度に化けました"
        assert abs(dist - 4.0) < 1e-6, f"距離が{dist}になりました"
    # 仰角つきも確認
    pos = position_for(0.0, 5.0, elevation_deg=30.0)
    _, el, dist = relative_spherical(listener, pos)
    assert abs(el - 30.0) < 1e-6, f"仰角が{el}度になりました"
    return f"24方向すべて往復一致 (最大誤差{worst:.2e}度)、仰角30度も一致"


@test("position_for: 左は西、右は東になる")
def t_left_is_west() -> str:
    left = position_for(90.0, 4.0)
    right = position_for(270.0, 4.0)
    front = position_for(0.0, 4.0)
    assert left[0] < -3.9, f"左は x が負のはず: {left}"
    assert right[0] > 3.9, f"右は x が正のはず: {right}"
    assert front[1] > 3.9, f"前は y が正のはず: {front}"
    return (f"左={left[0]:.1f},{left[1]:.1f} 右={right[0]:.1f},{right[1]:.1f} "
            f"前={front[0]:.1f},{front[1]:.1f}")


@test("direction_name: 代表的な方位が期待どおりの名前になる")
def t_names() -> str:
    cases = {0.0: "前", 90.0: "左", 180.0: "後ろ", 270.0: "右",
             45.0: "左斜め前", 315.0: "右斜め前"}
    for az, want in cases.items():
        got = direction_name(az)
        assert got == want, f"{az}度は「{want}」のはずが「{got}」"
    assert direction_name(358.0) == "前", direction_name(358.0)
    assert direction_name(-90.0) == "右", direction_name(-90.0)
    return "、".join(f"{az:g}度→{n}" for az, n in cases.items())


@test("angular_error: 0度と360度の境界をまたいでも正しい")
def t_error_wrap() -> str:
    assert angular_error(350.0, 10.0) == 20.0, angular_error(350.0, 10.0)
    assert angular_error(10.0, 350.0) == 20.0
    assert angular_error(0.0, 180.0) == 180.0
    assert angular_error(90.0, 90.0) == 0.0
    return "350度と10度の差は20度、0度と180度の差は180度"


@test("前後取り違えの判定")
def t_front_back() -> str:
    # 左前45度を左後ろ135度と答えた = 前後取り違え
    assert is_front_back_confusion(135.0, 45.0)
    assert is_front_back_confusion(45.0, 135.0)
    # 前を後ろと答えた
    assert is_front_back_confusion(180.0, 0.0)
    # 左を右と答えたのは左右取り違えであって前後ではない
    assert not is_front_back_confusion(270.0, 90.0)
    # 正解は取り違えではない
    assert not is_front_back_confusion(45.0, 45.0)
    return "左前45度↔左後ろ135度と前↔後ろを検出、左右取り違えとは区別"


@test("QuizSession: 出題数と同一方向の連続回避")
def t_session_build() -> str:
    s = QuizSession(difficulty=Difficulty.FOUR, n_trials=12, seed=1)
    assert len(s.trials) == 12
    angles = [t.truth_deg for t in s.trials]
    repeats = sum(1 for a, b in zip(angles, angles[1:]) if a == b)
    assert repeats == 0, f"同じ方向が連続しました: {angles}"
    assert set(angles) <= set(Difficulty.FOUR.angles)
    counts = {a: angles.count(a) for a in sorted(set(angles))}
    return f"12問、連続なし、出題分布={counts}"


@test("QuizSession: 回答と採点")
def t_session_score() -> str:
    s = QuizSession(difficulty=Difficulty.FOUR, n_trials=4, seed=2)
    assert s.position == 0 and not s.finished
    # 1問目は正解、2問目は前後取り違え、3問目は正解、4問目は的外れ
    t0 = s.current()
    s.answer(t0.truth_deg)
    t1 = s.current()
    s.answer((180.0 - t1.truth_deg) % 360.0)
    t2 = s.current()
    s.answer(t2.truth_deg)
    t3 = s.current()
    s.answer((t3.truth_deg + 90.0) % 360.0)
    assert s.finished, "4問答えたら終了のはず"
    assert s.n_correct == 2, f"正解数が{s.n_correct}になりました"
    assert s.mean_error_deg > 0
    return (f"正解{s.n_correct}/4、前後取り違え{s.n_front_back}問、"
            f"平均誤差{s.mean_error_deg:.0f}度\n{s.summary()}")


@test("QuizSession: 全問終了後の回答は例外")
def t_session_overrun() -> str:
    s = QuizSession(n_trials=2, seed=3)
    s.answer(0.0)
    s.answer(0.0)
    try:
        s.answer(0.0)
    except RuntimeError as exc:
        return f"想定どおり例外: {exc}"
    raise AssertionError("全問終了後も回答できてしまいました")


@test("QuizSession: 再生回数を記録する")
def t_session_replays() -> str:
    s = QuizSession(n_trials=3, seed=4)
    s.note_replay()
    s.note_replay()
    s.answer(0.0)
    assert s.trials[0].replays == 2, s.trials[0].replays
    assert s.trials[1].replays == 0
    return "1問目の再生回数=2、2問目=0"


@test("Speech: 無音モードでも発話内容を記録する")
def t_speech() -> str:
    sp = Speech(silent=True)
    sp.say("テスト", "です")
    sp.error("失敗しました")
    sp.warn("注意")
    assert sp.log == ["テスト です", "[エラー] 失敗しました", "[警告] 注意"], sp.log
    sp.say("", "  ")
    assert len(sp.log) == 3, "空文字は記録しない"
    return " / ".join(sp.log)


@test("出題した方向が実際に鳴らした音の定位と一致する")
def t_end_to_end() -> str:
    """quiz の座標とエンジンのレンダリングを突き合わせる。

    これが通れば、出題ロジックとエンジンの間で方位規約がずれていない。
    """
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=48000)
    lines = []
    for az, expect in ((90.0, "left"), (270.0, "right"), (0.0, "even")):
        engine = SpatialEngine(hrir, block=512, max_sources=2)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(48000).astype(np.float32) * 0.2
        engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                          position=position_for(az, 4.0))
        out = engine.render_seconds(0.3)
        el = float(np.sum(out[:, 0] ** 2))
        er = float(np.sum(out[:, 1] ** 2))
        ratio = el / max(er, 1e-12)
        if expect == "left":
            assert ratio > 1.5, f"方位{az}度が左に定位していません (L/R={ratio:.2f})"
        elif expect == "right":
            assert ratio < 0.67, f"方位{az}度が右に定位していません (L/R={ratio:.2f})"
        else:
            assert 0.9 < ratio < 1.1, f"正面が偏っています (L/R={ratio:.2f})"
        lines.append(f"{direction_name(az)}({az:g}度): L/R={ratio:.2f}")
    return "\n".join(lines)


@test("回答はリスナーの向きを考慮して world 座標に直される")
def t_answer_with_yaw() -> str:
    """首振りで自分が回れる以上、この変換がないと、向きを変えた
    プレイヤーが正しく答えても不正解になる。"""
    s = QuizSession(difficulty=Difficulty.FOUR, n_trials=2, seed=7)
    trial = s.current()
    truth = trial.truth_deg
    # 左に30度向き直った状態では、正解は相対的に truth-30 度に見える
    s.answer((truth - 30.0) % 360.0, listener_yaw_deg=30.0)
    assert trial.correct, (
        f"向き30度から相対{(truth - 30.0) % 360.0:g}度と答えたのに不正解。"
        f"world換算={trial.answer_deg:g} 正解={truth:g}")
    # 向きを無視して同じ回答をすると不正解になるはず
    t2 = s.current()
    s.answer((t2.truth_deg - 90.0) % 360.0, listener_yaw_deg=0.0)
    assert not t2.correct, "向きを考えない回答が正解になってしまいました"
    return (f"正解{truth:g}度を、向き30度から相対{(truth - 30) % 360:g}度と"
            f"回答して正解と判定")


# -- 訓練モード ------------------------------------------------------------
@test("TrainingSession: 連続正解で終了する")
def t_training_clear() -> str:
    s = TrainingSession(difficulty=Difficulty.FOUR, target_streak=5,
                        max_trials=40, seed=11)
    steps = 0
    while not s.finished:
        t = s.current()
        s.answer(t.truth_deg)
        steps += 1
        assert steps < 50, "終了しません"
    assert s.cleared, "連続正解したのに未達成扱いです"
    assert s.streak == 5, s.streak
    return f"全問正解で{steps}問で終了、連続{s.streak}問、達成={s.cleared}"


@test("TrainingSession: 全問誤答なら上限で打ち切られる")
def t_training_cap() -> str:
    s = TrainingSession(difficulty=Difficulty.FOUR, target_streak=5,
                        max_trials=12, seed=12)
    while not s.finished:
        t = s.current()
        s.answer((t.truth_deg + 90.0) % 360.0)
    assert not s.cleared, "誤答続きなのに達成扱いです"
    assert len(s.answered_trials) == 12, len(s.answered_trials)
    return f"{len(s.answered_trials)}問で打ち切り、達成={s.cleared}"


@test("TrainingSession: 間違えた方向が出やすくなる")
def t_training_adapts() -> str:
    """適応が効いていなければ、苦手方向の出題比率は一様のままになる。"""
    s = TrainingSession(difficulty=Difficulty.FOUR, target_streak=999,
                        max_trials=200, seed=13)
    weak = 180.0          # 後ろだけ必ず間違える
    while not s.finished:
        t = s.current()
        if t.truth_deg == weak:
            s.answer((t.truth_deg + 90.0) % 360.0)
        else:
            s.answer(t.truth_deg)
    counts = {a: 0 for a in Difficulty.FOUR.angles}
    for t in s.answered_trials:
        counts[t.truth_deg] += 1
    share = counts[weak] / len(s.answered_trials)
    assert share > 0.35, (
        f"苦手方向の出題比率が{share * 100:.0f}%しかありません。"
        "一様なら25%なので、適応が効いていません")
    return (f"苦手方向({direction_name(weak)})の出題比率 "
            f"{share * 100:.0f}パーセント (一様なら25パーセント)\n"
            f"出題数={counts}")


@test("TrainingSession: 前後取り違えのフィードバック文")
def t_training_feedback() -> str:
    s = TrainingSession(difficulty=Difficulty.FOUR, seed=14)
    t = s.current()
    truth = t.truth_deg
    s.answer((180.0 - truth) % 360.0)      # 前後を逆に答える
    msg = s.feedback(t)
    assert "前後の取り違え" in msg, msg
    assert direction_name(truth) in msg, msg
    ok = s.current()
    s.answer(ok.truth_deg)
    good = s.feedback(ok)
    assert "正解" in good and "連続" in good, good
    return f"誤答時: {msg}\n正答時: {good}"


# -- 耳介の選定 ------------------------------------------------------------
@test("CalibrationSession: 候補ごとにまとまった問数を出す")
def t_calib_rounds() -> str:
    cands = [("A", "耳介A"), ("B", "耳介B"), ("C", "耳介C")]
    s = CalibrationSession(cands, difficulty=Difficulty.FOUR,
                           trials_per_candidate=4, seed=21)
    assert s.n_trials == 12, s.n_trials
    tags = [t.tag for t in s.trials]
    assert tags == ["A"] * 4 + ["B"] * 4 + ["C"] * 4, tags
    starts = []
    while not s.finished:
        if s.is_round_start():
            starts.append(s.current_tag)
        t = s.current()
        s.answer(t.truth_deg)
    assert starts == ["A", "B", "C"], starts
    return f"12問を4問ずつ3候補に分割、切り替え検出={starts}"


@test("CalibrationSession: 成績の良い耳介が上位に来る")
def t_calib_rank() -> str:
    cands = [("good", "良い耳介"), ("bad", "悪い耳介")]
    s = CalibrationSession(cands, difficulty=Difficulty.FOUR,
                           trials_per_candidate=8, seed=22)
    while not s.finished:
        t = s.current()
        if t.tag == "good":
            s.answer(t.truth_deg)                       # 全問正解
        else:
            s.answer((180.0 - t.truth_deg) % 360.0)     # 全問前後取り違え
    ranked = s.scores()
    assert ranked[0].tag == "good", [r.tag for r in ranked]
    assert ranked[0].accuracy == 1.0, ranked[0].accuracy
    assert ranked[1].n_front_back > 0, "前後取り違えが数えられていません"
    best = s.best()
    assert best is not None and best.tag == "good"
    return (f"1位={ranked[0].describe()}\n2位={ranked[1].describe()}\n"
            f"{s.summary()}")


@test("CalibrationSession: 候補が空なら例外")
def t_calib_empty() -> str:
    try:
        CalibrationSession([], trials_per_candidate=4)
    except ValueError as exc:
        return f"想定どおり例外: {exc}"
    raise AssertionError("空の候補が通ってしまいました")


@test("【重要】4方向では前後の取り違えを検出できない")
def t_fb_testable() -> str:
    """真横は鏡像が自分自身なので、前後を間違えようがない。
    「前後左右」で出題すると前後を試せるのは半分だけになる。"""
    assert is_front_back_testable(0.0), "前は前後を試せるはず"
    assert is_front_back_testable(180.0), "後ろは前後を試せるはず"
    assert not is_front_back_testable(90.0), "真左は鏡像が自分自身"
    assert not is_front_back_testable(270.0), "真右は鏡像が自分自身"
    assert is_front_back_testable(45.0), "斜め前は前後を試せるはず"
    assert is_front_back_testable(135.0)

    four = sum(is_front_back_testable(a) for a in Difficulty.FOUR.angles)
    eight = sum(is_front_back_testable(a) for a in Difficulty.EIGHT.angles)
    twelve = sum(is_front_back_testable(a) for a in Difficulty.TWELVE.angles)
    assert four == 2 and eight == 6 and twelve == 10, (four, eight, twelve)
    return (f"前後を試せる方向の数: 4方向中{four}、8方向中{eight}、"
            f"12方向中{twelve}。8方向以上を使うべき")


@test("集計が前後の試行数を分母にする")
def t_fb_denominator() -> str:
    s = QuizSession(difficulty=Difficulty.EIGHT, n_trials=16, seed=31)
    while not s.finished:
        t = s.current()
        if t.front_back_testable:
            s.answer((180.0 - t.truth_deg) % 360.0)   # 前後を必ず取り違える
        else:
            s.answer(t.truth_deg)
    assert s.n_front_back == s.n_front_back_testable, (
        f"試せた{s.n_front_back_testable}問すべてで取り違えたはずが"
        f"{s.n_front_back}問")
    assert s.n_front_back_testable < len(s.answered_trials), \
        "真横が分母から除かれていません"
    text = s.front_back_text()
    assert "100パーセント" in text, text
    return (f"全{len(s.answered_trials)}問中、前後を試せたのは"
            f"{s.n_front_back_testable}問\n{text}")


@test("4方向のセッションは前後を検出できないと明言する")
def t_fb_warns() -> str:
    s = QuizSession(difficulty=Difficulty.FOUR, n_trials=8, seed=32)
    while not s.finished:
        # 真横だけを答え、前後の出題はスキップできないので正解しておく
        s.answer(s.current().truth_deg)
    text = s.front_back_text()
    assert "検出" in text or "試せた" in text, text
    # 4方向でも前と後ろは試せるので 0 にはならない
    assert s.n_front_back_testable > 0
    return f"前後を試せた問題数={s.n_front_back_testable}／{text}"


@test("耳介の成績が同点なら順位を主張しない")
def t_calib_tie() -> str:
    """前回の実施では5種類中4種類が100パーセントで並び、順位に意味が
    なかった。同点をそのまま1位と読み上げると誤解を招く。"""
    cands = [("a", "耳介A"), ("b", "耳介B"), ("c", "耳介C")]
    s = CalibrationSession(cands, difficulty=Difficulty.EIGHT,
                           trials_per_candidate=8, seed=33)
    while not s.finished:
        s.answer(s.current().truth_deg)          # 全員満点
    text = s.summary()
    assert "同成績" in text, f"同点であることが伝わりません: {text}"
    assert "難易度" in text or "問数" in text, text
    return text


@test("耳介の前後取り違え率は試せた問題を分母にする")
def t_calib_fb_rate() -> str:
    cands = [("good", "良い耳介"), ("bad", "悪い耳介")]
    s = CalibrationSession(cands, difficulty=Difficulty.EIGHT,
                           trials_per_candidate=8, seed=34)
    while not s.finished:
        t = s.current()
        if t.tag == "bad" and t.front_back_testable:
            s.answer((180.0 - t.truth_deg) % 360.0)
        else:
            s.answer(t.truth_deg)
    ranked = s.scores()
    bad = next(x for x in ranked if x.tag == "bad")
    good = next(x for x in ranked if x.tag == "good")
    assert good.front_back_rate == 0.0, good.front_back_rate
    assert bad.front_back_rate == 1.0, (
        f"試せた問題すべてで取り違えたので100%のはず: "
        f"{bad.n_front_back}/{bad.n_fb_testable}")
    assert ranked[0].tag == "good", [x.tag for x in ranked]
    return f"{good.describe()}\n{bad.describe()}"


@test("A/B: 首振りありなしが1問ごとに交互になる")
def t_ab_alternates() -> str:
    s = ABWobbleSession(difficulty=Difficulty.EIGHT, n_trials=32, seed=7)
    on = [t for t in s.trials if t.tag == ABWobbleSession.TAG_ON]
    off = [t for t in s.trials if t.tag == ABWobbleSession.TAG_OFF]
    assert len(on) == len(off) == 16, (len(on), len(off))
    for t in s.trials:
        want = (ABWobbleSession.TAG_ON if t.index % 2 == 0
                else ABWobbleSession.TAG_OFF)
        assert t.tag == want, f"{t.index}問目の条件が {t.tag}"
    assert s.wobble_enabled_for(s.trials[0])
    assert not s.wobble_enabled_for(s.trials[1])
    return f"32問 = あり{len(on)}問 + なし{len(off)}問、交互を確認"


@test("A/B: 条件別に集計され、差が読み上げ文に出る")
def t_ab_summary() -> str:
    s = ABWobbleSession(difficulty=Difficulty.EIGHT, n_trials=8, seed=3)
    while not s.finished:
        t = s.current()
        if t.tag == ABWobbleSession.TAG_ON:
            s.answer(t.truth_deg)                    # ありは全問正解
        else:
            s.answer((t.truth_deg + 90.0) % 360.0)   # なしは全問不正解
    text = s.summary()
    assert "首振りあり" in text and "首振りなし" in text, text
    assert "100パーセント" in text and "0パーセント" in text, text
    assert "＋100ポイント" in text, text
    assert "偶然" in text, f"少問数の注意がありません: {text}"
    lines = s.detail_lines()
    assert any("首振りあり" in x for x in lines)
    assert any("首振りなし" in x for x in lines)
    return text


@test("A/B: 前後を試せない難易度では注意が出る")
def t_ab_fb_warning() -> str:
    s = ABWobbleSession(difficulty=Difficulty.LEFT_RIGHT, n_trials=4, seed=1)
    while not s.finished:
        s.answer(s.current().truth_deg)
    text = s.summary()
    assert "検出できません" in text, text
    return text


@test("A/B: 向きを変えた回答も world 座標で採点される")
def t_ab_with_yaw() -> str:
    s = ABWobbleSession(difficulty=Difficulty.FOUR, n_trials=4, seed=5)
    t = s.current()
    # 90度左を向いた状態で、正解方向を「自分から見た方向」で答える
    rel = (t.truth_deg - 90.0) % 360.0
    trial = s.answer(rel, listener_yaw_deg=90.0)
    assert trial.correct, (trial.answer_deg, t.truth_deg)
    return f"正解{t.truth_deg:g}度を向き90度で回答 -> 正解判定"


# ---------------------------------------------------------------------------
# 上下（仰角）
# ---------------------------------------------------------------------------
VIKING_E = ROOT / "assets" / "hrtf" / "viking" / "subj_E.sofa"


@test("ElevationSession: 4択の高さが均等に出て、同じ音が続かない")
def t_elev_build() -> str:
    """真上は方位を変えても同じ点なので、方位が違っても「同じ音」と数える。"""
    assert [el for el, _ in ELEVATION_LEVELS] == [90.0, 45.0, 0.0, -45.0]
    s = ElevationSession(difficulty=Difficulty.FOUR, n_trials=32, seed=3)
    assert len(s.trials) == 32
    counts = {name: sum(1 for t in s.trials if t.truth_el == el)
              for el, name in ELEVATION_LEVELS}
    assert set(counts.values()) == {8}, f"高さの出題が偏っています: {counts}"
    assert {t.truth_deg for t in s.trials} <= set(Difficulty.FOUR.angles)
    worst = 0
    runs = 0
    for diff in Difficulty:
        for seed in range(100):
            for n in (24, 32, 48):
                s2 = ElevationSession(difficulty=diff, n_trials=n, seed=seed)
                keys = [ElevationSession.sound_key(t.truth_el, t.truth_deg)
                        for t in s2.trials]
                worst = max(worst, sum(1 for a, b in zip(keys, keys[1:])
                                       if a == b))
                runs += 1
    assert worst == 0, f"同じ音の連続が残りました（最悪{worst}回）"
    return f"32問で高さ別={counts}。{runs}通りの並びで同じ音の連続なし"


@test("FrontElevationSession: 方位は正面だけ、高さは均等、連続は許す")
def t_elev_front() -> str:
    """同じ高さ＝同じ音なので、連続を禁じると「直前と違う方」を選ぶだけで
    当て推量の水準が上がる。禁じていないことを確かめる。"""
    s = FrontElevationSession(n_trials=24, seed=1)
    assert {t.truth_deg for t in s.trials} == {0.0}, "正面以外の方位が出ました"
    counts = {name: sum(1 for t in s.trials if t.truth_el == el)
              for el, name in ELEVATION_LEVELS}
    assert set(counts.values()) == {6}, counts
    any_repeat = any(
        any(a.truth_el == b.truth_el for a, b in
            zip(FrontElevationSession(seed=k).trials,
                FrontElevationSession(seed=k).trials[1:]))
        for k in range(20))
    assert any_repeat, "同じ高さの連続が一度も起きません（禁止されている）"
    s.answer_elevation(s.trials[0].truth_el)
    line = s.detail_lines()[0]
    assert "方位" not in line, f"正面固定なのに方位を書いています: {line}"
    assert "25" in s.summary(), s.summary()
    return f"24問、方位は正面のみ、高さ別={counts}、連続あり。{line}"


@test("ElevationSession: 採点・上下取り違え・当て推量の注記")
def t_elev_score() -> str:
    s = ElevationSession(difficulty=Difficulty.FOUR, n_trials=6, seed=1)
    truths = [t.truth_el for t in s.trials]
    # 1問目は正解、2問目は上下を逆に（水平なら上と答える）、残りは水平
    s.answer_elevation(truths[0])
    wrong = -truths[1] if truths[1] else 45.0
    s.answer_elevation(wrong)
    for _ in range(4):
        s.answer_elevation(0.0)
    assert s.trials[0].correct
    assert not s.trials[1].correct
    assert s.trials[1].up_down == (truths[1] != 0.0)
    assert all(not t.front_back for t in s.trials), "前後は数えないはず"
    text = s.summary()
    for token in ("正答率", "当て推量", "25", "高さ別"):
        assert token in text, f"結果文に「{token}」がありません: {text}"
    try:
        s.answer_elevation(0.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("全問終了後の回答が通りました")
    try:
        ElevationSession(n_trials=3, seed=1).answer(0.0)
    except TypeError:
        pass
    else:
        raise AssertionError("方位での回答が通りました")
    assert elevation_name(40.0) == "上" and elevation_name(-50.0) == "下"
    assert elevation_name(80.0) == "真上"
    return text


def _render_left_spectrum(hrir: HrirSet, az: float, el: float,
                          pitch: float = 0.0) -> np.ndarray:
    """方位・仰角に置いたノイズを鳴らし、左耳の対数スペクトルを返す。

    HRTF だけの違いを見るので、前後の強調は切る。前後の処理は上側と下側で
    重みが違う（下は45度まで全量）ため、真横でも上と下で少し違う音になる。
    """
    engine = SpatialEngine(hrir, block=512, max_sources=2,
                           front_back_depth=0.0)
    engine.set_listener(pitch_deg=pitch)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(48000).astype(np.float32) * 0.2
    engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                      position=position_for(az, 4.0, elevation_deg=el))
    out = engine.render_seconds(0.5)[4800:, 0]
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out)))) ** 2
    # 1/3 オクターブ程度に均す（細かい櫛の凹凸で差を水増ししない）
    freqs = np.fft.rfftfreq(len(out), 1 / 48000)
    edges = 2000.0 * 2 ** (np.arange(0, 13) / 3)        # 2k〜32k の手前まで
    bands = [spec[(freqs >= lo) & (freqs < hi)].mean()
             for lo, hi in zip(edges, edges[1:]) if hi < 20000]
    return 10 * np.log10(np.asarray(bands) + 1e-20)


@test("全天球の Viking では上と下が違う音で鳴り、仰角0度だけでは同じ音になる")
def t_elev_render() -> str:
    """上下の出題が音として成立しているかを、耳ではなく数値で確かめる。

    上下の手がかりは 2kHz 以上のスペクトル形状にしかない。全天球なら
    上と下で高域の形が変わり、仰角0度だけの HRTF では最も近い水平の
    方向が選ばれて、上も下もまったく同じ出力になる。後者が、アプリで
    上下の測定がない HRTF のとき Shift+F3 と Ctrl+F3 を断る理由。
    """
    if not VIKING_E.exists():
        return f"{VIKING_E.name} が未取得のため省略"
    full = HrirSet.from_sofa(VIKING_E, sample_rate=48000)
    # 仰角0度だけに絞る。既定の horizontal() は ±5度を残すので、上下は
    # +5度と-5度に寄せられて僅かに違う音になる（実測で約5.5dB）。
    flat = full.horizontal(tolerance_deg=0.0)
    lines = []
    for az in (0.0, 90.0):
        up = _render_left_spectrum(full, az, 45.0)
        mid = _render_left_spectrum(full, az, 0.0)
        down = _render_left_spectrum(full, az, -45.0)
        d_ud = float(np.mean(np.abs(up - down)))
        d_um = float(np.mean(np.abs(up - mid)))
        assert d_ud > 1.0, (f"方位{az:g}度で上と下の高域差が"
                            f"{d_ud:.2f}dB しかありません")
        f_up = _render_left_spectrum(flat, az, 45.0)
        f_down = _render_left_spectrum(flat, az, -45.0)
        d_flat = float(np.max(np.abs(f_up - f_down)))
        assert d_flat < 1e-6, (f"水平面だけの HRTF で上と下が違う音に"
                               f"なりました ({d_flat:.3g}dB)")
        lines.append(f"{direction_name(az)}: 上と下の差 平均{d_ud:.1f}dB、"
                     f"上と水平の差 平均{d_um:.1f}dB、仰角0度のみでは差 0")
    return "\n".join(lines)


@test("顔を上に45度向けると、上45度の音が正面の音と同じになる")
def t_pitch_render() -> str:
    """Page Up / Page Down の顔の上下がエンジンの音まで届いているか。

    上を45度向けば、上45度の音は顔から見て正面になる。顔を水平にして
    正面で鳴らした音と、HRTF の同じ行を使うので出力は一致するはず。
    真上の音も、顔を真上に向ければ正面と同じになる。
    """
    if not VIKING_E.exists():
        return f"{VIKING_E.name} が未取得のため省略"
    full = HrirSet.from_sofa(VIKING_E, sample_rate=48000)
    ahead = _render_left_spectrum(full, 0.0, 0.0)
    up45_tilted = _render_left_spectrum(full, 0.0, 45.0, pitch=45.0)
    up45_level = _render_left_spectrum(full, 0.0, 45.0)
    top_tilted = _render_left_spectrum(full, 0.0, 90.0, pitch=90.0)
    d_tilt = float(np.max(np.abs(up45_tilted - ahead)))
    d_top = float(np.max(np.abs(top_tilted - ahead)))
    d_level = float(np.mean(np.abs(up45_level - ahead)))
    assert d_tilt < 1e-3, f"上を向いても上45度の音が正面と違います ({d_tilt:.3g}dB)"
    assert d_top < 1e-3, f"真上を向いても真上の音が正面と違います ({d_top:.3g}dB)"
    assert d_level > 1.0, "顔が水平のときは上45度と正面が違う音のはずです"
    return (f"上45度を向いたときの上45度の音と正面の差 {d_tilt:.1g}dB、"
            f"真上を向いたときの真上の音と正面の差 {d_top:.1g}dB、"
            f"顔が水平なら上45度と正面の差は平均{d_level:.1f}dB")


@test("首かしげの揺れで、上の音と下の音の左右差が逆向きに動く")
def t_roll_render() -> str:
    """首振りが前後で相関 -0.999 を示したのと同じ確かめ方を上下で行う。

    上45度と下45度（どちらも正面）の音を首かしげの揺れつきで鳴らし、
    短い区間ごとの左右の音量差（dB）を並べる。上と下で逆向きに動けば
    相関は強い負になる。正面・水平の音は揺れてもほとんど動かない。
    """
    if not VIKING_E.exists():
        return f"{VIKING_E.name} が未取得のため省略"
    from audiospace import ROLL_PRESETS

    full = HrirSet.from_sofa(VIKING_E, sample_rate=48000)
    wobble = ROLL_PRESETS["clear"]
    seconds = 2.0 / wobble.rate_hz          # 2周期

    def ild_series(el: float) -> np.ndarray:
        engine = SpatialEngine(full, block=512, max_sources=2)
        engine.set_roll_wobble(wobble)
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(48000 * 2).astype(np.float32) * 0.2
        engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                          position=position_for(0.0, 4.0, elevation_deg=el))
        out = engine.render_seconds(seconds)
        hop = 2400                           # 50ms ごと
        n = len(out) // hop
        seg = out[:n * hop].reshape(n, hop, 2)
        energy = (seg ** 2).sum(axis=1) + 1e-12
        return 10 * np.log10(energy[:, 0] / energy[:, 1])

    up, down, level = ild_series(45.0), ild_series(-45.0), ild_series(0.0)
    corr = float(np.corrcoef(up, down)[0, 1])
    swing_up = float(up.max() - up.min())
    swing_down = float(down.max() - down.min())
    swing_level = float(level.max() - level.min())
    assert corr < -0.8, f"上と下が逆向きに動いていません（相関{corr:.3f}）"
    assert swing_level < 0.25 * min(swing_up, swing_down), \
        (f"正面水平が揺れで動きすぎています（{swing_level:.2f}dB、"
         f"上{swing_up:.2f}dB、下{swing_down:.2f}dB）")
    return (f"左右差の動き: 上45度 {swing_up:.1f}dB、下45度 {swing_down:.1f}dB、"
            f"正面水平 {swing_level:.2f}dB。上と下の相関 {corr:.3f}")


@test("ElevationCalibrationSession: 耳介ごとに4つの高さを均等に、記録は上下として積む")
def t_elev_calib() -> str:
    import tempfile

    from demo.quiz import ElevationCalibrationSession
    from demo.records import RecordStore

    cands = [("subj_A.sofa", "耳介A"), ("subj_B.sofa", "耳介B"),
             ("subj_C.sofa", "耳介C")]
    s = ElevationCalibrationSession(cands, trials_per_candidate=12, seed=5)
    assert s.n_trials == 36 and len(s.trials) == 36
    for k, (tag, _) in enumerate(cands):
        block = s.trials[k * 12:(k + 1) * 12]
        assert {t.tag for t in block} == {tag}, "耳介が混ざっています"
        counts = {el: sum(1 for t in block if t.truth_el == el)
                  for el, _ in ELEVATION_LEVELS}
        assert set(counts.values()) == {3}, counts
        assert {t.truth_deg for t in block} == {0.0}, "正面以外が出ました"
    assert s.is_round_start()
    # 耳介Aは全問正解、Bは上下を逆に、Cは全部水平と答える
    flipped = {90.0: -45.0, 45.0: -45.0, 0.0: 45.0, -45.0: 45.0}
    for t in s.trials:
        if t.tag == "subj_A.sofa":
            s.answer_elevation(t.truth_el)
        elif t.tag == "subj_B.sofa":
            s.answer_elevation(flipped[t.truth_el])
        else:
            s.answer_elevation(0.0)
    scores = s.scores()
    assert scores[0].tag == "subj_A.sofa", [x.tag for x in scores]
    b = next(x for x in scores if x.tag == "subj_B.sofa")
    # 真上・上・下の各3問が上下の取り違え。水平を上と答えたのは1段ずれ
    assert b.n_front_back == 9, f"上下の取り違えの数が違います: {b.n_front_back}"
    assert "25" in s.summary(), s.summary()
    lines = s.detail_lines()
    assert any("上と下の取り違え" in line for line in lines), lines

    with tempfile.TemporaryDirectory() as tmp:
        store = RecordStore(Path(tmp) / "elev.json", fb_name="上下")
        store.update(scores, anchor_tag="subj_A.sofa", n_directions=8)
        store.save()
        again = RecordStore(Path(tmp) / "elev.json", fb_name="上下")
        text = "\n".join(again.detail_lines())
        assert "上下" in text and "前後" not in text, text
    return f"{s.summary()}\n" + "\n".join(lines)


@test("回る向き: 水平16通りと頭上を並べ、同じ音は続かない")
def t_rotation_build() -> str:
    from demo.quiz import RotationSession
    s = RotationSession(seed=3)
    assert len(s.trials) == 24
    horiz = [(t.truth_motion, t.truth_deg) for t in s.trials
             if t.kind == "horizontal"]
    over = [t.truth_motion for t in s.trials if t.kind == "overhead"]
    assert len(horiz) == 16 and len(set(horiz)) == 16, horiz
    assert sorted(over) == ["to_back"] * 4 + ["to_front"] * 4, over
    for t in s.trials:
        if t.truth_motion == "to_back":
            assert t.truth_deg == 0.0
        if t.truth_motion == "to_front":
            assert t.truth_deg == 180.0
    keys = [(t.truth_motion, t.truth_deg) for t in s.trials]
    assert all(a != b for a, b in zip(keys, keys[1:])), keys
    flat = RotationSession(seed=3, include_overhead=False)
    assert all(t.kind == "horizontal" for t in flat.trials)
    try:
        s.answer(0.0)
    except TypeError:
        pass
    else:
        raise AssertionError("方位での回答を受け付けました")
    return "水平16通り（各1回）＋頭上8問、隣り合う同じ音なし。水平だけの回も作れる"


@test("回る向き: 位置は聞き手から見た半周（左回りは前→左→後ろ）")
def t_rotation_path() -> str:
    from demo.quiz import RotationTrial

    def az_el(p):
        return relative_spherical(Listener(), p)[:2]

    left = RotationTrial(index=0, truth_deg=0.0, distance_m=4.0,
                         truth_motion="left")
    az, _ = az_el(left.position(0.5))
    assert abs(az - 90.0) < 1e-6, az               # 前から左へ
    az, _ = az_el(left.position(1.0))
    assert abs(az - 180.0) < 1e-6, az              # 後ろで終わる
    right = RotationTrial(index=0, truth_deg=90.0, distance_m=4.0,
                          truth_motion="right")
    az, _ = az_el(right.position(0.5))
    assert abs(az % 360.0) < 1e-6, az              # 左から前を通って右へ
    over = RotationTrial(index=0, truth_deg=0.0, distance_m=4.0,
                         truth_motion="to_back")
    _, el = az_el(over.position(0.5))
    assert abs(el - 90.0) < 1e-6, el               # 真上を通る
    az, el = az_el(over.position(0.25))
    assert abs(az) < 1e-6 and abs(el - 45.0) < 1e-6, (az, el)
    az, el = az_el(over.position(0.75))
    assert abs(az - 180.0) < 1e-6 and abs(el - 45.0) < 1e-6, (az, el)
    # 向きを変えていても、その人から見て同じ道筋になる
    turned = Listener(yaw_deg=90.0)
    az, _ = relative_spherical(turned, left.position(0.5, yaw_deg=90.0))[:2]
    assert abs(az - 90.0) < 1e-6, az
    # 範囲外の割合は端に止める
    assert left.position(1.7) == left.position(1.0)
    return "左回りは前→左→後ろ、右回りは左→前→右、頭上は真上を通る。向きにも追従"


@test("回る向き: 採点と、逆向き（前後の取り違え）の数え方")
def t_rotation_score() -> str:
    from demo.quiz import RotationSession
    s = RotationSession(seed=4)
    reverse = {"left": "right", "right": "left",
               "to_front": "to_back", "to_back": "to_front"}
    other_kind = {"left": "to_back", "right": "to_front",
                  "to_front": "left", "to_back": "right"}
    for i, t in enumerate(s.trials):
        if i % 3 == 0:
            s.answer_rotation(t.truth_motion)
        elif i % 3 == 1:
            s.answer_rotation(reverse[t.truth_motion])
        else:
            s.answer_rotation(other_kind[t.truth_motion])
    assert s.finished
    assert s.n_correct == 8 and s.n_reversed == 8, (s.n_correct, s.n_reversed)
    text = s.summary()
    assert "24問中8問正解" in text and "50パーセント" in text, text
    assert "逆に答えたのが8問" in text, text
    lines = s.detail_lines()
    assert any("回る向きの取り違え" in x for x in lines), lines
    assert any("から）" in x for x in lines), lines
    try:
        s.answer_rotation("up")
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("知らない回り方を受け付けました")
    return text


@test("回る向き: 左回りと右回りで、後ろを通るほうが途中で暗くなる（実測 HRTF）")
def t_rotation_render() -> str:
    """回る向きの手がかりが音に出ているかを数値で確かめる。左から回り
    始めると、左回りは後ろを、右回りは前を通る。前後の強調（既定 35%）と
    耳介 E の性質で、後ろを通る間は高い音域が弱まるはず。"""
    if not VIKING_E.exists():
        return f"{VIKING_E.name} が未取得のため省略"
    from demo.quiz import ROTATION_ARC_SECONDS, RotationTrial
    hrir = HrirSet.from_sofa(VIKING_E, sample_rate=48000)
    noise = (np.random.default_rng(5).standard_normal(48000) * 0.2
             ).astype(np.float32)

    def brightness_profile(trial) -> list[float]:
        engine = SpatialEngine(hrir, block=512, max_sources=2)
        handle = engine.add_source(BufferSource(noise, loop=True),
                                   position=trial.position(0.0),
                                   reverb_send=0.0)
        n = int(ROTATION_ARC_SECONDS * 48000 / 512)
        out = []
        for b in range(n):
            engine.set_source_position(handle, trial.position(b / (n - 1)))
            out.append(engine.render())
        x = np.concatenate(out).mean(axis=1)
        thirds = np.array_split(x, 3)
        values = []
        for part in thirds:
            spec = np.abs(np.fft.rfft(part)) ** 2
            f = np.fft.rfftfreq(len(part), 1 / 48000)
            values.append(float(10 * np.log10(spec[f >= 4000].mean()
                                              / spec[(f > 200) & (f < 2000)]
                                              .mean())))
        return values

    left = brightness_profile(RotationTrial(0, 90.0, 4.0,
                                            truth_motion="left"))
    right = brightness_profile(RotationTrial(0, 90.0, 4.0,
                                             truth_motion="right"))
    assert left[1] < right[1] - 2.0, (left, right)
    over = brightness_profile(RotationTrial(0, 0.0, 4.0,
                                            truth_motion="to_back"))
    assert over[0] > over[2] + 2.0, over
    return (f"左から回り始めたときの真ん中の明るさ: 左回り（後ろを通る）"
            f"{left[1]:+.1f}dB、右回り（前を通る）{right[1]:+.1f}dB。"
            f"頭上を前から後ろへ: 始め {over[0]:+.1f}dB、終わり {over[2]:+.1f}dB")


TESTS = [t_roundtrip, t_left_is_west, t_names, t_error_wrap, t_front_back,
         t_fb_testable, t_fb_denominator, t_fb_warns,
         t_session_build, t_session_score, t_session_overrun,
         t_session_replays, t_answer_with_yaw, t_speech, t_end_to_end,
         t_training_clear, t_training_cap, t_training_adapts,
         t_training_feedback,
         t_calib_rounds, t_calib_rank, t_calib_empty,
         t_calib_tie, t_calib_fb_rate,
         t_ab_alternates, t_ab_summary, t_ab_fb_warning, t_ab_with_yaw,
         t_elev_build, t_elev_front, t_elev_score, t_elev_render,
         t_pitch_render, t_roll_render, t_elev_calib,
         t_rotation_build, t_rotation_path, t_rotation_score,
         t_rotation_render]


def main() -> int:
    log(f"quiz domain tests — Python {sys.version.split()[0]}")
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
