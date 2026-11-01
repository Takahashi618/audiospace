"""定位テストのドメインロジック。

GUI にも音声出力にも依存しない純粋なロジックなので、ヘッドレスで検証
できる。出題・採点・統計だけを担当し、音を鳴らすのは呼び出し側。

方位の規約は audiospace 本体と同じ (0度=正面、反時計回り、90度=左)。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from audiospace.i18n import join_sentences, list_sep, tr


def question_no(n: int) -> str:
    """「3問目」（英語は「Question 3」）。n は1始まり。"""
    return tr(f"{n}問目", f"Question {n}")


def mark_text(correct: bool) -> str:
    """正解か不正解かの読み上げ。"""
    return tr("正解", "Correct") if correct else tr("不正解", "Wrong")


def not_answered(n: int) -> str:
    return tr(f"{n}問目 未回答", f"Question {n}: not answered")


#: 結果がまだ無いときの読み上げ
NO_ANSWERS = tr("まだ回答がありません。", "No answers yet.")
#: 終わった問題にもう一度答えようとしたとき
ALL_DONE = tr("全問終了済みです", "All questions are finished")
#: 前後を試せない難易度での注意
NO_FRONT_BACK = tr("この難易度では前後の取り違えを検出できません。"
                   "8方向以上を選んでください。",
                   "This difficulty cannot detect front/back confusions. "
                   "Choose 8 directions or more.")
#: 問数が少ないときの注意
FEW_QUESTIONS = tr("問数が少ないため、小さい差は偶然の範囲です。",
                   "With this few questions, small differences may be "
                   "chance.")


def _result_line(t, truth: str, answer: str, extra: str,
                 cond: str = "") -> str:
    """1問の結果の行（方向の問題）。"""
    n = t.index + 1
    return tr(f"{n}問目{cond} {mark_text(t.correct)}。正解は{truth}、"
              f"回答は{answer}。誤差{t.error_deg:.0f}度{extra}",
              f"Question {n}{cond}: {mark_text(t.correct)}. "
              f"The correct answer was {truth}, you answered {answer}. Error "
              f"{t.error_deg:.0f} degrees{extra}")


#: 結果の行に足す「前後の取り違え」
FB_EXTRA = tr("、前後の取り違え", ", front/back confusion")


class Difficulty(Enum):
    """出題する方向の粒度。"""

    LEFT_RIGHT = (tr("左右のみ", "Left and right only"), (90.0, 270.0))
    FOUR = (tr("前後左右", "4 directions"),
            (0.0, 90.0, 180.0, 270.0))
    EIGHT = (tr("8方向", "8 directions"),
             tuple(float(i * 45) for i in range(8)))
    TWELVE = (tr("12方向", "12 directions"),
              tuple(float(i * 30) for i in range(12)))

    def __init__(self, label: str, angles: tuple[float, ...]) -> None:
        self.label = label
        self.angles = angles


#: 方位を方向名にするための対応表 (反時計回り)
DIRECTION_NAMES: tuple[tuple[float, str], ...] = (
    (0.0, tr("前", "front")),
    (30.0, tr("左前30度", "front left 30 degrees")),
    (45.0, tr("左斜め前", "front left")),
    (60.0, tr("左前60度", "front left 60 degrees")),
    (90.0, tr("左", "left")),
    (120.0, tr("左後ろ120度", "back left 120 degrees")),
    (135.0, tr("左斜め後ろ", "back left")),
    (150.0, tr("左後ろ150度", "back left 150 degrees")),
    (180.0, tr("後ろ", "back")),
    (210.0, tr("右後ろ150度", "back right 150 degrees")),
    (225.0, tr("右斜め後ろ", "back right")),
    (240.0, tr("右後ろ120度", "back right 120 degrees")),
    (270.0, tr("右", "right")),
    (300.0, tr("右前60度", "front right 60 degrees")),
    (315.0, tr("右斜め前", "front right")),
    (330.0, tr("右前30度", "front right 30 degrees")),
)


def direction_name(azimuth_deg: float) -> str:
    """方位に最も近い方向名を返す。"""
    az = azimuth_deg % 360.0
    best = min(DIRECTION_NAMES,
               key=lambda item: abs(_signed_delta(az, item[0])))
    return best[1]


def _signed_delta(a: float, b: float) -> float:
    """a から b への角度差を -180..180 で返す。"""
    return (a - b + 180.0) % 360.0 - 180.0


def angular_error(answer_deg: float, truth_deg: float) -> float:
    """回答と正解の角度差 (絶対値、0..180)。"""
    return abs(_signed_delta(answer_deg, truth_deg))


def is_front_back_testable(truth_deg: float,
                           tolerance_deg: float = 30.0) -> bool:
    """その方向で前後の取り違えが起こりうるか。

    真横（方位90度・270度）は鏡像が自分自身になるので、前後を間違えようが
    ない。「前後左右」の4方向で出題すると、前後を実際に試せているのは
    前と後ろの半分だけになる。正答率だけ見ていると、この希釈に気づかない
    まま「前後が改善した」と誤読しかねないので、集計で分けて数える。
    """
    mirrored = (180.0 - truth_deg) % 360.0
    return angular_error(mirrored, truth_deg) > tolerance_deg


def is_front_back_confusion(answer_deg: float, truth_deg: float,
                            tolerance_deg: float = 30.0) -> bool:
    """前後取り違え (左右は合っているが前後が逆) かどうか。

    視覚障害者向け音響ゲームで最も報告の多い誤りなので、単なる誤答とは
    分けて数える。合成HRIRでは原理的に区別できないため、実測HRTFの
    効果を測る指標にもなる。
    """
    mirrored = (180.0 - truth_deg) % 360.0
    return (angular_error(answer_deg, mirrored) <= tolerance_deg
            and angular_error(answer_deg, truth_deg) > tolerance_deg)


@dataclass
class Trial:
    """1 問。"""

    index: int
    truth_deg: float
    distance_m: float
    answer_deg: float | None = None
    replays: int = 0
    tag: str = ""            # 耳介の比較などで、どの条件だったかを残す

    @property
    def answered(self) -> bool:
        return self.answer_deg is not None

    @property
    def error_deg(self) -> float:
        if self.answer_deg is None:
            return 0.0
        return angular_error(self.answer_deg, self.truth_deg)

    @property
    def correct(self) -> bool:
        """許容 22.5 度以内なら正解 (8方向の隣と混ざらない範囲)。"""
        return self.answered and self.error_deg <= 22.5

    @property
    def front_back(self) -> bool:
        if self.answer_deg is None:
            return False
        return is_front_back_confusion(self.answer_deg, self.truth_deg)

    @property
    def front_back_testable(self) -> bool:
        """この問題で前後の取り違えが起こりうるか（真横なら不可）。"""
        return is_front_back_testable(self.truth_deg)


@dataclass
class QuizSession:
    """1 セッション分の出題と集計。"""

    difficulty: Difficulty = Difficulty.FOUR
    n_trials: int = 10
    distance_m: float = 4.0
    seed: int | None = None
    trials: list[Trial] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.trials = []
        self._build()

    def _build(self) -> None:
        """同じ方向が続けて出ないように並べる。"""
        angles = list(self.difficulty.angles)
        picks: list[float] = []
        while len(picks) < self.n_trials:
            batch = angles[:]
            self._rng.shuffle(batch)
            if picks and batch[0] == picks[-1] and len(batch) > 1:
                batch[0], batch[1] = batch[1], batch[0]
            picks.extend(batch)
        for i, az in enumerate(picks[:self.n_trials]):
            self.trials.append(Trial(index=i, truth_deg=az,
                                     distance_m=self.distance_m))

    # -- 進行 ---------------------------------------------------------
    @property
    def position(self) -> int:
        """未回答の最初の問題番号 (0 始まり)。全部終わっていれば n_trials。"""
        for t in self.trials:
            if not t.answered:
                return t.index
        return self.n_trials

    @property
    def finished(self) -> bool:
        return self.position >= self.n_trials

    def current(self) -> Trial | None:
        return None if self.finished else self.trials[self.position]

    def answer(self, azimuth_deg: float,
               listener_yaw_deg: float = 0.0) -> Trial:
        """回答する。

        `listener_yaw_deg` はプレイヤーが向きを変えている場合の現在の向き。
        回答は「いまの自分から見た方向」なので、world 座標に直してから
        正解と比べる。首振りで自分が回れる以上、この変換を省くと、
        向きを変えたプレイヤーが正しく答えても不正解になる。
        """
        trial = self.current()
        if trial is None:
            raise RuntimeError(ALL_DONE)
        trial.answer_deg = (azimuth_deg + listener_yaw_deg) % 360.0
        return trial

    def note_replay(self) -> None:
        trial = self.current()
        if trial is not None:
            trial.replays += 1

    # -- 集計 ---------------------------------------------------------
    @property
    def answered_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.answered]

    @property
    def n_correct(self) -> int:
        return sum(1 for t in self.answered_trials if t.correct)

    @property
    def n_front_back(self) -> int:
        return sum(1 for t in self.answered_trials if t.front_back)

    @property
    def n_front_back_testable(self) -> int:
        """前後の取り違えが起こりうる出題が何問あったか。

        真横は鏡像が自分自身なので前後を試せない。この数を見ないと、
        「前後左右」の4方向で前後の検証をしたつもりになってしまう
        （実際は半分しか試せていない）。
        """
        return sum(1 for t in self.answered_trials if t.front_back_testable)

    def front_back_text(self) -> str:
        """前後についての読み上げ文。試行数の裏づけつき。"""
        testable = self.n_front_back_testable
        if testable == 0:
            return NO_FRONT_BACK
        if self.n_front_back:
            rate = self.n_front_back / testable * 100
            return tr(f"前後を試せた{testable}問のうち、"
                      f"取り違えが{self.n_front_back}問、"
                      f"{rate:.0f}パーセントでした。",
                      f"Of the {testable} questions that tested front and "
                      f"back, {self.n_front_back} were confused, "
                      f"{rate:.0f} percent.")
        return tr(f"前後を試せた{testable}問で、取り違えはありませんでした。",
                  f"No front/back confusions in the {testable} questions "
                  "that tested front and back.")

    @property
    def mean_error_deg(self) -> float:
        done = self.answered_trials
        if not done:
            return 0.0
        return sum(t.error_deg for t in done) / len(done)

    def summary(self) -> str:
        """読み上げ用の結果文。"""
        done = self.answered_trials
        if not done:
            return NO_ANSWERS
        rate = self.n_correct / len(done) * 100
        parts = [
            tr(f"{len(done)}問中{self.n_correct}問正解、"
               f"正答率{rate:.0f}パーセント。",
               f"{self.n_correct} of {len(done)} correct, {rate:.0f} "
               "percent."),
            tr(f"平均誤差{self.mean_error_deg:.0f}度。",
               f"Mean error {self.mean_error_deg:.0f} degrees."),
            self.front_back_text(),
        ]
        if self.n_front_back:
            parts.append(tr(
                "前後が苦手な場合は、首振りを強めるか、"
                "実測HRTFの耳介を変えると改善することがあります。",
                "If front and back are hard, a stronger head wobble or "
                "another measured pinna may help."))
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        """1 問ずつの結果。リストボックス表示用。"""
        lines = []
        for t in self.trials:
            if not t.answered:
                lines.append(not_answered(t.index + 1))
                continue
            extra = FB_EXTRA if t.front_back else ""
            lines.append(_result_line(t, direction_name(t.truth_deg),
                                      direction_name(t.answer_deg), extra))
        return lines


def position_for(azimuth_deg: float, distance_m: float,
                 elevation_deg: float = 0.0) -> tuple[float, float, float]:
    """リスナーが北を向いている前提で、方位から世界座標を作る。

    audiospace の座標系は x=東 / y=北 / z=上、方位は正面から反時計回り。
    したがって方位 90 度 (左) は西、つまり x が負になる。
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horiz = distance_m * math.cos(el)
    return (-horiz * math.sin(az), horiz * math.cos(az),
            distance_m * math.sin(el))


# ---------------------------------------------------------------------------
# 首振りの A/B 検証
# ---------------------------------------------------------------------------
class ABWobbleSession(QuizSession):
    """首振りの効果を、同一セッション内の A/B 比較で測る。

    前後の取り違えが減ったのが首振りのおかげか、単なる練習効果かは、
    別々のセッションを比べても分からない（聞き手は回を重ねるほど上達
    する）。そこで 1 回のセッションの中で、首振りありの問題となしの
    問題を交互に混ぜ、条件別に集計する。上達は両条件へ等しく効くので、
    差を取れば打ち消される。耳介選定で基準を同席させるのと同じ考え方。

    出題の並びは QuizSession と同じ生成（同方向の連続なし）で、条件は
    偶数問=あり、奇数問=なしを機械的に割り当てる。方向の混ざり方は
    どちらの条件でもシャッフルされた同じ母集団になる。
    """

    TAG_ON = "wobble_on"
    TAG_OFF = "wobble_off"

    def _build(self) -> None:
        super()._build()
        for t in self.trials:
            t.tag = self.TAG_ON if t.index % 2 == 0 else self.TAG_OFF

    def wobble_enabled_for(self, trial: Trial) -> bool:
        """この問題を首振りありで鳴らすべきか。"""
        return trial.tag == self.TAG_ON

    def _condition(self, tag: str) -> tuple[int, int, int, int]:
        """(回答数, 正解数, 前後取り違え, 前後を試せた数)。"""
        done = [t for t in self.answered_trials if t.tag == tag]
        return (len(done),
                sum(1 for t in done if t.correct),
                sum(1 for t in done if t.front_back),
                sum(1 for t in done if t.front_back_testable))

    def summary(self) -> str:
        if not self.answered_trials:
            return NO_ANSWERS
        n_on, c_on, fb_on, ft_on = self._condition(self.TAG_ON)
        n_off, c_off, fb_off, ft_off = self._condition(self.TAG_OFF)
        parts: list[str] = []
        for on, n, c, fb, ft in ((True, n_on, c_on, fb_on, ft_on),
                                 (False, n_off, c_off, fb_off, ft_off)):
            if n == 0:
                continue
            if ft:
                fb_text = tr(f"、前後の取り違え{fb}問（試せた{ft}問中）",
                             f", {fb} front/back confusions (of {ft} that "
                             "tested it)")
            else:
                fb_text = ""
            if on:
                parts.append(tr(f"首振りありは{n}問中{c}問正解、"
                                f"{c / n * 100:.0f}パーセント{fb_text}。",
                                f"With head wobble: {c} of {n} correct, "
                                f"{c / n * 100:.0f} percent{fb_text}."))
            else:
                parts.append(tr(f"首振りなしは{n}問中{c}問正解、"
                                f"{c / n * 100:.0f}パーセント{fb_text}。",
                                f"Without head wobble: {c} of {n} correct, "
                                f"{c / n * 100:.0f} percent{fb_text}."))
        if n_on and n_off:
            diff = (c_on / n_on - c_off / n_off) * 100
            sign = tr("＋", "+") if diff > 0 else ""
            parts.append(tr(f"正答率の差は、首振りあり{sign}{diff:.0f}ポイント。",
                            f"Difference in accuracy: {sign}{diff:.0f} points "
                            "with head wobble."))
            if ft_on == 0 and ft_off == 0:
                parts.append(NO_FRONT_BACK)
            if min(n_on, n_off) < 16:
                parts.append(join_sentences(
                    FEW_QUESTIONS, tr("もう一度実施すると確かになります。",
                                      "Running it again will make it surer.")))
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        lines = []
        for t in self.trials:
            if not t.answered:
                lines.append(not_answered(t.index + 1))
                continue
            if t.tag == self.TAG_ON:
                cond = tr("(首振りあり)", " (head wobble)")
            else:
                cond = tr("(首振りなし)", " (no head wobble)")
            extra = FB_EXTRA if t.front_back else ""
            lines.append(_result_line(t, direction_name(t.truth_deg),
                                      direction_name(t.answer_deg), extra,
                                      cond))
        return lines


# ---------------------------------------------------------------------------
# 訓練モード
# ---------------------------------------------------------------------------
class TrainingSession(QuizSession):
    """間違えた方向を重点的に出し直す適応型の練習。

    聴感確認で「最初は前後を取り違えるが、慣れると分かるようになる」と
    分かった。つまり訓練で解決する。ならば、その慣れを短時間で得られる
    形にした方がよい。

    通常のクイズとの違い:
    - 出題は固定リストではなく、間違えた方向ほど選ばれやすい重み付き抽選
    - 回答直後に正解方向をもう一度鳴らして、違いを聞き比べさせる
    - 終了条件は問題数ではなく、連続正解数（達成できたら終わり）
    """

    def __init__(self, difficulty: Difficulty = Difficulty.FOUR,
                 distance_m: float = 4.0, target_streak: int = 8,
                 max_trials: int = 60, seed: int | None = None) -> None:
        self.difficulty = difficulty
        self.distance_m = distance_m
        self.target_streak = max(2, target_streak)
        self.max_trials = max(target_streak, max_trials)
        self.n_trials = self.max_trials
        self.seed = seed
        self._rng = random.Random(seed)
        self.trials: list[Trial] = []
        # 方向ごとの重み。間違えるほど大きくなり、正解すると戻っていく
        self._weight: dict[float, float] = {a: 1.0
                                            for a in difficulty.angles}
        self._streak = 0
        self._best_streak = 0
        self._append_trial()

    # QuizSession の固定生成は使わない
    def _build(self) -> None:  # pragma: no cover - 親の __init__ 用
        pass

    def _pick_angle(self) -> float:
        angles = list(self._weight)
        weights = [self._weight[a] for a in angles]
        # 直前と同じ方向は避ける（連続すると記憶で答えられてしまう）
        if self.trials:
            last = self.trials[-1].truth_deg
            if len(angles) > 1:
                idx = angles.index(last)
                weights = list(weights)
                weights[idx] *= 0.15
        return self._rng.choices(angles, weights=weights, k=1)[0]

    def _append_trial(self) -> None:
        self.trials.append(Trial(index=len(self.trials),
                                 truth_deg=self._pick_angle(),
                                 distance_m=self.distance_m))

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def best_streak(self) -> int:
        return self._best_streak

    @property
    def finished(self) -> bool:
        return (self._streak >= self.target_streak
                or len(self.answered_trials) >= self.max_trials)

    @property
    def cleared(self) -> bool:
        """目標の連続正解に到達して終わったか。"""
        return self._streak >= self.target_streak

    def current(self) -> Trial | None:
        if self.finished:
            return None
        return self.trials[-1]

    def answer(self, azimuth_deg: float,
               listener_yaw_deg: float = 0.0) -> Trial:
        trial = self.current()
        if trial is None:
            raise RuntimeError(tr("練習は終了しています",
                                  "The training has finished"))
        trial.answer_deg = (azimuth_deg + listener_yaw_deg) % 360.0

        weight = self._weight[trial.truth_deg]
        if trial.correct:
            self._streak += 1
            self._best_streak = max(self._best_streak, self._streak)
            self._weight[trial.truth_deg] = max(1.0, weight * 0.6)
        else:
            self._streak = 0
            # 前後の取り違えは特に重く見る（今回の課題そのものなので）
            bump = 3.0 if trial.front_back else 2.0
            self._weight[trial.truth_deg] = min(12.0, weight * bump)

        if not self.finished:
            self._append_trial()
        return trial

    def note_replay(self) -> None:
        trial = self.current()
        if trial is not None:
            trial.replays += 1

    def feedback(self, trial: Trial) -> str:
        """回答直後の読み上げ文。"""
        truth = direction_name(trial.truth_deg)
        if trial.correct:
            return tr(f"正解。{truth}です。連続{self._streak}問目。",
                      f"Correct. It was {truth}. {self._streak} in a row.")
        if trial.front_back:
            return tr(f"前後の取り違えです。正解は{truth}。"
                      "もう一度その方向で鳴らします。よく聞いてください。",
                      f"Front and back were confused. The correct answer was "
                      f"{truth}. It will play from there again. Listen "
                      "closely.")
        return tr(f"不正解。正解は{truth}。もう一度その方向で鳴らします。",
                  f"Wrong. The correct answer was {truth}. It will play from there "
                  "again.")

    def summary(self) -> str:
        done = self.answered_trials
        if not done:
            return NO_ANSWERS
        rate = self.n_correct / len(done) * 100
        if self.cleared:
            head = tr("目標の連続正解に到達しました。",
                      "You reached the target streak.")
        else:
            head = tr("回数の上限に達しました。",
                      "You reached the question limit.")
        weak = [direction_name(a) for a, w in sorted(
            self._weight.items(), key=lambda kv: -kv[1]) if w > 1.5][:3]
        if weak:
            tail = tr(f"苦手な方向は{'、'.join(weak)}です。",
                      f"Hard directions: {', '.join(weak)}.")
        else:
            tail = tr("苦手な方向は特にありません。",
                      "No direction was especially hard.")
        return join_sentences(
            head,
            tr(f"{len(done)}問中{self.n_correct}問正解、"
               f"正答率{rate:.0f}パーセント。"
               f"最長連続正解{self._best_streak}問。"
               f"前後の取り違えは{self.n_front_back}問。",
               f"{self.n_correct} of {len(done)} correct, {rate:.0f} percent. "
               f"Longest streak {self._best_streak}. "
               f"Front/back confusions: {self.n_front_back}."),
            tail)

    def detail_lines(self) -> list[str]:
        """親と違い、未回答の最新問題は一覧に出さない（毎回1問先に
        生成しているため、そのままでは常に「未回答」が末尾に並ぶ）。"""
        lines = []
        for t in self.answered_trials:
            extra = FB_EXTRA if t.front_back else ""
            lines.append(_result_line(t, direction_name(t.truth_deg),
                                      direction_name(t.answer_deg), extra))
        return lines


# ---------------------------------------------------------------------------
# 耳介の選定
# ---------------------------------------------------------------------------
@dataclass
class CandidateScore:
    """1 つの耳介（HRTF）の成績。"""

    tag: str
    label: str
    n_answered: int = 0
    n_correct: int = 0
    n_front_back: int = 0
    n_fb_testable: int = 0
    total_error_deg: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_answered if self.n_answered else 0.0

    @property
    def front_back_rate(self) -> float:
        """前後を試せた問題に対する取り違えの割合。

        全問で割ると、真横の出題（前後を試せない）が分母に入って
        取り違え率が実際より低く見える。
        """
        return (self.n_front_back / self.n_fb_testable
                if self.n_fb_testable else 0.0)

    @property
    def mean_error_deg(self) -> float:
        return (self.total_error_deg / self.n_answered
                if self.n_answered else 0.0)

    @property
    def rank_key(self) -> tuple[float, float, float]:
        """並べ替え用。正答率が高く、前後取り違えと誤差が小さいほど良い。"""
        return (-self.accuracy, self.front_back_rate, self.mean_error_deg)

    def describe(self) -> str:
        if self.n_fb_testable:
            fb = tr(f"前後{self.n_front_back}/{self.n_fb_testable}問",
                    f"front/back confusions {self.n_front_back} of "
                    f"{self.n_fb_testable}")
        else:
            fb = tr("前後は検出不可", "front/back not detectable")
        return tr(f"{self.label}: 正答率{self.accuracy * 100:.0f}パーセント "
                  f"({self.n_correct}/{self.n_answered})、{fb}、"
                  f"平均誤差{self.mean_error_deg:.0f}度",
                  f"{self.label}: {self.accuracy * 100:.0f} percent correct "
                  f"({self.n_correct} of {self.n_answered}), {fb}, "
                  f"mean error {self.mean_error_deg:.0f} degrees")


class CalibrationSession(QuizSession):
    """複数の耳介を順に試して、その人に合うものを見つける。

    耳介の形は個人差が大きく、他人のHRTFで前後が分かるかどうかは
    組み合わせ次第で変わる。Viking は耳介が20種類あるので、実際に
    聞き比べて選べる。

    候補ごとにまとまった問数を続けて出す（1問ごとに切り替えると、
    HRTF の差し替えより「直前の音の記憶」の影響が大きくなって比較に
    ならない）。
    """

    def __init__(self, candidates: list[tuple[str, str]],
                 difficulty: Difficulty = Difficulty.FOUR,
                 trials_per_candidate: int = 8,
                 distance_m: float = 4.0,
                 seed: int | None = None) -> None:
        if not candidates:
            raise ValueError(tr("候補が空です", "There are no candidates"))
        self.candidates = candidates
        self.trials_per_candidate = max(2, trials_per_candidate)
        self.difficulty = difficulty
        self.distance_m = distance_m
        self.seed = seed
        self.n_trials = len(candidates) * self.trials_per_candidate
        self._rng = random.Random(seed)
        self.trials = []
        self._build_rounds()

    def _build(self) -> None:  # pragma: no cover - 親の __init__ 用
        pass

    def _build_rounds(self) -> None:
        angles = list(self.difficulty.angles)
        index = 0
        for tag, _label in self.candidates:
            picks: list[float] = []
            while len(picks) < self.trials_per_candidate:
                batch = angles[:]
                self._rng.shuffle(batch)
                if picks and batch[0] == picks[-1] and len(batch) > 1:
                    batch[0], batch[1] = batch[1], batch[0]
                picks.extend(batch)
            for az in picks[:self.trials_per_candidate]:
                self.trials.append(Trial(index=index, truth_deg=az,
                                         distance_m=self.distance_m,
                                         tag=tag))
                index += 1

    @property
    def current_tag(self) -> str | None:
        trial = self.current()
        return trial.tag if trial else None

    def is_round_start(self) -> bool:
        """いまの問題が新しい耳介の 1 問目か。"""
        trial = self.current()
        if trial is None:
            return False
        return trial.index % self.trials_per_candidate == 0

    def scores(self) -> list[CandidateScore]:
        labels = dict(self.candidates)
        table = {tag: CandidateScore(tag, labels[tag])
                 for tag, _ in self.candidates}
        for t in self.answered_trials:
            score = table[t.tag]
            score.n_answered += 1
            score.n_correct += int(t.correct)
            score.n_front_back += int(t.front_back)
            score.n_fb_testable += int(t.front_back_testable)
            score.total_error_deg += t.error_deg
        return sorted(table.values(), key=lambda s: s.rank_key)

    def best(self) -> CandidateScore | None:
        ranked = [s for s in self.scores() if s.n_answered]
        return ranked[0] if ranked else None

    def summary(self) -> str:
        ranked = [s for s in self.scores() if s.n_answered]
        if not ranked:
            return NO_ANSWERS
        top = ranked[0]
        # 同点が並ぶと順位に意味がない。前回の実施では5種類中4種類が
        # 100パーセントで並び、耳介の比較ができなかった。
        tied = [s for s in ranked
                if abs(s.accuracy - top.accuracy) < 1e-9
                and s.n_front_back == top.n_front_back]
        parts: list[str] = []
        if len(tied) > 1:
            names = list_sep().join(s.label for s in tied)
            parts.append(tr(
                f"{len(tied)}種類が同成績で並びました。{names}が"
                f"いずれも正答率{top.accuracy * 100:.0f}パーセントです。"
                "差をつけるには難易度を上げるか、問数を増やしてください。",
                f"{len(tied)} tied with the same result: {names}, all "
                f"{top.accuracy * 100:.0f} percent correct. To tell them "
                "apart, raise the difficulty or ask more questions."))
        else:
            parts.append(tr(f"最も成績が良かったのは{top.label}で、"
                            f"正答率{top.accuracy * 100:.0f}パーセントです。",
                            f"The best was {top.label}, "
                            f"{top.accuracy * 100:.0f} percent correct."))
            worst = ranked[-1]
            if abs(top.accuracy - worst.accuracy) > 1e-9:
                parts.append(tr(f"最も悪かったのは{worst.label}で"
                                f"正答率{worst.accuracy * 100:.0f}パーセントです。",
                                f"The worst was {worst.label}, "
                                f"{worst.accuracy * 100:.0f} percent correct."))

        testable = sum(s.n_fb_testable for s in ranked)
        confused = sum(s.n_front_back for s in ranked)
        if testable == 0:
            parts.append(tr("なお、", "Note: ") + NO_FRONT_BACK)
        elif confused:
            parts.append(tr(f"全体では前後を試せた{testable}問のうち"
                            f"{confused}問で取り違えがありました。",
                            f"Overall, {confused} of the {testable} "
                            "questions that tested front and back were "
                            "confused."))
        else:
            parts.append(tr(f"全体では前後を試せた{testable}問で"
                            "取り違えはありませんでした。",
                            f"Overall, no front/back confusions in the "
                            f"{testable} questions that tested it."))
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        return [s.describe() for s in self.scores() if s.n_answered]


# ---------------------------------------------------------------------------
# 上下（仰角）の聴き取り
# ---------------------------------------------------------------------------
#: 出題する高さ。(仰角, 読み上げ名)。上から順に並べる（テンキーの
#: / 8 5 2 の縦の並びと対応する）。Viking は仰角 -45〜+90 度を測って
#: おり、下は -45 度が限界なので「下90度」は作れない。
#: 2026-10-15 に真上（90度）を足して4択にした。それ以前のログは
#: 上・水平・下の3択（当て推量33%）なので、成績を直接は比べられない。
ELEVATION_LEVELS: tuple[tuple[float, str], ...] = (
    (90.0, tr("真上", "straight up")),
    (45.0, tr("上", "up")),
    (0.0, tr("水平", "level")),
    (-45.0, tr("下", "down")),
)

#: 真上の音。方位を変えても同じ点なので、方位を名乗らない。
ZENITH_DEG = 90.0


def elevation_name(elevation_deg: float) -> str:
    """仰角に最も近い高さの名前を返す。"""
    return min(ELEVATION_LEVELS,
               key=lambda item: abs(item[0] - elevation_deg))[1]


@dataclass
class ElevationTrial(Trial):
    """上下の1問。方位も持つが、答えるのは高さだけ。"""

    truth_el: float = 0.0
    answer_el: float | None = None

    @property
    def answered(self) -> bool:
        return self.answer_el is not None

    @property
    def error_deg(self) -> float:
        if self.answer_el is None:
            return 0.0
        return abs(self.answer_el - self.truth_el)

    @property
    def correct(self) -> bool:
        return self.answered and self.error_deg < 1e-6

    @property
    def front_back(self) -> bool:
        return False

    @property
    def front_back_testable(self) -> bool:
        return False

    @property
    def up_down(self) -> bool:
        """上と下の取り違え（水平を飛び越えて逆に答えた）か。

        前後の取り違えと同じく、耳介の手がかりが合わないときの典型的な
        誤りなので、単なる「1段ずれ」とは分けて数える。
        """
        return (self.answer_el is not None
                and self.truth_el * self.answer_el < 0)


class ElevationSession(QuizSession):
    """上下（仰角）を出題する聴き取りテスト。

    上下の手がかりは耳介によるスペクトルの変化だけで、左右のような
    両耳間の差がほとんど無い。前後と同じく他人の HRTF では外れやすく、
    しかも首振り（水平の回転）は助けにならない。実測の全天球 HRTF が
    要る（合成 HRIR と水平面だけの既定アセットでは原理的に区別できない）。

    方位は難易度の方向から選ぶ。正面だけで出すと、正面の上下という
    1つのスペクトルの組を覚えて答えられてしまうため（正面だけで聞き比べ
    たいときは FrontElevationSession）。回答は高さだけで、方位は問わない。
    当て推量でも正答率は「1÷選択肢の数」になる点を結果に書く。
    """

    #: 完全に同じ音（高さも方位も同じ）が続くのを避けるか
    avoid_same_sound = True

    def __init__(self, difficulty: Difficulty = Difficulty.FOUR,
                 n_trials: int = 24, distance_m: float = 4.0,
                 seed: int | None = None,
                 levels: tuple[tuple[float, str], ...] = ELEVATION_LEVELS
                 ) -> None:
        self.levels = levels
        super().__init__(difficulty=difficulty, n_trials=n_trials,
                         distance_m=distance_m, seed=seed)

    def azimuths(self) -> tuple[float, ...]:
        return tuple(self.difficulty.angles)

    @staticmethod
    def sound_key(el: float, az: float) -> tuple[float, float | None]:
        """聞こえる音の同一性。真上は方位を変えても同じ点。"""
        return (el, None if el >= ZENITH_DEG else az)

    def _build(self) -> None:
        """高さと方位の全組み合わせをシャッフルして並べる。

        組み合わせ単位で一巡させるので、どの高さも同じ回数ずつ出る。
        完全に同じ音が続くと、直前の記憶で答えられるので避ける。真上は
        方位が違っても同じ音なので、一巡の中にも同じ音が複数入る。
        境目だけでなく全体を見て、隣り合ったら後ろの別の音と入れ替える。
        高さだけが続くのは許す（連続を禁じると、直前と違う方を選ぶだけで
        正答率が上がってしまう）。
        """
        combos = [(el, az) for el, _ in self.levels
                  for az in self.azimuths()]
        picks: list[tuple[float, float]] = []
        while len(picks) < self.n_trials:
            batch = combos[:]
            self._rng.shuffle(batch)
            picks.extend(batch)
        picks = picks[:self.n_trials]
        if self.avoid_same_sound:
            self._separate_same_sounds(picks)
        for i, (el, az) in enumerate(picks):
            self.trials.append(ElevationTrial(
                index=i, truth_deg=az, distance_m=self.distance_m,
                truth_el=el))

    def _separate_same_sounds(self, picks: list[tuple[float, float]]) -> None:
        """同じ音の隣り合いを、入れ替えでなくす（その場で書き換える）。

        入れ替えるのは、入れ替えた後にどちらの位置でも新しい隣り合いが
        できないときだけ。前方に相手がいなければ後方も探す。
        """
        key = self.sound_key

        def clashes(seq: list, pos: int) -> bool:
            k = key(*seq[pos])
            return ((pos > 0 and key(*seq[pos - 1]) == k)
                    or (pos + 1 < len(seq) and key(*seq[pos + 1]) == k))

        for i in range(1, len(picks)):
            if key(*picks[i]) != key(*picks[i - 1]):
                continue
            others = list(range(i + 1, len(picks))) + list(range(i - 1))
            for j in others:
                # 入れ替えてみて、どちらの位置にも隣り合いができなければ
                # 確定。できたら元に戻して次の相手を試す
                picks[i], picks[j] = picks[j], picks[i]
                if not clashes(picks, i) and not clashes(picks, j):
                    break
                picks[i], picks[j] = picks[j], picks[i]

    def answer(self, azimuth_deg: float,
               listener_yaw_deg: float = 0.0) -> Trial:
        raise TypeError(tr("上下のテストは answer_elevation() で答えます",
                           "Answer the up/down test with answer_elevation()"))

    def answer_elevation(self, elevation_deg: float) -> ElevationTrial:
        """高さで回答する。向きを変えても仰角は変わらないので補正は不要。"""
        trial = self.current()
        if trial is None:
            raise RuntimeError(ALL_DONE)
        trial.answer_el = float(elevation_deg)
        return trial

    # -- 集計 ---------------------------------------------------------
    @property
    def n_up_down(self) -> int:
        return sum(1 for t in self.answered_trials if t.up_down)

    def level_stats(self) -> list[tuple[str, int, int]]:
        """高さごとの (名前, 出題数, 正解数)。"""
        out = []
        for el, name in self.levels:
            done = [t for t in self.answered_trials if t.truth_el == el]
            out.append((name, len(done), sum(1 for t in done if t.correct)))
        return out

    def summary(self) -> str:
        done = self.answered_trials
        if not done:
            return NO_ANSWERS
        rate = self.n_correct / len(done) * 100
        chance = 100.0 / len(self.levels)
        parts = [tr(f"{len(done)}問中{self.n_correct}問正解、"
                    f"正答率{rate:.0f}パーセント。"
                    f"当て推量でもおよそ{chance:.0f}パーセントになります。",
                    f"{self.n_correct} of {len(done)} correct, {rate:.0f} "
                    f"percent. Guessing would give about {chance:.0f} "
                    "percent.")]
        per = [tr(f"{name}は{n}問中{c}問", f"{name} {c} of {n}")
               for name, n, c in self.level_stats() if n]
        parts.append(tr("高さ別では、" + "、".join(per) + "。",
                        "By height: " + ", ".join(per) + "."))
        if self.n_up_down:
            parts.append(tr(f"上と下の取り違えが{self.n_up_down}問ありました。",
                            f"Up and down were confused {self.n_up_down} "
                            "times."))
        else:
            parts.append(tr("上と下の取り違えはありませんでした。",
                            "Up and down were never confused."))
        if len(done) < 24:
            parts.append(FEW_QUESTIONS)
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        lines = []
        for t in self.trials:
            if not t.answered:
                lines.append(not_answered(t.index + 1))
                continue
            extra = tr("、上下の取り違え", ", up/down confusion") if t.up_down else ""
            # 真上は方位が無意味。正面固定では方位は毎回同じなので書かない
            if t.truth_el >= ZENITH_DEG or len(self.azimuths()) == 1:
                where = ""
            else:
                where = tr(f"（方位は{direction_name(t.truth_deg)}）",
                           f" (direction {direction_name(t.truth_deg)})")
            n = t.index + 1
            lines.append(tr(
                f"{n}問目 {mark_text(t.correct)}。正解は"
                f"{elevation_name(t.truth_el)}{where}、"
                f"回答は{elevation_name(t.answer_el)}{extra}",
                f"Question {n}: {mark_text(t.correct)}. The correct answer was "
                f"{elevation_name(t.truth_el)}{where}, you answered "
                f"{elevation_name(t.answer_el)}{extra}"))
        return lines


class FrontElevationSession(ElevationSession):
    """正面の同じ方位から、高さだけを変えて出題する。

    ElevationSession は方位も混ぜるので、聞こえ方の変化が高さによるものか
    方位によるものか、聞き手には切り分けにくい。こちらは方位を正面に
    固定し、変わるのは高さだけにする。上下の手がかりそのものを聞き比べ
    たり、慣れたりするためのモード。

    同じ高さが続くのも許す。方位が1つなので、同じ高さ＝同じ音になる。
    続けて出ないと分かっていると「直前と違うもの」を選ぶだけで正答率が
    上がり、当て推量の水準（1÷選択肢の数）が崩れる。
    """

    avoid_same_sound = False

    def __init__(self, n_trials: int = 24, distance_m: float = 4.0,
                 seed: int | None = None,
                 levels: tuple[tuple[float, str], ...] = ELEVATION_LEVELS,
                 azimuth_deg: float = 0.0) -> None:
        self.azimuth_deg = float(azimuth_deg)
        super().__init__(difficulty=Difficulty.FOUR, n_trials=n_trials,
                         distance_m=distance_m, seed=seed, levels=levels)

    def azimuths(self) -> tuple[float, ...]:
        return (self.azimuth_deg,)


class ElevationCalibrationSession(ElevationSession):
    """上下に合う耳介を探す。前後用の CalibrationSession の上下版。

    耳介E は前後の聞き分けで選んだもので、上下に合うとは限らない。上下の
    手がかりは耳介のスペクトルそのものなので、前後以上に耳介の差が出る
    可能性がある。

    出題は正面固定（FrontElevationSession と同じ）。耳介ごとに、4つの
    高さを同じ回数ずつまとめて出す（1問ごとに切り替えると、耳介の差より
    「直前の音の記憶」が効いてしまう。前後用と同じ理由）。
    結果は CandidateScore で返すので、前後用と同じ RecordStore に
    （別のファイルで）積み上げられる。前後の取り違えの欄には、上と下の
    取り違えを入れる。
    """

    avoid_same_sound = False

    def __init__(self, candidates: list[tuple[str, str]],
                 trials_per_candidate: int = 12, distance_m: float = 4.0,
                 seed: int | None = None,
                 levels: tuple[tuple[float, str], ...] = ELEVATION_LEVELS
                 ) -> None:
        if not candidates:
            raise ValueError(tr("候補が空です", "There are no candidates"))
        self.candidates = candidates
        self.levels = levels
        # 1つの耳介で全部の高さが同じ回数ずつ出るよう、高さの数の倍数にする
        per = max(len(levels), trials_per_candidate)
        self.trials_per_candidate = per - per % len(levels)
        self.difficulty = Difficulty.FOUR
        self.distance_m = distance_m
        self.seed = seed
        self.n_trials = len(candidates) * self.trials_per_candidate
        self._rng = random.Random(seed)
        self.trials = []
        self._build_rounds()

    def _build(self) -> None:  # pragma: no cover - 親の __init__ 用
        pass

    def azimuths(self) -> tuple[float, ...]:
        return (0.0,)

    def _build_rounds(self) -> None:
        index = 0
        rounds = self.trials_per_candidate // len(self.levels)
        for tag, _label in self.candidates:
            heights = [el for el, _ in self.levels] * rounds
            self._rng.shuffle(heights)
            for el in heights:
                self.trials.append(ElevationTrial(
                    index=index, truth_deg=0.0, distance_m=self.distance_m,
                    tag=tag, truth_el=el))
                index += 1

    @property
    def current_tag(self) -> str | None:
        trial = self.current()
        return trial.tag if trial else None

    def is_round_start(self) -> bool:
        """いまの問題が新しい耳介の 1 問目か。"""
        trial = self.current()
        if trial is None:
            return False
        return trial.index % self.trials_per_candidate == 0

    def scores(self) -> list[CandidateScore]:
        """耳介ごとの成績。前後の取り違えの欄に上と下の取り違えを入れる。"""
        labels = dict(self.candidates)
        table = {tag: CandidateScore(tag, labels[tag])
                 for tag, _ in self.candidates}
        for t in self.answered_trials:
            score = table[t.tag]
            score.n_answered += 1
            score.n_correct += int(t.correct)
            score.n_front_back += int(t.up_down)
            score.n_fb_testable += 1
            score.total_error_deg += t.error_deg
        return sorted(table.values(), key=lambda s: s.rank_key)

    def summary(self) -> str:
        ranked = [s for s in self.scores() if s.n_answered]
        if not ranked:
            return NO_ANSWERS
        chance = 100.0 / len(self.levels)
        top = ranked[0]
        tied = [s for s in ranked
                if abs(s.accuracy - top.accuracy) < 1e-9
                and s.n_front_back == top.n_front_back]
        parts: list[str] = []
        if len(tied) > 1:
            names = list_sep().join(s.label for s in tied)
            parts.append(tr(f"{len(tied)}種類が同成績で並びました。{names}が"
                            f"いずれも正答率{top.accuracy * 100:.0f}パーセントです。",
                            f"{len(tied)} tied with the same result: {names}, "
                            f"all {top.accuracy * 100:.0f} percent correct."))
        else:
            parts.append(tr(f"上下で最も成績が良かったのは{top.label}で、"
                            f"正答率{top.accuracy * 100:.0f}パーセントです。",
                            f"For up and down, the best was {top.label}, "
                            f"{top.accuracy * 100:.0f} percent correct."))
            worst = ranked[-1]
            if abs(top.accuracy - worst.accuracy) > 1e-9:
                parts.append(tr(f"最も悪かったのは{worst.label}で"
                                f"正答率{worst.accuracy * 100:.0f}パーセントです。",
                                f"The worst was {worst.label}, "
                                f"{worst.accuracy * 100:.0f} percent correct."))
        parts.append(tr(f"当て推量でもおよそ{chance:.0f}パーセントになります。",
                        f"Guessing would give about {chance:.0f} percent."))
        parts.append(tr("1つの耳介が"
                        f"{self.trials_per_candidate}問なので、"
                        "1、2問の差は偶然の範囲です。",
                        f"Each pinna had {self.trials_per_candidate} "
                        "questions, so a difference of 1 or 2 may be "
                        "chance."))
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        lines = []
        for s in self.scores():
            if not s.n_answered:
                continue
            steps = s.mean_error_deg / 45.0
            lines.append(tr(
                f"{s.label}: 正答率{s.accuracy * 100:.0f}パーセント "
                f"({s.n_correct}/{s.n_answered})、"
                f"上と下の取り違え{s.n_front_back}問、"
                f"平均のずれ{steps:.1f}段",
                f"{s.label}: {s.accuracy * 100:.0f} percent correct "
                f"({s.n_correct} of {s.n_answered}), up/down confusions "
                f"{s.n_front_back}, mean offset {steps:.1f} steps"))
        return lines


# ---------------------------------------------------------------------------
# 回る向きの聴き取り
# ---------------------------------------------------------------------------
#: 回り方。(キー, 読み上げ名)。左回りは上から見て反時計回り（前→左→後ろ→右）
ROTATION_MOTIONS: tuple[tuple[str, str], ...] = (
    ("left", tr("左回り", "counterclockwise")),
    ("right", tr("右回り", "clockwise")),
    ("to_front", tr("頭上を後ろから前へ", "overhead, back to front")),
    ("to_back", tr("頭上を前から後ろへ", "overhead, front to back")),
)

#: 逆向きの組。前後を取り違えると、回る向きが逆に聞こえる
ROTATION_REVERSE = {"left": "right", "right": "left",
                    "to_front": "to_back", "to_back": "to_front"}

#: 半周にかける秒数（180度を3秒、毎秒60度）。速すぎると前後の音色の変化を
#: 追えず、遅すぎると1問が長くなって集中が続かない
ROTATION_ARC_SECONDS = 3.0


def rotation_name(motion: str) -> str:
    return dict(ROTATION_MOTIONS)[motion]


def rotation_kind(motion: str) -> str:
    """水平の半周なら "horizontal"、頭上を越える半周なら "overhead"。"""
    return "horizontal" if motion in ("left", "right") else "overhead"


@dataclass
class RotationTrial(Trial):
    """回る向きの1問。

    truth_deg は回り始める方位（聞き手から見て）。水平は8方向のどれか、
    頭上は前（0度、前から後ろへ）か後ろ（180度、後ろから前へ）。
    """

    truth_motion: str = "left"
    answer_motion: str | None = None

    @property
    def answered(self) -> bool:
        return self.answer_motion is not None

    @property
    def error_deg(self) -> float:
        return 0.0

    @property
    def correct(self) -> bool:
        return self.answered and self.answer_motion == self.truth_motion

    @property
    def front_back(self) -> bool:
        return False

    @property
    def front_back_testable(self) -> bool:
        return False

    @property
    def reversed(self) -> bool:
        """回る向きを逆に答えたか。前後の取り違えと同じ誤り。"""
        return (self.answer_motion is not None
                and self.answer_motion == ROTATION_REVERSE[self.truth_motion])

    @property
    def kind(self) -> str:
        return rotation_kind(self.truth_motion)

    def position(self, fraction: float, yaw_deg: float = 0.0
                 ) -> tuple[float, float, float]:
        """回り始めからの割合 fraction（0〜1）での音源の位置。

        `yaw_deg` は聞き手のいまの向き。前後と左右は聞き手から見たものなので、
        向きを変えていても、その人の前から回り始める。
        """
        f = min(1.0, max(0.0, float(fraction)))
        d = self.distance_m
        if self.kind == "horizontal":
            sign = 1.0 if self.truth_motion == "left" else -1.0
            return position_for(self.truth_deg + sign * 180.0 * f + yaw_deg, d)
        # 頭上: 前（0度）から真上（90度）を通って後ろ（180度）へ。聞き手の
        # 正面を通る縦の円の上を動く
        phi = 180.0 * f if self.truth_motion == "to_back" else 180.0 * (1 - f)
        p = math.radians(phi)
        y = math.radians(yaw_deg)
        h = d * math.cos(p)                  # 正面方向の成分（後ろなら負）
        return (-h * math.sin(y), h * math.cos(y), d * math.sin(p))


class RotationSession(QuizSession):
    """音が自分の周りを回る向きを当てる聴き取りテスト。

    音を正面だけから回し始めると、最初に左右どちらへ動くかで答えが分かる。
    だから回り始めの方位を8方向からばらばらに選び、半周させる。前と後ろを
    取り違えると回る向きが逆に聞こえるので、前後の聞き分けそのものを試す
    問題になる。

    頭上の半周は、前から真上を通って後ろへ、または逆。下45度より下は
    実測データが無いので、下を通る縦の一周は作れない。水平か頭上かは
    聞けば分かるので、当て推量の正答率は 50%（同じ種類の2択）。
    """

    #: 水平の回り始め（聞き手から見た方位、反時計回り）
    HORIZONTAL_STARTS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

    def __init__(self, n_trials: int = 24, distance_m: float = 4.0,
                 seed: int | None = None, include_overhead: bool = True,
                 overhead_share: float = 1 / 3) -> None:
        self.include_overhead = bool(include_overhead)
        self.overhead_share = float(overhead_share)
        super().__init__(difficulty=Difficulty.EIGHT, n_trials=n_trials,
                         distance_m=distance_m, seed=seed)

    def _build(self) -> None:
        """水平の16通り（8方位×2向き）と頭上の2通りを、割合どおりに並べる。

        水平は16通りを一巡ずつ、頭上は2通りを同じ数ずつ足す。全く同じ音が
        続くと直前の記憶で答えられるので、隣り合ったら入れ替える。
        """
        horizontal = [(m, s) for s in self.HORIZONTAL_STARTS
                      for m in ("left", "right")]
        overhead = [("to_back", 0.0), ("to_front", 180.0)]
        n_over = (int(round(self.n_trials * self.overhead_share))
                  if self.include_overhead else 0)
        n_horiz = self.n_trials - n_over
        picks: list[tuple[str, float]] = []
        while len(picks) < n_horiz:
            batch = horizontal[:]
            self._rng.shuffle(batch)
            picks.extend(batch)
        picks = picks[:n_horiz]
        picks.extend(overhead[i % 2] for i in range(n_over))
        self._rng.shuffle(picks)
        self._separate_same(picks)
        for i, (motion, start) in enumerate(picks):
            self.trials.append(RotationTrial(
                index=i, truth_deg=start, distance_m=self.distance_m,
                truth_motion=motion))

    @staticmethod
    def _separate_same(picks: list[tuple[str, float]]) -> None:
        """同じ音の隣り合いを入れ替えでなくす（ElevationSession と同じ方針）。"""
        def clashes(pos: int) -> bool:
            return ((pos > 0 and picks[pos - 1] == picks[pos])
                    or (pos + 1 < len(picks) and picks[pos + 1] == picks[pos]))

        for i in range(1, len(picks)):
            if picks[i] != picks[i - 1]:
                continue
            for j in list(range(i + 1, len(picks))) + list(range(i - 1)):
                picks[i], picks[j] = picks[j], picks[i]
                if not clashes(i) and not clashes(j):
                    break
                picks[i], picks[j] = picks[j], picks[i]

    def answer(self, azimuth_deg: float,
               listener_yaw_deg: float = 0.0) -> Trial:
        raise TypeError(tr("回る向きのテストは answer_rotation() で答えます",
                           "Answer the rotation test with answer_rotation()"))

    def answer_rotation(self, motion: str) -> RotationTrial:
        """回り方で回答する。左回り・右回りは向きを変えても変わらない。"""
        if motion not in ROTATION_REVERSE:
            raise ValueError(f"unknown motion {motion!r}")
        trial = self.current()
        if trial is None:
            raise RuntimeError(ALL_DONE)
        trial.answer_motion = motion
        return trial

    # -- 集計 ---------------------------------------------------------
    @property
    def n_reversed(self) -> int:
        return sum(1 for t in self.answered_trials if t.reversed)

    def kind_stats(self) -> list[tuple[str, int, int, int]]:
        """種類ごとの (名前, 出題数, 正解数, 逆向き)。"""
        out = []
        for kind, name in (("horizontal", tr("水平", "horizontal")),
                           ("overhead", tr("頭上", "overhead"))):
            done = [t for t in self.answered_trials if t.kind == kind]
            out.append((name, len(done), sum(1 for t in done if t.correct),
                        sum(1 for t in done if t.reversed)))
        return out

    def summary(self) -> str:
        done = self.answered_trials
        if not done:
            return NO_ANSWERS
        rate = self.n_correct / len(done) * 100
        parts = [tr(f"{len(done)}問中{self.n_correct}問正解、"
                    f"正答率{rate:.0f}パーセント。"
                    "水平か頭上かは聞けば分かるので、当て推量でもおよそ"
                    "50パーセントになります。",
                    f"{self.n_correct} of {len(done)} correct, {rate:.0f} "
                    "percent. You can hear whether it is horizontal or "
                    "overhead, so guessing would give about 50 percent.")]
        per = [tr(f"{name}は{n}問中{c}問", f"{name} {c} of {n}")
               for name, n, c, _ in self.kind_stats() if n]
        parts.append(tr("種類別では、" + "、".join(per) + "。",
                        "By kind: " + ", ".join(per) + "."))
        if self.n_reversed:
            parts.append(tr(f"回る向きを逆に答えたのが{self.n_reversed}問"
                            "ありました。前と後ろを取り違えると逆に聞こえます。",
                            f"{self.n_reversed} answers had the direction "
                            "reversed. Confusing front and back makes it "
                            "sound reversed."))
        else:
            parts.append(tr("回る向きを逆に答えた問題はありませんでした。",
                            "No answer had the direction reversed."))
        if len(done) < 24:
            parts.append(FEW_QUESTIONS)
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        lines = []
        for t in self.trials:
            if not t.answered:
                lines.append(not_answered(t.index + 1))
                continue
            if t.reversed:
                extra = tr("、回る向きの取り違え", ", direction reversed")
            else:
                extra = ""
            if t.kind == "horizontal":
                where = tr(f"（{direction_name(t.truth_deg)}から）",
                           f" (from {direction_name(t.truth_deg)})")
            else:
                where = ""
            n = t.index + 1
            lines.append(tr(
                f"{n}問目 {mark_text(t.correct)}。正解は"
                f"{rotation_name(t.truth_motion)}{where}、"
                f"回答は{rotation_name(t.answer_motion)}{extra}",
                f"Question {n}: {mark_text(t.correct)}. The correct answer was "
                f"{rotation_name(t.truth_motion)}{where}, you answered "
                f"{rotation_name(t.answer_motion)}{extra}"))
        return lines
