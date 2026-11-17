"""ゲームの中で、自分に合う耳介（実測 HRTF）を選ぶ。

耳介の形は人によって大きく違い、他人の耳介の HRTF では前後や上下を
取り違えやすい。audiospace の既定（耳介 E）は作者の聴き取りで選んだもので、
誰にでも合うとは限らない。Viking HRTF には耳介が20種類（と、耳介なし）あり、
実際に聞き比べて選べる。ここは、その聞き比べをゲームの画面に組み込むための
まとめ役。

    from audiospace import Difficulty, PinnaSelection, RecordStore
    from audiospace.pinna import viking_candidates

    candidates = viking_candidates("assets/hrtf/viking")   # [(タグ, 表示名)]
    store = RecordStore(settings_dir / "pinna_records.json")
    selection = PinnaSelection(player, candidates, store,
                               difficulty=Difficulty.EIGHT)
    speak(selection.intro())
    while not selection.finished:
        if selection.prepare():                 # 次の耳介に切り替えた
            speak(selection.round_text())
        selection.play(sound)                   # 出題の音
        selection.answer(wait_for_answer())     # 答えた方位（テンキーなど）
    speak(selection.finish())                   # 通算に積んで保存、元に戻す
    best = selection.best()                     # 通算の最良 (タグ, 表示名)

1回で試すのは数種類（現在の最良を基準として毎回入れ、残りを新顔で埋める）。
聞き手は回を重ねるごとに上達するので、同じ回の中で基準と比べた差だけで
順位を付ける（`audiospace.records`）。何回かに分けて全種類を試せる。

耳介のデータ（SOFA、1つ約2.7MB）は wheel に入っていない。配布の zip の
assets/hrtf/viking にある（取り直すなら tools/fetch_viking_hrtf.py）。SOFA を
読むには sofar が要る。npz（`tools/bake_hrtf_asset.py` で焼いたもの）も使える。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .hrtf import HrirSet
from .i18n import join_sentences, tr
from .quiz import (ALL_DONE, ELEVATION_LEVELS, CalibrationSession, Difficulty,
                   ElevationCalibrationSession, Trial, position_for)
from .records import MIN_DIRECTIONS_FOR_RANKING, RecordStore, _tag_key

#: 選び方。前後の聞き分け（方位）と、上下の聞き分け（正面の高さ）
MODE_FRONT_BACK = "front_back"
MODE_ELEVATION = "elevation"


def pinna_label(path: str | Path) -> str:
    """Viking の耳介のファイルの表示名（「実測 Viking 耳介E」など）。"""
    letter = Path(path).stem.replace("subj_", "")
    if letter == "Z":
        return tr("実測 Viking 耳介なし", "Measured Viking without pinna")
    return tr(f"実測 Viking 耳介{letter}", f"Measured Viking pinna {letter}")


def viking_candidates(folder: str | Path) -> list[tuple[str, str]]:
    """フォルダの Viking の耳介を、(タグ, 表示名) の一覧で返す。

    タグはファイルのパス（`load_pinna` に渡せる）。subj_A〜subj_T と、
    耳介なしの subj_Z を、名前の順に並べる。同じ耳介の .npz と .sofa が
    あれば .npz を使う（sofar が要らず、読むのも速い）。フォルダが無ければ
    FileNotFoundError（パスの書き違いに気づけるように）。
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(tr(
            f"耳介のフォルダがありません: {folder}",
            f"The pinna folder does not exist: {folder}"))
    found: dict[str, Path] = {}
    for path in sorted(folder.glob("subj_*.sofa")):
        found[path.stem] = path
    for path in sorted(folder.glob("subj_*.npz")):
        found[path.stem] = path
    return [(str(found[stem]), pinna_label(found[stem]))
            for stem in sorted(found)]


def load_pinna(tag: str | Path, sample_rate: int) -> HrirSet:
    """耳介のデータ（.sofa か .npz）を読み、レートを合わせて返す。"""
    path = Path(tag)
    if path.suffix.lower() == ".npz":
        return HrirSet.load_npz(path).resampled(sample_rate)
    return HrirSet.from_sofa(path, sample_rate=sample_rate)


def best_pinna(store: RecordStore, candidates: list[tuple[str, str]]
               ) -> tuple[str, str] | None:
    """通算の最良の耳介を、候補の (タグ, 表示名) で返す（まだ無ければ None）。

    記録の中のタグはファイル名だけなので、渡した候補から元のパスを探す。
    """
    champion = store.champion()
    if champion is None:
        return None
    for tag, label in candidates:
        if _tag_key(tag) == champion.tag:
            return tag, label
    return None


def _has_elevation(hrir: HrirSet) -> bool:
    return float(np.max(hrir.elevation) - np.min(hrir.elevation)) > 20.0


class PinnaSelection:
    """耳介の聞き比べを、HRTF の切り替えと通算の記録まで含めて進める。

    出題と採点は `CalibrationSession`（前後）か
    `ElevationCalibrationSession`（上下）、通算は `RecordStore` が受け持つ。
    ここが足すのは、顔ぶれの決め方（基準の同席）、問題の数、耳介の切り替え、
    出題の音の位置、終わったときの記録と元の HRTF への戻し。

    mode                 MODE_FRONT_BACK（既定、方位で答える）か
                         MODE_ELEVATION（正面の高さで答える。全天球の
                         耳介のデータが要る）。記録は別々のファイルに積む
    difficulty           前後のとき、出題する方向（8方向以上でないと前後の
                         取り違えを数えられない）
    trials_per_candidate 1つの耳介の問題数。既定は前後なら方向の数の2倍
                         （最低8問）、上下なら12問（4つの高さを3回ずつ）
    slots                1回で比べる耳介の数。既定は合計がおよそ48問に
                         なる数（2〜5）
    preload              True なら、作るときに顔ぶれの耳介を全部読む
                         （途中の切り替えの待ちが無くなる。1つ数百ミリ秒）

    出題の音は、聞き手から見た方向に、聞き手に付いて動く位置
    （`relative=True`）で鳴らす。答えは「いまの自分から見た方向」で、
    向きを変えていたら `listener_yaw_deg` に今の向きを渡す。
    """

    def __init__(self, player, candidates: list[tuple[str, str]],
                 store: RecordStore, *, mode: str = MODE_FRONT_BACK,
                 difficulty: Difficulty = Difficulty.EIGHT,
                 trials_per_candidate: int | None = None,
                 slots: int | None = None, distance_m: float = 4.0,
                 seed: int | None = None, preload: bool = False,
                 loader: Callable[[str, int], HrirSet] | None = None) -> None:
        if mode not in (MODE_FRONT_BACK, MODE_ELEVATION):
            raise ValueError(f"mode must be {MODE_FRONT_BACK!r} or "
                             f"{MODE_ELEVATION!r}, got {mode!r}")
        candidates = [(str(tag), str(label)) for tag, label in candidates]
        if len(candidates) < 2:
            raise ValueError(tr(
                "耳介の候補が2つ以上要ります（基準と、比べる相手）",
                "At least 2 pinna candidates are needed (a reference and "
                "one to compare)"))
        self.player = player
        self.engine = player.engine
        self.candidates = candidates
        self.store = store
        self.mode = mode
        self.difficulty = difficulty
        if mode == MODE_FRONT_BACK:
            per = trials_per_candidate or max(8, len(difficulty.angles) * 2)
        else:
            per = trials_per_candidate or len(ELEVATION_LEVELS) * 3
        if slots is None:
            slots = max(2, min(5, 48 // per))
        self.lineup, self.anchor = store.build_lineup(candidates, slots)
        if mode == MODE_FRONT_BACK:
            self.session = CalibrationSession(
                self.lineup, difficulty=difficulty, trials_per_candidate=per,
                distance_m=distance_m, seed=seed)
        else:
            self.session = ElevationCalibrationSession(
                self.lineup, trials_per_candidate=per, distance_m=distance_m,
                seed=seed)
        self._loader = loader or load_pinna
        self._cache: dict[str, HrirSet] = {}
        self._original = self.engine.hrir
        self._loaded: str | None = None
        self._handle: int | None = None
        self.done = False
        if preload:
            for tag, _ in self.lineup:
                self._load(tag)

    # -- 進行 -----------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.session.finished

    def current(self) -> Trial | None:
        return self.session.current()

    @property
    def current_label(self) -> str:
        """いまの問題の耳介の表示名（終わっていれば空）。"""
        tag = self.session.current_tag
        return dict(self.lineup).get(tag, "") if tag else ""

    def _load(self, tag: str) -> HrirSet:
        hrir = self._cache.get(tag)
        if hrir is None:
            hrir = self._loader(tag, self.engine.sample_rate)
            if self.mode == MODE_ELEVATION and not _has_elevation(hrir):
                raise ValueError(tr(
                    f"{Path(tag).name} には上下の測定がありません。上下の"
                    "選定には全天球のデータ（SOFA）が要ります",
                    f"{Path(tag).name} has no measurements above or below. "
                    "Choosing for up and down needs full-sphere data "
                    "(SOFA)"))
            self._cache[tag] = hrir
        return hrir

    def prepare(self) -> bool:
        """いまの問題の耳介に切り替える。切り替えたら True。

        新しい耳介の1問目の前に呼ぶ（`play` も呼ぶ）。読み込みに数百
        ミリ秒かかることがあるので、切り替えたことを読み上げる間などに
        呼ぶとよい（`preload=True` なら待ちは無い）。
        """
        tag = self.session.current_tag
        if tag is None or tag == self._loaded:
            return False
        self.engine.set_hrir(self._load(tag))
        self._loaded = tag
        return True

    def play(self, sound, **play_kwargs) -> int:
        """いまの問題の音を、その方向（と高さ）で鳴らす。ハンドルを返す。

        `play_kwargs` は `SpatialPlayer.play` にそのまま渡す（`gain`、
        `reverb_send` など）。position と relative はここで決める。
        """
        trial = self.session.current()
        if trial is None:
            raise RuntimeError(ALL_DONE)
        # 聞き直しで重ならないよう、前の出題の音は止める
        self.stop_sound()
        self.prepare()
        elevation = getattr(trial, "truth_el", 0.0)
        position = position_for(trial.truth_deg, trial.distance_m, elevation)
        self._handle = self.player.play(sound, position=position,
                                        relative=True, **play_kwargs)
        return self._handle

    def stop_sound(self) -> None:
        """鳴っている出題の音を止める（鳴っていなければ何もしない）。"""
        if self._handle is not None:
            self.player.stop_sound(self._handle)
            self._handle = None

    def answer(self, azimuth_deg: float,
               listener_yaw_deg: float = 0.0) -> Trial:
        """方位で答える（前後の選定）。いまの自分から見た方向で渡す。"""
        return self.session.answer(azimuth_deg, listener_yaw_deg)

    def answer_elevation(self, elevation_deg: float) -> Trial:
        """高さで答える（上下の選定）。ELEVATION_LEVELS の仰角で渡す。"""
        return self.session.answer_elevation(elevation_deg)

    def note_replay(self) -> None:
        """聞き直したことを数える（結果の記録に残る）。"""
        self.session.note_replay()

    # -- 読み上げ -------------------------------------------------------
    def intro(self) -> str:
        """始めるときの説明（何種類を何問ずつ、基準、進み具合）。"""
        per = self.session.trials_per_candidate
        parts = [tr(f"耳介の選定を始めます。{len(self.lineup)}種類を"
                    f"それぞれ{per}問ずつ、全{self.session.n_trials}問。",
                    f"Starting to choose a pinna. {len(self.lineup)} "
                    f"pinnae, {per} questions each, "
                    f"{self.session.n_trials} in all.")]
        if self.anchor:
            name = dict(self.lineup)[self.anchor]
            parts.append(tr(f"うち{name}は現在の最良で、比較の基準として"
                            "毎回入れています。",
                            f"{name} is the current best and is included "
                            "every time as the reference."))
        parts.append(self.store.progress_text(self.candidates))
        if self.mode == MODE_FRONT_BACK and len(self.difficulty.angles) <= 4:
            parts.append(tr("なお、前後左右の4方向では前後の取り違えを検出"
                            "できません。前後を試すなら8方向以上にします。",
                            "Note: with 4 directions, front/back confusions "
                            "cannot be detected. Use 8 directions or more "
                            "to test front and back."))
        if self.mode == MODE_ELEVATION:
            parts.append(tr("音は毎回正面から鳴り、変わるのは高さだけです。",
                            "The sound always comes from the front; only "
                            "the height changes."))
        return join_sentences(*parts)

    def round_text(self) -> str:
        """新しい耳介に切り替えたときの読み上げ（何番目の耳介か）。"""
        trial = self.session.current()
        if trial is None:
            return ""
        number = trial.index // self.session.trials_per_candidate + 1
        return tr(f"{len(self.lineup)}種類のうち{number}番目の耳介です。",
                  f"Pinna {number} of {len(self.lineup)}.")

    # -- 終わり ---------------------------------------------------------
    def finish(self, save: bool = True) -> str:
        """結果を通算に積み（`save` なら保存も）、元の HRTF に戻す。

        読み上げ用の結果（この回の順位と、通算の最良）を返す。全問を
        終える前に呼ぶと RuntimeError（途中でやめるときは `cancel`）。
        保存に失敗したときは OSError がそのまま出る（積んだ分は
        `store` に残るので、もう一度 `store.save()` を試せる）。
        """
        if not self.session.finished:
            raise RuntimeError(tr(
                "まだ全問が終わっていません。途中でやめるときは cancel() を"
                "呼びます",
                "Not all questions are answered yet. To stop early, call "
                "cancel()"))
        self.restore()
        if not self.done:
            n_directions = (len(self.difficulty.angles)
                            if self.mode == MODE_FRONT_BACK
                            # 上下の4択は、全員が満点に張り付くほど易しくない
                            # ので、判別力のある回として数える
                            else MIN_DIRECTIONS_FOR_RANKING)
            self.store.update(self.session.scores(), self.anchor,
                              n_directions=n_directions)
            self.done = True
            if save:
                self.store.save()
        return join_sentences(self.session.summary(),
                              self.store.summary(self.candidates))

    def cancel(self) -> None:
        """途中でやめる。記録には何も積まず、元の HRTF に戻す。"""
        self.done = True
        self.restore()

    def restore(self) -> None:
        """出題の音を止め、選定を始める前の HRTF に戻す。"""
        self.stop_sound()
        if self._loaded is not None:
            self.engine.set_hrir(self._original)
            self._loaded = None

    def best(self) -> tuple[str, str] | None:
        """通算の最良の耳介 (タグ, 表示名)。`load_pinna` で読んで使う。"""
        return best_pinna(self.store, self.candidates)
