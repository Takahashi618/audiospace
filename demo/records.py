"""耳介ごとの戦績を蓄積して、セッションをまたいで比較する。

なぜ必要か
----------
耳介は21種類あるが、1回のセッションで全部試すと300問を超えて現実的で
ない。数種類ずつ何回かに分けることになる。

ところがそこには交絡がある。**聞き手は回を重ねるごとに上達する**。
実際、初回は前後の取り違えがあったが、慣れると分かるようになった。
この状態で1回目に試した耳介と5回目に試した耳介を単純比較すると、
後から試した耳介が不当に有利になる。

対策として、毎回**チャンピオン（現在の最良）を基準として同席させ**、
同一セッション内での差だけを評価に使う。上達はチャンピオンの成績にも
等しく効くので、差を取れば打ち消される。

保存先: logs/pinna_records.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from audiospace.i18n import join_sentences, tr  # noqa: E402

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "logs" / "pinna_records.json"
#: 取り違えの欄の呼び名（前後用の記録）
FB_NAME = tr("前後", "front/back")
#: これ未満の試行数では順位を主張しない
MIN_TRIALS_FOR_RANKING = 12


def _tag_key(tag: str) -> str:
    """タグをファイル名に正規化する。

    以前は SOFA の絶対パスをそのままタグにしていたため、フォルダを
    移動したり別の PC から開いたりすると、積み上げた
    通算成績が全部「別の耳介」扱いになった。ファイル名だけなら場所が
    変わっても同じ耳介として積み上がる。旧記録は読み込み時に移行する。
    """
    return Path(str(tag)).name
#: 順位付けに使える最低の方向数。
#: 4方向（前後左右）は易しすぎて全員が満点に張り付き、差が出ない。
#: 実際、4方向の回では5種類中4種類が100パーセントで並んだ。
#: さらに真横は前後の鏡像が自分自身なので、前後の検証にもならない。
MIN_DIRECTIONS_FOR_RANKING = 8


@dataclass
class PinnaRecord:
    """1 つの耳介の通算成績。

    集計を2系統持つ。通算（表示用）と、判別力のある難易度に限った分
    （順位用）。易しい回の満点を混ぜると、難しい回で本当に良かった
    耳介が負けてしまう。
    """

    tag: str
    label: str
    n_answered: int = 0
    n_correct: int = 0
    n_fb_testable: int = 0
    n_fb_confusions: int = 0
    total_error_deg: float = 0.0
    n_sessions: int = 0
    #: 同一セッション内でのチャンピオンとの正答率差。上達の影響を除くため
    #: 生の正答率ではなくこちらを順位の主指標にする。
    relative_deltas: list[float] = field(default_factory=list)
    #: 上と対になる、そのセッションの方向数。易しい回を除外するのに使う。
    delta_directions: list[int] = field(default_factory=list)
    #: 判別力のある難易度に限った試行数
    n_answered_hard: int = 0
    n_correct_hard: int = 0
    last_seen: str = ""

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_answered if self.n_answered else 0.0

    @property
    def accuracy_hard(self) -> float:
        return (self.n_correct_hard / self.n_answered_hard
                if self.n_answered_hard else 0.0)

    @property
    def mean_error_deg(self) -> float:
        return (self.total_error_deg / self.n_answered
                if self.n_answered else 0.0)

    @property
    def fb_rate(self) -> float:
        return (self.n_fb_confusions / self.n_fb_testable
                if self.n_fb_testable else 0.0)

    @property
    def hard_deltas(self) -> list[float]:
        """判別力のある難易度で得られた差だけを返す。"""
        if len(self.delta_directions) != len(self.relative_deltas):
            # 旧形式のデータ。方向数が不明なので順位には使わない。
            return []
        return [d for d, n in zip(self.relative_deltas, self.delta_directions)
                if n >= MIN_DIRECTIONS_FOR_RANKING]

    @property
    def mean_delta(self) -> float:
        """チャンピオンとの正答率差の平均。基準自身は 0 になる。"""
        deltas = self.hard_deltas
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    @property
    def ranked(self) -> bool:
        """順位を主張してよいだけの材料があるか。

        易しい回だけで判断しないよう、判別力のある難易度での試行数を見る。
        """
        return self.n_answered_hard >= MIN_TRIALS_FOR_RANKING

    @property
    def rank_key(self) -> tuple[float, float, float]:
        """差が大きく、前後取り違えと誤差が小さいほど良い。"""
        return (-self.mean_delta, self.fb_rate, self.mean_error_deg)

    def describe(self, fb_name: str | None = None) -> str:
        """読み上げ用の1行。`fb_name` は取り違えの欄の呼び名。

        上下用の記録（ElevationCalibrationSession）は、この欄に上と下の
        取り違えを入れて積み上げるので「上下」と呼ぶ。
        """
        if fb_name is None:
            fb_name = FB_NAME
        delta = self.mean_delta * 100
        if self.n_fb_testable:
            fb = tr(f"{fb_name}{self.n_fb_confusions}/{self.n_fb_testable}問",
                    f"{fb_name} confusions {self.n_fb_confusions} of "
                    f"{self.n_fb_testable}")
        else:
            fb = tr(f"{fb_name}は未検証", f"{fb_name} not tested")
        hard = ""
        if self.n_answered_hard and self.n_answered_hard != self.n_answered:
            hard = tr(f"（うち判別可能な難易度は"
                      f"{self.n_correct_hard}/{self.n_answered_hard}問、"
                      f"{self.accuracy_hard * 100:.0f}パーセント）",
                      f" (at difficulties that can tell them apart: "
                      f"{self.n_correct_hard} of {self.n_answered_hard}, "
                      f"{self.accuracy_hard * 100:.0f} percent)")
        sign = tr("＋", "+") if delta > 0 else ""
        return tr(f"{self.label}: 通算{self.n_correct}/{self.n_answered}問 "
                  f"({self.accuracy * 100:.0f}パーセント){hard}、"
                  f"基準との差{sign}{delta:.0f}ポイント、{fb}、"
                  f"平均誤差{self.mean_error_deg:.0f}度",
                  f"{self.label}: total {self.n_correct} of "
                  f"{self.n_answered} ({self.accuracy * 100:.0f} percent)"
                  f"{hard}, difference from the reference {sign}{delta:.0f} "
                  f"points, {fb}, mean error "
                  f"{self.mean_error_deg:.0f} degrees")


def _merge_records(a: PinnaRecord, b: PinnaRecord) -> PinnaRecord:
    """同じ耳介の記録を1つに畳む（旧形式からの移行用）。

    counters は加算、差のリストは連結する。片方が旧形式で方向数を
    持っていない場合、長さが揃わなくなるが、その場合 `hard_deltas` は
    空を返す設計なので、順位に不正な差が混ざることはない。
    """
    out = PinnaRecord(tag=a.tag, label=b.label or a.label)
    for name in ("n_answered", "n_correct", "n_fb_testable",
                 "n_fb_confusions", "n_sessions",
                 "n_answered_hard", "n_correct_hard"):
        setattr(out, name, getattr(a, name) + getattr(b, name))
    out.total_error_deg = a.total_error_deg + b.total_error_deg
    out.relative_deltas = list(a.relative_deltas) + list(b.relative_deltas)
    out.delta_directions = (list(a.delta_directions)
                            + list(b.delta_directions))
    out.last_seen = max(a.last_seen, b.last_seen)
    return out


class RecordStore:
    """戦績の読み書きと順位付け。"""

    def __init__(self, path: Path | None = None,
                 fb_name: str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.fb_name = fb_name if fb_name is not None else FB_NAME
        self.records: dict[str, PinnaRecord] = {}
        #: 読めなかったときの説明（空なら問題なし）。アプリは起動時に「状況」の
        #: 欄の先頭に書く
        self.load_error = ""
        self.load()

    # -- 永続化 -------------------------------------------------------
    def _set_aside(self, why) -> None:
        """読めないファイルを .broken に移して残す（消すと積み上げた成績が
        失われる。次の保存で上書きされるのも防ぐ）。"""
        broken = self.path.with_suffix(self.path.suffix + ".broken")
        try:
            os.replace(self.path, broken)
        except OSError:
            pass
        self.load_error = tr(
            f"耳介の通算成績のファイルが読めないため、{broken.name} に移して"
            f"空から始めます（{why}）",
            f"The pinna totals file cannot be read, so it was moved to "
            f"{broken.name} and you start with none ({why}).")

    def load(self) -> None:
        self.load_error = ""
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = list(raw.get("records", []))
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            self._set_aside(exc)
            return
        for item in items:
            try:
                rec = PinnaRecord(**item)
            except TypeError:
                continue
            # 旧形式（絶対パスのタグ）をファイル名へ移行する。同じ耳介が
            # 旧新の両形式で並んでいたら1つに畳む。
            rec.tag = _tag_key(rec.tag)
            known = self.records.get(rec.tag)
            self.records[rec.tag] = (rec if known is None
                                     else _merge_records(known, rec))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "note": ("順位の主指標は relative_deltas の平均。"
                     "同一セッション内でのチャンピオンとの正答率差で、"
                     "聞き手の上達による交絡を除くためにこうしている。"),
            "records": [asdict(r) for r in self.records.values()],
        }
        # 一時ファイルに書いてから置き換える（途中で落ちても元が壊れない）
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # -- 更新 ---------------------------------------------------------
    def update(self, scores, anchor_tag: str | None = None,
               n_directions: int = 0) -> None:
        """CalibrationSession.scores() の結果を取り込む。

        `anchor_tag` を指定すると、その耳介の正答率を基準にした差を
        各耳介に記録する。基準自身の差は 0。

        `n_directions` はそのセッションの方向数。4方向のような易しい回は
        全員が満点に張り付いて差が出ないため、順位付けからは除外する。

        タグはパスでもファイル名でもよく、内部ではファイル名に正規化して
        積み上げる（`_tag_key` 参照）。
        """
        by_tag = {_tag_key(s.tag): s for s in scores if s.n_answered}
        anchor_key = _tag_key(anchor_tag) if anchor_tag is not None else None
        base = (by_tag[anchor_key].accuracy
                if anchor_key in by_tag else None)
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        hard = n_directions >= MIN_DIRECTIONS_FOR_RANKING

        for tag, s in by_tag.items():
            rec = self.records.get(tag)
            if rec is None:
                rec = PinnaRecord(tag=tag, label=s.label)
                self.records[tag] = rec
            rec.label = s.label
            rec.n_answered += s.n_answered
            rec.n_correct += s.n_correct
            rec.n_fb_testable += s.n_fb_testable
            rec.n_fb_confusions += s.n_front_back
            rec.total_error_deg += s.total_error_deg
            rec.n_sessions += 1
            rec.last_seen = stamp
            if hard:
                rec.n_answered_hard += s.n_answered
                rec.n_correct_hard += s.n_correct
            if base is not None:
                rec.relative_deltas.append(s.accuracy - base)
                rec.delta_directions.append(int(n_directions))

    # -- 参照 ---------------------------------------------------------
    def ranking(self) -> list[PinnaRecord]:
        """順位付けできるだけの試行数がある耳介を、良い順に。"""
        return sorted((r for r in self.records.values() if r.ranked),
                      key=lambda r: r.rank_key)

    def champion(self) -> PinnaRecord | None:
        ranked = self.ranking()
        return ranked[0] if ranked else None

    def tested_tags(self) -> set[str]:
        """判別力のある難易度で十分に試した耳介。

        4方向でしか試していないものは「未検証」として扱い、もう一度
        出題の対象にする。易しい回の満点は情報を持たない。
        """
        return {t for t, r in self.records.items() if r.ranked}

    def next_challengers(self, all_candidates: list[tuple[str, str]],
                         count: int) -> list[tuple[str, str]]:
        """まだ十分に試していない耳介を、リスト順に `count` 件。"""
        done = self.tested_tags()
        return [(tag, label) for tag, label in all_candidates
                if _tag_key(tag) not in done][:count]

    def build_lineup(self, all_candidates: list[tuple[str, str]],
                     slots: int) -> tuple[list[tuple[str, str]], str | None]:
        """今回のセッションで比べる顔ぶれを決める。

        チャンピオンがいれば必ず1枠に入れる（基準として同席させないと、
        上達の影響と耳介の差を切り分けられない）。残りを新顔で埋める。

        Returns (候補リスト, 基準の tag)。候補・基準とも、返すのは
        呼び出し側が渡した形のまま（アプリはパスを渡し、それで SOFA を
        読み込む）。記録側の正規化とは独立している。
        """
        if slots < 2:
            raise ValueError(tr("最低2枠必要です（基準＋挑戦者）",
                                "At least 2 slots are needed (reference "
                                "and challenger)"))
        champ = self.champion()
        by_key = {_tag_key(t): (t, label) for t, label in all_candidates}
        lineup: list[tuple[str, str]] = []
        anchor: str | None = None
        if champ is not None and champ.tag in by_key:
            original = by_key[champ.tag]
            anchor = original[0]
            lineup.append(original)
        lineup.extend(self.next_challengers(all_candidates,
                                            slots - len(lineup)))
        if len(lineup) < 2:
            # 新顔が尽きた。順位下位を再検証に回す。
            for rec in reversed(self.ranking()):
                cand = by_key.get(rec.tag)
                if cand is not None and all(cand[0] != t for t, _ in lineup):
                    lineup.append(cand)
                if len(lineup) >= slots:
                    break
        return lineup, anchor

    def progress_text(self, all_candidates: list[tuple[str, str]]) -> str:
        """進捗の読み上げ文。"""
        total = len(all_candidates)
        done = len([t for t, _ in all_candidates
                    if _tag_key(t) in self.tested_tags()])
        remaining = total - done
        if remaining <= 0:
            return tr(f"全{total}種類の検証が終わりました。",
                      f"All {total} have been tested.")
        return tr(f"{total}種類中{done}種類が検証済み、残り{remaining}種類です。",
                  f"{done} of {total} have been tested, {remaining} to go.")

    def relabel(self, all_candidates: list[tuple[str, str]]) -> None:
        """記録の表示名を、いまの候補の表示名に合わせる。

        表示名は保存したときの言語のまま残るので、言語を変えて起動したとき
        に古い言語の名前を読まないよう、候補（いまの言語）の名前で上書きする。
        """
        names = {_tag_key(t): label for t, label in all_candidates}
        for tag, rec in self.records.items():
            if tag in names:
                rec.label = names[tag]

    def summary(self, all_candidates: list[tuple[str, str]]) -> str:
        """通算成績の読み上げ文。"""
        self.relabel(all_candidates)
        ranked = self.ranking()
        if not ranked:
            return tr("まだ順位を出せるだけの試行がありません。"
                      f"1つの耳介につき{MIN_TRIALS_FOR_RANKING}問以上必要です。",
                      "There are not enough trials to rank yet. Each pinna "
                      f"needs {MIN_TRIALS_FOR_RANKING} questions or more.")
        top = ranked[0]
        parts = [tr(f"通算の最良は{top.label}です。",
                    f"The best overall is {top.label}."),
                 top.describe(self.fb_name) + tr("。", ".")]
        if len(ranked) > 1:
            second = ranked[1]
            gap = (top.mean_delta - second.mean_delta) * 100
            # 16問での1問差は約6ポイント。この程度は実力差と呼べないので
            # 偶然の可能性に触れる。10ポイント（16問中1.6問）を境目とする。
            if gap < 10.0:
                parts.append(tr(f"ただし2位の{second.label}との差は"
                                f"{gap:.0f}ポイントで小さく、"
                                "偶然の範囲かもしれません。"
                                "問数を増やすと確かになります。",
                                f"But the gap to second place, "
                                f"{second.label}, is only {gap:.0f} points "
                                "and may be chance. More questions will "
                                "make it surer."))
            else:
                parts.append(tr(f"2位は{second.label}で、"
                                f"差は{gap:.0f}ポイントです。",
                                f"Second is {second.label}, {gap:.0f} "
                                "points behind."))
        parts.append(self.progress_text(all_candidates))
        return join_sentences(*parts)

    def detail_lines(self) -> list[str]:
        lines = [r.describe(self.fb_name) for r in self.ranking()]
        pending = [r for r in self.records.values() if not r.ranked]
        for r in pending:
            lines.append(tr(f"{r.label}: 試行不足 ({r.n_answered}問)",
                            f"{r.label}: not enough trials "
                            f"({r.n_answered} questions)"))
        return lines
