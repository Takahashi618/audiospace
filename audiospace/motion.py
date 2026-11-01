"""リスナーの動きによる動的な方向手がかり。

なぜ必要か
----------
ヘッドホンで他人のHRTFを使うと、前後の取り違えがほぼ必ず起きる。脳は
自分の耳介のスペクトル特性で前後を判断しているため、他人の耳介では
手がかりが合わないからである。訓練すれば改善するが、初見のプレイヤーに
それを期待はできない。

一方、頭をわずかに動かしたときの音の動き方は、耳介の個人差と無関係に
前後を決定づける。音源が前にあれば、頭を左に振ると音は相対的に右へ動く。
後ろにあれば逆に動く。この差は学習不要で誰でも即座に分かる。実世界で
人が無意識にやっていることでもある。

したがって「首振り」は、HRTFの品質を上げるより確実な前後対策になる。

使い方
------
ゲームがプレイヤーの向きを自由に変えられるなら、それが最良で、この
モジュールは不要。向きが固定される場面（音を聞くだけの探索、メニュー、
方向当て）では、ごく小さな自動振りを足すと前後が分かるようになる。

    engine.set_head_wobble(amplitude_deg=6.0, rate_hz=0.5)

振り幅は 4〜10 度で十分。大きくすると音像が揺れて不快になる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .i18n import tr

#: 読み上げでの呼び名（首振りと首かしげ）
WOBBLE_NOUN = tr("首振り", "Head wobble")
ROLL_NOUN = tr("首かしげ", "Head tilt")


@dataclass
class HeadWobble:
    """リスナーの向きに加える微小な往復回転。

    Attributes
    ----------
    amplitude_deg 片側の振り幅。0 で無効
    rate_hz       往復の周波数。速すぎると音像が震えて聞き取りにくい
    shape         "sine" は滑らかで自然。"triangle" は角速度が一定で、
                  動きの向きが分かりやすい代わりに折り返しが目立つ
    """

    amplitude_deg: float = 0.0
    rate_hz: float = 0.5
    shape: str = "sine"
    #: 読み上げでの呼び名。首かしげ（ロール）にも同じ揺れを使うため
    noun: str = WOBBLE_NOUN

    def __post_init__(self) -> None:
        if self.amplitude_deg < 0:
            raise ValueError(tr("amplitude_deg は 0 以上である必要があります",
                                "amplitude_deg must be 0 or more"))
        if self.rate_hz <= 0:
            raise ValueError(tr("rate_hz は正である必要があります",
                                "rate_hz must be positive"))
        if self.shape not in ("sine", "triangle"):
            raise ValueError(tr(f"未知の shape: {self.shape}",
                                f"unknown shape: {self.shape}"))

    @property
    def enabled(self) -> bool:
        return self.amplitude_deg > 0.0

    def offset_at(self, seconds: float) -> float:
        """時刻 `seconds` における yaw のオフセット（度）。"""
        if not self.enabled:
            return 0.0
        phase = (seconds * self.rate_hz) % 1.0
        if self.shape == "triangle":
            # 0 -> 1 -> 0 -> -1 -> 0 を直線でつなぐ
            wave = 4.0 * phase - 1.0 if phase < 0.5 else 3.0 - 4.0 * phase
        else:
            wave = math.sin(2.0 * math.pi * phase)
        return self.amplitude_deg * wave

    def describe(self) -> str:
        """読み上げ用の説明。"""
        if not self.enabled:
            return tr(f"{self.noun}なし", f"{self.noun} off")
        if self.shape == "triangle":
            name = tr("三角波", "triangle wave")
        else:
            name = tr("サイン波", "sine wave")
        return tr(f"{self.noun}あり、振り幅{self.amplitude_deg:g}度、"
                  f"周期{1.0 / self.rate_hz:.1f}秒、{name}",
                  f"{self.noun} on, {self.amplitude_deg:g} degrees, "
                  f"period {1.0 / self.rate_hz:.1f} seconds, {name}")


#: 用途別のおすすめ設定
WOBBLE_PRESETS: dict[str, HeadWobble] = {
    "off": HeadWobble(0.0),
    "subtle": HeadWobble(4.0, 0.6),
    "clear": HeadWobble(8.0, 0.45),
    "strong": HeadWobble(14.0, 0.35, "triangle"),
}


def wobble_labels() -> list[tuple[str, str]]:
    """(キー, 読み上げ用の表示名) の一覧。"""
    return [
        ("off", tr("なし", "Off")),
        ("subtle", tr("弱め (振り幅4度)", "Subtle (4 degrees)")),
        ("clear", tr("標準 (振り幅8度、推奨)",
                     "Standard (8 degrees, recommended)")),
        ("strong", tr("強め (振り幅14度、三角波)",
                      "Strong (14 degrees, triangle wave)")),
    ]


# ---------------------------------------------------------------------------
# 首かしげ（ロール）
# ---------------------------------------------------------------------------
# 首振り（yaw）の上下版。首を左右の肩へ交互に傾ける。
#
# 傾けると、上の音と下の音は左右へ逆向きに動く（左耳を下げれば、上の音は
# 右耳側へ、下の音は左耳側へ）。正面・水平の音だけは動かない。左右の動きは
# 両耳間の差で聞こえるので、耳介の個人差と無関係に効く。首振りが前後を
# 決めたのと同じ理屈で、上下を決める手がかりになる。
#
# 首振りより振り幅を大きめにしてある。上45度の音を左右へ動かす量は
# おおよそ sin(振り幅)×cos(45度) で、首振りが正面の音を動かす量より小さい。
#: 首かしげの用途別の設定。キーと意味は WOBBLE_PRESETS と揃えてある
ROLL_PRESETS: dict[str, HeadWobble] = {
    "off": HeadWobble(0.0, noun=ROLL_NOUN),
    "subtle": HeadWobble(6.0, 0.4, noun=ROLL_NOUN),
    "clear": HeadWobble(12.0, 0.35, noun=ROLL_NOUN),
    "strong": HeadWobble(20.0, 0.3, noun=ROLL_NOUN),
}


def roll_labels() -> list[tuple[str, str]]:
    """(キー, 読み上げ用の表示名) の一覧。"""
    return [
        ("off", tr("なし", "Off")),
        ("subtle", tr("弱め (振り幅6度)", "Subtle (6 degrees)")),
        ("clear", tr("標準 (振り幅12度)", "Standard (12 degrees)")),
        ("strong", tr("強め (振り幅20度)", "Strong (20 degrees)")),
    ]
