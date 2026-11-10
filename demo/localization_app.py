"""定位確認アプリ -- 実際に耳で聞いて HRTF の効きを確かめる。

前後の取り違えは、他人のHRTFをヘッドホンで使うと必ず起きる。訓練すれば
分かるようになるが、初見のプレイヤーにそれは期待できない。そこで本アプリ
には3つの対策を入れてある。

  首振り  頭をわずかに動かしたときの音の動き方は、前後で必ず逆になる。
          これは耳介の個人差と無関係な幾何的な手がかりなので、学習なしで
          誰にでも効く。矢印キーで自分で向きを変えることもできる。
  訓練    間違えた方向を重点的に出し直し、直後に正解方向を鳴らして
          聞き比べさせる。連続正解で終了。
  耳介選定 Viking の耳介を順に試して、その人に合うものを見つける。

上下（仰角）の聴き取りテストもある。上下の手がかりは耳介だけなので、
実測の全天球 HRTF（Viking）を選んでいるときにだけ実施できる。

アクセシビリティ方針 (全プロジェクト共通):
- すべてのウィジェットに SetName、StaticText は対応コントロールより先に生成
- キー処理はパネルの EVT_CHAR_HOOK 一本。ウィジェット個別の
  EVT_KEY_DOWN は使わない
- 動作に文字キーを割り当てない (IME と衝突する)。F キー、テンキー、
  矢印、Escape のみ
- リストは wx.ListBox。読み取り専用の複数行 wx.TextCtrl をフォーカス先に
- speak() は UI 更新より先に呼ぶ (スクリーンリーダーの自動読み上げに
  上書きされるのを防ぐ)

起動:
    python -X utf8 demo/localization_app.py
    python -X utf8 demo/localization_app.py --elevation   (上下の聴き取りから始める)
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import wx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import (DEFAULT_PINNA, ROLL_PRESETS,  # noqa: E402
                        WOBBLE_PRESETS, HrirSet, SpatialPlayer,
                        preset_names, roll_labels, space_labels,
                        wobble_labels)
from demo.quiz import (ELEVATION_LEVELS, ROTATION_ARC_SECONDS,  # noqa: E402
                       ABWobbleSession, CalibrationSession, Difficulty,
                       ElevationCalibrationSession, ElevationSession,
                       FrontElevationSession, QuizSession, RotationSession,
                       RotationTrial, TrainingSession, direction_name,
                       elevation_name, position_for, rotation_name)
from demo.records import (MIN_DIRECTIONS_FOR_RANKING,  # noqa: E402
                          RecordStore)
from demo.preset_editor import ReverbEditor, SpaceEditor  # noqa: E402
from demo.user_presets import DEFAULT_PATH as USER_PRESETS_PATH  # noqa: E402
from demo.user_presets import UserPresetStore  # noqa: E402
from audiospace import (FURNISH_LEVELS, contents_text,  # noqa: E402
                        furnish_counts, reverb_from_spec, space_from_spec,
                        space_preset)
from demo.speech import Speech  # noqa: E402
from demo.preset_editor import USER_PREFIX  # noqa: E402
from audiospace.i18n import is_japanese, join_sentences, tr  # noqa: E402

HRTF_DIR = ROOT / "assets" / "hrtf" / "viking"
LOG_DIR = ROOT / "logs"
APP_TITLE = tr("定位確認 - audiospace", "Localization Tool - audiospace")
TURN_STEP_DEG = 15.0
# 顔の上下（Page Up で上、Page Down で下）。真上・真下で止める
PITCH_STEP_DEG = 15.0
PITCH_LIMIT_DEG = 90.0
# 繰り返し再生の間隔。音の長さ＋REPEAT_GAP_S、ただし REPEAT_MIN_S 以上。
# ノイズバースト3連（1.2秒）なら 2.2秒ごとに鳴る。
REPEAT_GAP_S = 1.0
REPEAT_MIN_S = 2.0
# 基準音（上下の聴き取りで、先に鳴らす同じ方位・水平の音）。基準音と問題の
# 音の間は短く、繰り返しの間は長くする。逆だと、どちらが基準でどちらが
# 問題か、組の区切りが分からなくなる。
REFERENCE_GAP_S = 0.5
REFERENCE_REPEAT_GAP_S = 1.5
# 上下の強調はエンジンの機能（SpatialEngine.elevation_emphasis）。以前は
# ここで音を作るときに焼き込んでいたが、鳴らし始めた後は変えられず、F2 の
# 試聴の回る音にはかけられなかった。エンジンなら音源ごとにその場で効く。
# 床の反射。床は耳の FLOOR_DEPTH_M 下。人が立った耳の高さ（約1.6m）に
# しないのは、4m 先の下45度の音（耳の2.83m下）が床の下に潜ってしまう
# ため。反射音は床に映した鏡像の位置から、行程の差だけ遅れて届く。
# 高い音ほど遅れが長い（真上で約17ms、水平で約9ms、下45度で約0.7ms）。
FLOOR_DEPTH_M = 3.0
FLOOR_REFLECTION_GAIN = 0.6     # 床で失う分。硬い床ほど 1 に近い
FLOOR_IMAGE_MIN_EL = -45.0      # 反射音の方向の下限（Viking の測定範囲）
SPEED_OF_SOUND = 343.0
#: 耳介の選定に要る実測 HRTF が無いとき
NOT_ENOUGH_HRTF = tr("実測HRTFが足りません。"
                     "tools のフェッチスクリプトで取得してください。",
                     "There are not enough measured HRTFs. Get them with "
                     "the fetch script in tools.")


def answer_text(number: int, correct: bool, extra: str, truth: str) -> str:
    """答えた直後の読み上げ（「3問目 正解。正解は前でした。」）。"""
    mark = tr("正解", "Correct") if correct else tr("不正解", "Wrong")
    return tr(f"{number}問目 {mark}{extra}。正解は{truth}でした。",
              f"Question {number}: {mark}{extra}. The correct answer was "
              f"{truth}.")


def list_hrtf_choices() -> list[tuple[str, Path | None]]:
    """(表示名, SOFAパス) の一覧。合成HRIRは None。"""
    items: list[tuple[str, Path | None]] = [
        (tr("合成HRIR (前後の区別なし、依存最小)",
            "Synthetic HRIR (no front/back difference, fewest "
            "dependencies)"), None)]
    for path in sorted(HRTF_DIR.glob("subj_*.sofa")):
        letter = path.stem.replace("subj_", "")
        if letter == "Z":
            label = tr("実測 Viking 耳介なし", "Measured Viking without pinna")
        else:
            label = tr(f"実測 Viking 耳介{letter}",
                       f"Measured Viking pinna {letter}")
        items.append((label, path))
    return items


def floor_image(position: tuple[float, float, float]
                ) -> tuple[tuple[float, float, float] | None, float]:
    """床の反射の鏡像の位置と、直接音からの遅れ（秒）を返す。

    聞き手の耳は原点、床は z = -FLOOR_DEPTH_M。床に映した位置
    （z を床について折り返す）から来る音として鳴らせば、方向も距離による
    減衰もエンジンがそのまま扱う。遅れは行程の差を音速で割ったもの。
    音源が床より下にあるときは反射がない（None を返す）。
    """
    x, y, z = position
    floor = -FLOOR_DEPTH_M
    if z < floor:
        return None, 0.0
    image = (x, y, 2.0 * floor - z)
    direct = float(np.linalg.norm(position))
    reflected = float(np.linalg.norm(image))
    delay = (reflected - direct) / SPEED_OF_SOUND
    # 反射音は下56〜90度から来るが、Viking の測定は下45度まで。そのまま
    # 鳴らすとエンジンは最も近い測定方向を選び、真下（真上の音の反射）では
    # 下45度の全方位が等距離になって、左右どちらかに寄った音になりうる。
    # 方向だけ「同じ方位の、測定範囲の一番下」に寄せる。真上の音は方位を
    # 持たないので正面にする（左右対称を保つ）。時間差は本来の値のまま。
    horiz = math.hypot(x, y)
    el = math.degrees(math.atan2(image[2], horiz))
    if el < FLOOR_IMAGE_MIN_EL:
        az = math.atan2(-x, y) if horiz > 1e-9 else 0.0
        e = math.radians(FLOOR_IMAGE_MIN_EL)
        image = (-reflected * math.cos(e) * math.sin(az),
                 reflected * math.cos(e) * math.cos(az),
                 reflected * math.sin(e))
    return image, delay


class SoundBank:
    """定位確認に向いた音を生成する。

    広帯域ノイズが最も定位しやすい。純音は両耳間強度差が周波数に強く
    依存するため、方向手がかりとしては弱い。
    """

    #: 遅らせた音を保持する数。1問で使うのは多くて3種類（問題の音と、
    #: 基準音・問題の音それぞれの床の反射）、1回のテストでも十数種類なので
    #: 十分足りる。上限が無いと、遅れの種類だけ1〜2秒ぶんの音が増え続ける。
    VARIANT_CACHE_MAX = 64

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._cache: dict[str, np.ndarray] = {}
        self._variants: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def get(self, kind: str) -> np.ndarray:
        if kind not in self._cache:
            self._cache[kind] = self._make(kind)
        return self._cache[kind]

    def variant(self, kind: str, delay_s: float = 0.0) -> np.ndarray:
        """遅らせた音。

        遅らせるのは、基準音のあとの出題音と、床の反射音のため。2つの音を
        別々の音源として同時に再生し、後の方を無音で遅らせる。タイマーで
        後から鳴らすと、その間に回答や中止が入ったときの取り消しが要る。
        同時に始めれば、止めるときも一緒に止まる。

        作ったものは最近使った順に VARIANT_CACHE_MAX 個まで保持する。
        """
        pad_n = int(round(delay_s * self.sample_rate))
        if pad_n == 0:
            return self.get(kind)
        key = (kind, pad_n)
        data = self._variants.get(key)
        if data is not None:
            self._variants.move_to_end(key)
            return data
        data = np.concatenate([np.zeros(pad_n, dtype=np.float32),
                               self.get(kind)])
        self._variants[key] = data
        while len(self._variants) > self.VARIANT_CACHE_MAX:
            self._variants.popitem(last=False)
        return data

    def _make(self, kind: str) -> np.ndarray:
        sr = self.sample_rate
        rng = np.random.default_rng(12345)

        def noise(n: int, peak: float) -> np.ndarray:
            """ピークを揃えたノイズ。

            ガウス雑音は裾が伸びるので、標準偏差で調整するとピークが
            1 を超えてクリップする。実測で 1.73 まで出たため、生成後に
            正規化する。
            """
            x = rng.standard_normal(n).astype(np.float32)
            top = float(np.max(np.abs(x)))
            return x * (peak / top) if top > 0 else x

        if kind == "noise_bursts":
            out = np.zeros(int(sr * 1.2), dtype=np.float32)
            burst = int(sr * 0.06)
            env = np.concatenate([
                np.linspace(0, 1, int(burst * 0.02), dtype=np.float32),
                np.ones(burst - int(burst * 0.02) - int(burst * 0.5),
                        dtype=np.float32),
                np.linspace(1, 0, int(burst * 0.5), dtype=np.float32)])
            for i in range(3):
                start = int(sr * 0.35) * i
                out[start:start + len(env)] = noise(len(env), 0.7) * env
            return out
        if kind == "continuous":
            return noise(int(sr * 2.0), 0.5)
        if kind == "click":
            out = np.zeros(int(sr * 0.15), dtype=np.float32)
            out[0] = 0.9
            return out
        t = np.arange(int(sr * 0.4), dtype=np.float32) / sr
        env = np.minimum(1.0, np.minimum(t * 60, (t[-1] - t) * 60))
        return (0.4 * np.sin(2 * np.pi * 660 * t) * env).astype(np.float32)


SOUND_KINDS = [
    (tr("ノイズバースト3連 (定位しやすい、推奨)",
        "Three noise bursts (easy to locate, recommended)"), "noise_bursts"),
    (tr("連続ノイズ", "Continuous noise"), "continuous"),
    (tr("クリック", "Click"), "click"),
    (tr("ビープ音", "Beep"), "beep"),
]

HELP_TEXT_JA = """定位確認ツール

■ モード
F2  試聴の開始と停止 (音が周囲を回ります)
Shift+F2 上下の聞き比べ (採点なし)
    テンキーのスラッシュで真上、8で上、5で水平、2で下の音が
    正面で鳴ります。テストの前に4つの高さを聞き比べるためのモードです。
    押したときは読み上げません。
Ctrl+Shift+F2 8方向の聞き比べ (採点なし)
    テンキーの1から9 (5を除く) で、その方向の音が水平に鳴ります。
    キーの並びがそのまま方向の配置です。8で前、2で後ろ、4で左、6で右。
    方向は、いまの自分から見た向きです。押したときは読み上げません。
F3  クイズ (難易度に応じて10問から24問、成績を測る)
F7  訓練 (間違えた方向を重点的に、連続正解で終了)
F8  耳介の選定 (複数の耳介を試して合うものを探す)
    毎回、現在の最良を基準として同席させます。回を重ねると
    聞き手も上達するため、同じ回の中で比べないと差が分かりません。
    結果は通算で積み上がるので、何回かに分けて全種類を試せます。
F11 首振りの効果測定 (ありとなしを1問ごとに交互に出して比べる)
    どちらの条件かは出題時には言いません。実施中は矢印キーでの
    向きの変更を控えると、条件の比較が正確になります。
Shift+F3 上下の聴き取り (真上・上・水平・下の4択)
    実測 Viking を選んでいるときだけ使えます。方位は難易度の
    方向から選びますが、答えるのは高さだけです。
Ctrl+F3 正面固定の上下の聴き取り (真上・上・水平・下の4択)
    音は毎回正面から鳴り、変わるのは高さだけです。
    上下の聞こえ方の違いそのものを聞き比べるためのモードです。
Ctrl+F8 上下用の耳介の選定
    F8 の上下版です。正面固定の上下4択で耳介を順に試します。
    通算成績は前後用とは別に積み上がり、F10 で続けて読み上げます。
Ctrl+F2 回る向きの聴き取り (24問)
    音が自分の周りを半周します。水平に回る音は、左回りか右回りかを
    答えます。左回りは上から見て反時計回り、前から左、後ろ、右の順です。
    回り始めの方向は毎回変わります。前と後ろを取り違えると、回る向きが
    逆に聞こえます。頭上を越える音は、前から後ろへか、後ろから前へかを
    答えます (実測 Viking を選んでいるときだけ出題します)。

■ 操作
テンキーの0  繰り返し再生のオンとオフ (起動時はオン)
    オンのときは、テンキーで答えるまで出題音を間を空けて
    鳴らし続けます。読み上げに紛れて聞き逃しても待てば聞けます。
テンキーの*  基準音のオンとオフ (起動時はオン)
    上下の聴き取りで、問題の音の前に同じ方位・水平の音を鳴らします。
    2つ目の音が1つ目より上か下かを比べて答えられます。
F4  出題をもう一度聞く
F5  結果の読み上げ
F6  音を止める (繰り返しも止まります。F4 で再開)
F9  いまの設定の読み上げ
F10 耳介の通算成績の読み上げ
F1  この説明の窓を開く (上下の矢印で1行ずつ読み、Escape で閉じる)
Escape  終了

■ 回答 (テンキー)
7 8 9      左斜め前 / 前 / 右斜め前
4   6      左       /    / 右
1 2 3      左斜め後ろ / 後ろ / 右斜め後ろ
キーの並びがそのまま方向の配置になっています。

■ 回る向きの聴き取りでの回答 (テンキー)
4  左回り (前から左へ回る向き)
6  右回り (前から右へ回る向き)
8  頭上を後ろから前へ
2  頭上を前から後ろへ
音が進む向きを、キーの位置で表しています。

■ 上下の聴き取りでの回答 (テンキー、上から縦に並ぶ)
/  真上 (90度)
8  上 (45度)
5  水平
2  下 (45度)
下は45度までです。実測データが下45度までしかないためです。

■ 向きを変える (前後を見分けるのに有効)
左右の矢印キー  その場で15度ずつ向きを変える
上下の矢印キーは画面の操作 (選択肢や一覧の移動) に使います。
音源は動きません。自分が回ると、前の音と後ろの音は
逆向きに動きます。この差は誰にでも分かります。
回答は「いまの自分から見た方向」で答えてください。

■ 上下を聞き分ける手がかり (設定欄。起動時はすべてオフ)
首かしげ     首を左右の肩へ交互に傾ける揺れを自動で加えます。
             上の音と下の音が左右へ逆向きに動き、正面・水平の音は
             動きません。耳の形の個人差に左右されない手がかりです。
上下の強調   上の音ほど高い音域を明るく、下の音ほどこもらせます。
             自然な聞こえ方ではありませんが、覚えやすい手がかりです。
             顔から見た高さで決まり、F2 の試聴の回る音にもかかります。
             試聴中に Page Up で上を向くと回る音が暗く、Page Down で
             下を向くと明るくなります。
床の反射     床で跳ね返った音を加えます。高い音ほど、直接の音と
             反射音の時間差が長くなります。
いずれも Tab で設定欄へ移動して切り替えます。

■ 空間 (設定欄、起動時はなし)
木の床と木の壁の部屋、コンクリートの部屋、何もない道、建物の間の道
などから選びます。床・壁・天井の材質で、壁や床からの反射が方向つきで
聞こえ、残響の長さと音色も変わります。選んでいる間は「残響」の設定より
優先します。空間には床や地面の反射が含まれるので、「床の反射」は
オフにしてください (両方オンだと床の反射が二重になります)。
「精密な残響」をオンにすると、閉じた部屋の残響を精密に計算した響きに
します (用意した部屋は計算済みですぐ切り替わります)。建物の間の道や
何もない道など、開いた面のある空間では速い版のままです。
「空間を自分で作る」と「残響を自分で作る」のボタン (設定欄、Tab で移動) で、
大きさや6つの面の材質、残響の長さなどを決め、名前を付けて保存できます。
保存したものは「自作」として空間や残響の選択肢に並びます。窓の中の
「この設定で鳴らす」で、保存する前に正面の音で聞けます。
「家具の量」と「人の数」(設定欄) で、選んでいる空間に家具と人を足せます。
部屋の大きさは同じでも、物や人がいると残響が短くなり、反射が散らされます。
家具は閉じた部屋のときだけ効き、人はどの空間にも足されます。

■ 顔を上下に向ける
Page Up    15度ずつ上を向く (真上の90度まで)
Page Down  15度ずつ下を向く (真下の90度まで)
音源は動きません。上を向くと、上の音が正面に近づきます。
上下の聴き取りでは、顔の向きに関係なく
「体から見た音の高さ」で答えてください。
顔を大きく上に向けたまま下の音を聞くと、実測データの範囲
(下45度まで) を外れ、正しい音になりません。

■ 注意
ヘッドホンでの使用を前提としています。
スピーカーでは左右の音が混ざるため定位しません。
"""

# 英語の説明。見出しは「■」ではなく、行末のコロンで表す（記号を読み上げで
# 「black square」などと読まれないように）。tools/build_manual.py が
# キーの行を読むので、キーの行の形（キー、空白、説明。続きは4字下げ）は
# 日本語と同じにする
HELP_TEXT_EN = """Localization Tool

Modes:
F2  Start and stop the preview (the sound circles around you)
Shift+F2 Compare heights (not scored)
    Numpad slash plays a sound straight up, 8 up, 5 level and 2 down,
    in front of you. Use it to compare the four heights before a test.
    Pressing a key does not speak.
Ctrl+Shift+F2 Compare 8 directions (not scored)
    Numpad 1 to 9 (except 5) play a level sound from that direction.
    The keys are laid out like the directions: 8 front, 2 back,
    4 left, 6 right. Directions are as seen from where you face now.
    Pressing a key does not speak.
F3  Quiz (10 to 24 questions depending on the difficulty, scored)
F7  Training (focuses on directions you miss, ends after a streak)
F8  Choose a pinna (tries several pinnae to find one that suits you)
    Every round includes the current best as a reference. Listeners
    improve as they go, so differences only show within one round.
    Results add up, so you can try them all over several rounds.
F11 Measure the head wobble effect (alternates with and without)
    It does not say which condition a question uses. Avoid turning
    with the arrow keys during the test for a fairer comparison.
Shift+F3 Up/down test (straight up, up, level or down)
    Only with a measured Viking HRTF. The directions come from the
    difficulty, but you answer only the height.
Ctrl+F3 Up/down test from the front (straight up, up, level or down)
    The sound always comes from the front, and only the height changes.
    Use it to learn how the heights sound different.
Ctrl+F8 Choose a pinna for up and down
    The up/down version of F8. It tries pinnae one after another with
    the up/down test from the front. Its totals are kept apart from
    the front/back ones, and F10 reads them after those.
Ctrl+F2 Rotation test (24 questions)
    A sound travels half a circle around you. For a level circle,
    answer counterclockwise or clockwise. Counterclockwise is as seen
    from above: front, then left, back, right. The starting point
    changes every time. Confusing front and back makes the direction
    sound reversed. For a sound passing overhead, answer back to front
    or front to back (asked only with a measured Viking HRTF).

Controls:
Numpad 0  Repeat on or off (on at start)
    When on, the question sound plays again and again with pauses
    until you answer with the keypad. If speech covers it, just wait.
Numpad *  Reference sound on or off (on at start)
    In the up/down tests, a level sound from the same direction plays
    before the question sound. You can answer whether the second sound
    is higher or lower than the first.
F4  Hear the question again
F5  Read the results
F6  Stop the sound (repeating stops too; F4 starts it again)
F9  Read the current settings
F10 Read the pinna totals
F1  Open this help (up and down arrows read line by line, Escape closes)
Escape  Quit

Answers (numeric keypad):
7 8 9      front left / front / front right
4   6      left       /       / right
1 2 3      back left  / back  / back right
The keys are laid out like the directions.

Answers in the rotation test (numeric keypad):
4  counterclockwise (from the front toward the left)
6  clockwise (from the front toward the right)
8  overhead, back to front
2  overhead, front to back
The position of the key shows which way the sound moves.

Answers in the up/down tests (numeric keypad, top to bottom):
/  straight up (90 degrees)
8  up (45 degrees)
5  level
2  down (45 degrees)
Down goes only to 45 degrees, because the measurements stop there.

Turning (helps to tell front from back):
Left/Right arrows  Turn 15 degrees on the spot
The up and down arrows are for the screen (choices and lists).
The sounds do not move. When you turn, sounds in front and sounds
behind move in opposite directions. Anyone can hear this difference.
Answer with the direction as seen from where you face now.

Up/down cues (settings, all off at start):
Head tilt         Adds a sway of tilting your head toward each
                  shoulder in turn. Sounds above and below move left
                  and right in opposite directions, and sounds in
                  front at ear level do not move. It does not depend
                  on the shape of your ears.
Up/down emphasis  Makes higher sounds brighter and lower sounds more
                  muffled. Not natural, but easy to learn. It follows
                  the height as seen from your face, and also applies
                  to the circling sound of the F2 preview. Facing up
                  with Page Up during the preview makes it darker,
                  and facing down with Page Down makes it brighter.
Floor reflection  Adds the sound that bounces off the floor. The
                  higher the sound, the longer the gap between the
                  direct sound and the reflection.
Move to the settings with Tab to change them.

Spaces (settings, none at start):
Choose from a room with wooden floor and walls, a concrete room, an
empty road, a street between buildings and more. The materials of the
floor, walls and ceiling give reflections with a direction, and change
the length and tone of the reverb. While a space is selected, it takes
the place of the Reverb setting. Spaces include the floor or ground
reflection, so turn Floor reflection off (with both on, the floor
reflection is doubled).
Turn on Precise reverb to use a precisely computed reverb for closed
rooms (the provided rooms are computed already and switch at once).
Spaces with open sides, such as a street or an empty road, keep the
fast reverb.
With the Design a space and Design a reverb buttons (in the settings,
reach them with Tab) you can set the size, the materials of the six
surfaces, the reverb length and more, and save them with a name.
Saved ones appear as Yours in the space and reverb choices. Play with
these settings, in that window, lets you hear it before saving.
Furniture amount and Number of people (in the settings) add furniture
and people to the selected space. Even in a room of the same size,
things and people shorten the reverb and scatter the reflections.
Furniture works only in closed rooms; people are added to any space.

Facing up and down:
Page Up    Face 15 degrees further up (up to straight up, 90 degrees)
Page Down  Face 15 degrees further down (down to straight down)
The sounds do not move. Facing up brings sounds above toward the front.
In the up/down tests, answer with the height of the sound as seen
from your body, whichever way your face points.
Listening to a low sound with your face turned far up goes beyond
the measured range (down to 45 degrees), and it will not sound right.

Note:
This tool is meant for headphones.
With speakers, left and right mix, so the directions cannot be heard.
"""

HELP_TEXT = HELP_TEXT_JA if is_japanese() else HELP_TEXT_EN


class HelpDialog(wx.Dialog):
    """キー操作の説明を読む窓（F1）。

    以前は F1 で「状況」の欄に説明を書き込むだけで、フォーカスは元の場所に
    残った。スクリーンリーダーは「表示しました」としか読まず、起動直後は
    欄の中身も変わらないので、押しても何も起きないように聞こえた。
    専用の窓を開いて読み取り専用の欄にフォーカスを移せば、上下の矢印で
    1行ずつ読める。Escape で閉じると元の場所に戻る。

    モーダルにしない（ShowModal は入力を待って返らず、テストから開けない）。
    """

    def __init__(self, parent: wx.Window, text: str) -> None:
        super().__init__(parent, title=tr("キー操作の説明。Escape で閉じます",
                                          "Keyboard help. Escape closes"),
                         size=(680, 600),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetName(tr("キー操作の説明", "Keyboard help"))
        panel = wx.Panel(self)
        # 読み上げ名は直前の StaticText が勝つので、操作まで含めた文にする
        label = wx.StaticText(
            panel, label=tr("キー操作の説明。上下の矢印で1行ずつ読めます。"
                            "Escape で閉じます。",
                            "Keyboard help. Read it line by line with the "
                            "up and down arrows. Escape closes."))
        label.SetName(tr("キー操作の説明", "Keyboard help"))
        self.text = wx.TextCtrl(
            panel, value=text,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.text.SetName(tr("キー操作の説明の本文", "Keyboard help text"))
        self.btn_close = wx.Button(panel, wx.ID_CLOSE, label=tr("閉じる", "Close"))
        self.btn_close.SetName(tr("閉じる", "Close"))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(label, 0, wx.ALL, 8)
        sizer.Add(self.text, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        sizer.Add(self.btn_close, 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        panel.SetSizer(sizer)
        self.text.SetMinSize(wx.Size(-1, 200))
        self.btn_close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key)
        # F1 を押すと、Windows はキーとは別に「ヘルプの要求」も送る。親の
        # 窓まで伝わると説明を開き直そうとするので、ここで止める
        self.Bind(wx.EVT_HELP, lambda e: None)

    def show_text(self) -> None:
        """窓を前に出し、本文の先頭にフォーカスを置く。"""
        self.Show()
        self.Raise()
        self.text.SetFocus()
        self.text.SetInsertionPoint(0)
        self.text.ShowPosition(0)

    def on_key(self, event: wx.KeyEvent) -> None:
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.Close()
            return
        if event.GetKeyCode() == wx.WXK_F1:
            # 開いたまま先頭へ戻す（閉じる操作にすると、同じ押下で届く
            # 「ヘルプの要求」で開き直してしまうことがある）
            self.show_text()
            return
        event.Skip()


class MainFrame(wx.Frame):
    """設定と4つのモードを持つ1画面構成。"""

    def __init__(self, preset_path: Path | None = None) -> None:
        super().__init__(None, title=APP_TITLE, size=(820, 680))
        # 自分で作った空間と残響（設定欄のボタンから作る）。テストは別の
        # 場所を渡す（利用者のファイルを書き換えないように）
        self.user_presets = UserPresetStore(preset_path or USER_PRESETS_PATH)
        self._space_editor: SpaceEditor | None = None
        self._reverb_editor: ReverbEditor | None = None
        self.speech = Speech()
        self.player: SpatialPlayer | None = None
        self.bank: SoundBank | None = None
        self.session: QuizSession | None = None
        self.mode = "idle"  # idle / orbit / quiz / training / calib / ab / elev
        self._handle: int | None = None
        self._ref_handle: int | None = None   # 基準音の音源
        self._extra_handles: list[int] = []   # 床の反射音の音源
        # 基準音。上下の聴き取りで、問題の音の前に同じ方位・水平の音を
        # 鳴らす。「どの段か」を当てる絶対判断を、「基準より上か下か」の
        # 比較に変える。初回の4択では誤りの9割が隣の段とのずれだった。
        self.reference_enabled = True
        # 試聴（F2）の動かすタイマー。1つを使い回す（始めるたびに作って
        # つなぐと、つないだ処理が増え続ける）。動いている間だけ _orbit_timer
        # がそれを指す
        self._orbit_timer: wx.Timer | None = None
        self._orbit_obj = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_orbit, self._orbit_obj)
        # 回る向きの聴き取り: 音を動かすタイマー（1つを使い回す。鳴らすたびに
        # 作ってつなぐと、つないだ処理が増え続ける）と、いま回している問題
        self._rot_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_rotation_tick, self._rot_timer)
        self._rot_trial: RotationTrial | None = None
        self._rot_t0 = 0.0
        self._rot_yaw = 0.0
        self._orbit_angle = 0.0
        self._yaw = 0.0
        self._pitch = 0.0           # 顔の上下。正で上を向く
        self._hrir_cache: dict[str, HrirSet] = {}
        self._anchor_tag: str | None = None
        # 回答の受付は「出題音が鳴ってから」。回答直後の読み上げ待ちの間に
        # テンキーを押すと、まだ鳴っていない次の問題に回答できてしまう
        # （二度押しで1問を無自覚に消費する）ため、フラグで締める。
        self._accepting = False
        # 耳介の選定でいま実際に読み込んでいる耳介。切替失敗後の F4 再試行の
        # 判定と、ラウンド先頭の再出題（F4）で切替を繰り返さないために持つ。
        self._active_tag: str | None = None
        self._ab_wobble_key = "clear"
        # いま設定してある (空間の名前, 精密な残響か)
        self._applied_space: tuple | None = (None, False)
        # いま設定してある残響（空間が無いときの）。"unset" は未設定。
        # 同じ残響を毎回作り直すと、鳴っている響きがそのたびに途切れる
        self._applied_room: object = "unset"
        self._begin_gen = 0
        # 答えた直後（次の問題が鳴るまで）の印。この間の F4 は受け付けない
        self._pending_next = False
        self.records = RecordStore()
        # 上下用の耳介の通算成績。前後用とは別のファイルに積み上げる
        # （前後に合う耳介と上下に合う耳介は同じとは限らない）
        self.records_elev = RecordStore(LOG_DIR / "pinna_records_elev.json",
                                        fb_name=tr("上下", "up/down"))
        # 繰り返し再生。スクリーンリーダーの読み上げに出題音が紛れて
        # 聞き取れないことがあるので、回答するまで間を空けて鳴らし直す。
        # 1回ごとに張り直す単発タイマーにしてあるのは、音の種類を
        # 途中で変えたときに間隔が追従するようにするため。
        self.repeat_enabled = True
        self._repeat_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_repeat, self._repeat_timer)

        self.panel = wx.Panel(self)
        self.panel.SetName(tr("メイン", "Main"))
        self._build_ui()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.panel.Bind(wx.EVT_CHAR_HOOK, self.on_key)
        # F1 はキー（CHAR_HOOK）とは別に「ヘルプの要求」としても届く。
        # スクリーンリーダーがキーのほうを横取りしても開けるように、こちらでも
        # 説明を開く（同じ押下で両方届いても、窓は1つだけ）
        self.Bind(wx.EVT_HELP, lambda e: self.show_help())
        self._help_dlg: HelpDialog | None = None
        self.CreateStatusBar()
        self.SetStatusText(tr("F1 でキー操作の説明", "F1 for keyboard help"))
        self.Centre()

    # -- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        root = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=10)
        grid.AddGrowableCol(1, 1)

        def label(text: str, name: str) -> wx.StaticText:
            """ラベルを先に生成する。

            引数でコントロールを受け取るヘルパーにすると、呼び出し側で
            コントロールが先に評価されてしまい、生成順が逆転する。
            スクリーンリーダーは生成順で関連付けを推測することがあるので、
            ラベルを必ず先に作ってから返す形にしてある。
            """
            static = wx.StaticText(self.panel, label=text)
            static.SetName(name)
            return static

        def add(static: wx.StaticText, ctrl: wx.Window) -> None:
            grid.Add(static, 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(ctrl, 0, wx.EXPAND)

        lbl = label(tr("HRTF (&H)", "&HRTF"), "HRTF")
        self.hrtf_choices = list_hrtf_choices()
        self.cmb_hrtf = wx.Choice(
            self.panel, choices=[n for n, _ in self.hrtf_choices])
        self.cmb_hrtf.SetName(tr("HRTF の選択", "HRTF choice"))
        # 初期値は聴取テストで採用した既定の耳介（E）。以前はソート順の
        # 先頭（A）が選ばれていて、ライブラリの既定とずれていた。
        default_sel = next(
            (i for i, (_, p) in enumerate(self.hrtf_choices)
             if p is not None and p.stem == f"subj_{DEFAULT_PINNA}"),
            1 if len(self.hrtf_choices) > 1 else 0)
        self.cmb_hrtf.SetSelection(default_sel)
        add(lbl, self.cmb_hrtf)

        lbl = label(tr("音の種類 (&S)", "&Sound type"),
                    tr("音の種類", "Sound type"))
        self.cmb_sound = wx.Choice(
            self.panel, choices=[n for n, _ in SOUND_KINDS])
        self.cmb_sound.SetName(tr("音の種類の選択", "Sound type choice"))
        self.cmb_sound.SetSelection(0)
        add(lbl, self.cmb_sound)

        lbl = label(tr("首振り (&W)", "Head &wobble"),
                    tr("首振り", "Head wobble"))
        self.wobble_keys = [k for k, _ in wobble_labels()]
        self.cmb_wobble = wx.Choice(
            self.panel, choices=[v for _, v in wobble_labels()])
        self.cmb_wobble.SetName(tr("首振りの選択", "Head wobble choice"))
        self.cmb_wobble.SetSelection(2)          # 標準
        add(lbl, self.cmb_wobble)

        # 首かしげ（ロール）の揺れ。上下の音を左右へ逆向きに動かす
        lbl = label(tr("首かしげ (&T)", "Head &tilt"),
                    tr("首かしげ", "Head tilt"))
        self.roll_keys = [k for k, _ in roll_labels()]
        self.cmb_roll = wx.Choice(
            self.panel, choices=[v for _, v in roll_labels()])
        self.cmb_roll.SetName(tr("首かしげの選択", "Head tilt choice"))
        self.cmb_roll.SetSelection(0)            # 起動時はなし（利用者の指定）
        add(lbl, self.cmb_roll)

        lbl = label(tr("残響 (&R)", "&Reverb"), tr("残響", "Reverb"))
        # 選択肢の中身は refresh_user_choices で作る（自分で作った残響を
        # 後ろに足す）。room_keys は None / ("preset", 名前) / ("user", 名前)
        self.cmb_room = wx.Choice(self.panel, choices=[])
        self.cmb_room.SetName(tr("残響の選択", "Reverb choice"))
        add(lbl, self.cmb_room)
        # 自分で作る窓を開くボタン。ボタンは自分の文字が読み上げ名になるので
        # 左の欄にはラベルを置かない
        self.btn_reverb_edit = wx.Button(
            self.panel, label=tr("残響を自分で作る (&G)", "Desi&gn a reverb"))
        self.btn_reverb_edit.SetName(tr("残響を自分で作る", "Design a reverb"))
        grid.Add((0, 0))
        grid.Add(self.btn_reverb_edit, 0)

        # 材質つきの空間。選ぶと、面からの方向つきの反射と、材質から決まる
        # 残響になる（上の「残響」より優先する）
        lbl = label(tr("空間 (&P)", "S&pace"), tr("空間", "Space"))
        # space_keys は None / 用意した空間の名前 / ("user", 名前)
        self.cmb_space = wx.Choice(self.panel, choices=[])
        self.cmb_space.SetName(tr("空間の選択", "Space choice"))
        add(lbl, self.cmb_space)
        self.btn_space_edit = wx.Button(
            self.panel, label=tr("空間を自分で作る (&E)", "D&esign a space"))
        self.btn_space_edit.SetName(tr("空間を自分で作る", "Design a space"))
        grid.Add((0, 0))
        grid.Add(self.btn_space_edit, 0)
        self.refresh_user_choices()

        # 精密な残響: 空間の残響を、pyroomacoustics で精密に計算した尾にする
        # （用意した閉じた部屋は焼き込み済み。開いた面のある空間は速い版のまま）
        # 置いてあるもの。選んでいる空間に家具と人を足す（部屋の大きさは
        # 同じでも、物や人がいると残響が短くなり、反射が散らされる）
        lbl = label(tr("家具の量 (&M)。閉じた部屋のときだけ効きます",
                       "Furniture a&mount. Works only in closed rooms"),
                    tr("家具の量", "Furniture amount"))
        self.furnish_keys = [k for k, _ in FURNISH_LEVELS]
        self.cmb_furnish = wx.Choice(self.panel,
                                     choices=[v for _, v in FURNISH_LEVELS])
        self.cmb_furnish.SetName(tr("家具の量の選択", "Furniture amount choice"))
        self.cmb_furnish.SetSelection(0)
        add(lbl, self.cmb_furnish)
        lbl = label(tr("人の数 (&N)。立っている人として空間に足します",
                       "&Number of people. Added to the space as standing "
                       "people"),
                    tr("人の数", "Number of people"))
        self.spn_people = wx.SpinCtrl(self.panel, min=0, max=200, initial=0)
        self.spn_people.SetName(tr("人の数", "Number of people"))
        add(lbl, self.spn_people)

        lbl = label(tr("精密な残響 (&Q)", "Precise reverb (&Q)"),
                    tr("精密な残響", "Precise reverb"))
        self.chk_precise = wx.CheckBox(
            self.panel, label=tr("空間の残響を精密に計算する",
                                 "Compute the reverb of the space precisely"))
        self.chk_precise.SetName(tr("空間の残響を精密に計算する",
                                    "Compute the reverb of the space "
                                    "precisely"))
        self.chk_precise.SetValue(False)
        add(lbl, self.chk_precise)

        lbl = label(tr("難易度 (&D)", "&Difficulty"), tr("難易度", "Difficulty"))
        self.difficulties = list(Difficulty)
        self.cmb_diff = wx.Choice(
            self.panel, choices=[d.label for d in self.difficulties])
        self.cmb_diff.SetName(tr("難易度の選択", "Difficulty choice"))
        self.cmb_diff.SetSelection(1)
        add(lbl, self.cmb_diff)

        # 数値は整数のスピンで選ぶ。以前は wx.Slider に SL_LABELS を付けて
        # いたが、目盛りの数字（最小・最大・現在値）が小さな Static として
        # スライダーの直前に作られ、スクリーンリーダーは直前の Static を
        # 名前として読むので、名前が「35」になっていた。値も範囲に対する
        # 割合で返り、0〜90 の 35 は「38」と読まれた（MSAA で確認）。
        # 整数の wx.SpinCtrl は、直前のラベルが名前になり、値もそのまま読める。
        lbl = label(tr("前後の強調、パーセント (&B)",
                       "Front/&back emphasis, percent"),
                    tr("前後の強調", "Front/back emphasis"))
        self.spn_depth = wx.SpinCtrl(self.panel, min=0, max=90, initial=35)
        self.spn_depth.SetName(tr("前後の強調、パーセント",
                                  "Front/back emphasis, percent"))
        add(lbl, self.spn_depth)

        # 上下の強調。上の音ほど高域を持ち上げ、下の音ほど削る人工の手がかり
        lbl = label(tr("上下の強調、パーセント (&U)",
                       "&Up/down emphasis, percent"),
                    tr("上下の強調", "Up/down emphasis"))
        self.spn_tilt = wx.SpinCtrl(self.panel, min=0, max=100, initial=0)
        self.spn_tilt.SetName(tr("上下の強調、パーセント",
                                 "Up/down emphasis, percent"))
        add(lbl, self.spn_tilt)

        # 床の反射。音源の高さで直接音と反射音の時間差が変わる
        lbl = label(tr("床の反射 (&F)", "&Floor reflection"),
                    tr("床の反射", "Floor reflection"))
        self.chk_floor = wx.CheckBox(
            self.panel, label=tr("床の反射を加える", "Add the floor reflection"))
        self.chk_floor.SetName(tr("床の反射を加える", "Add the floor reflection"))
        self.chk_floor.SetValue(False)
        add(lbl, self.chk_floor)

        root.Add(grid, 0, wx.EXPAND | wx.ALL, 10)

        lbl_log = wx.StaticText(self.panel, label=tr("状況 (&L)", "Status &log"))
        lbl_log.SetName(tr("状況", "Status log"))
        self.txt_log = wx.TextCtrl(
            self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.txt_log.SetName(tr("状況の表示", "Status log text"))
        root.Add(lbl_log, 0, wx.LEFT | wx.RIGHT, 10)
        root.Add(self.txt_log, 1, wx.EXPAND | wx.ALL, 10)

        lbl_result = wx.StaticText(self.panel,
                                   label=tr("結果の一覧 (&K)", "Results (&K)"))
        lbl_result.SetName(tr("結果の一覧", "Results"))
        self.lst_result = wx.ListBox(self.panel, choices=[])
        self.lst_result.SetName(tr("結果の一覧の項目", "Results list"))
        root.Add(lbl_result, 0, wx.LEFT | wx.RIGHT, 10)
        root.Add(self.lst_result, 1, wx.EXPAND | wx.ALL, 10)

        self.panel.SetSizer(root)
        # HRTF は差し替えにエンジン再構築が要る。残りは実行中に反映できる。
        self.cmb_hrtf.Bind(wx.EVT_CHOICE, self.on_hrtf_changed)
        self.cmb_room.Bind(wx.EVT_CHOICE, self.on_setting_changed)
        self.cmb_space.Bind(wx.EVT_CHOICE, self.on_setting_changed)
        self.cmb_furnish.Bind(wx.EVT_CHOICE, self.on_setting_changed)
        self.spn_people.Bind(wx.EVT_SPINCTRL, self.on_setting_changed)
        self.spn_people.Bind(wx.EVT_TEXT, self.on_setting_changed)
        self.btn_reverb_edit.Bind(wx.EVT_BUTTON,
                                  lambda e: self.open_reverb_editor())
        self.btn_space_edit.Bind(wx.EVT_BUTTON,
                                 lambda e: self.open_space_editor())
        self.chk_precise.Bind(wx.EVT_CHECKBOX, self.on_setting_changed)
        self.cmb_wobble.Bind(wx.EVT_CHOICE, self.on_setting_changed)
        self.cmb_roll.Bind(wx.EVT_CHOICE, self.on_setting_changed)
        # 矢印での増減は EVT_SPINCTRL、数字の打ち込みは EVT_TEXT で届く
        self.spn_depth.Bind(wx.EVT_SPINCTRL, self.on_setting_changed)
        self.spn_depth.Bind(wx.EVT_TEXT, self.on_setting_changed)
        self.spn_tilt.Bind(wx.EVT_SPINCTRL, self.on_setting_changed)
        self.spn_tilt.Bind(wx.EVT_TEXT, self.on_setting_changed)
        # 床の反射は鳴らす音を遅らせて足すので、次に鳴らす音から効く
        self.txt_log.SetFocus()
        # 自分で作った空間と残響のファイルが読めなかったときは、説明の前に書く
        # （黙って空にすると、作ったものが消えたように見える）
        warning = "\n".join(w for w in (self.user_presets.load_error,
                                        self.records.load_error,
                                        self.records_elev.load_error) if w)
        self.txt_log.SetValue((warning + "\n\n" if warning else "") + HELP_TEXT)

    def log(self, text: str, speak: bool = True) -> None:
        """先に読み上げてから画面を更新する。順序を逆にすると、
        スクリーンリーダーの自動読み上げに発話が上書きされる。"""
        if speak:
            self.speech.say(text)
        self.txt_log.SetValue(text)

    # -- 設定 ---------------------------------------------------------
    def current_hrtf(self) -> tuple[str, Path | None]:
        return self.hrtf_choices[max(0, self.cmb_hrtf.GetSelection())]

    def settings_text(self) -> str:
        wobble = dict(wobble_labels())[
            self.wobble_keys[max(0, self.cmb_wobble.GetSelection())]]
        if self.mode == "ab":
            # A/B 測定中は首振りが問題ごとに切り替わる。設定値を書くと
            # 「この首振りで全問試した」と読めてしまう。
            wobble = tr("効果測定中のため問題ごとにありなしを切替",
                        "switched on and off per question for the "
                        "measurement")
        # 耳介の選定中は HRTF が問題ごとに切り替わるので、ドロップダウンの
        # 値を書くと誤解を招く。ログを後から読んだとき「耳介Aで試した」と
        # 読めてしまう。
        if self.mode in ("calib", "calib_elev"):
            hrtf = tr("耳介の選定中のため問題ごとに切替",
                      "switched per question while choosing a pinna")
        else:
            hrtf = self.current_hrtf()[0]
        # 正面固定の上下では難易度（方位の数）を使わない。書くと「前後左右で
        # 試した」と読めてしまう。上下用の耳介選定も正面固定で出題する。
        # 回る向きの聴き取りも難易度を使わない（回り始めは常に8方向）。
        if self.mode in ("elev_front", "calib_elev"):
            diff = tr("正面固定のため使わない",
                      "not used, because the sound is fixed in front")
        elif self.mode == "rotation":
            diff = tr("回る向きの聴き取りでは使わない（回り始めは8方向）",
                      "not used in the rotation test (it starts from 8 "
                      "directions)")
        else:
            diff = self.difficulties[max(0, self.cmb_diff.GetSelection())].label
        roll = dict(roll_labels())[self._roll_key()]
        sound = SOUND_KINDS[max(0, self.cmb_sound.GetSelection())][0]
        none = tr("なし", "none")
        room = self.cmb_room.GetStringSelection() or none
        space = self.cmb_space.GetStringSelection() or none
        furnish = self.cmb_furnish.GetStringSelection()
        people = self.spn_people.GetValue()
        on, off = tr("オン", "on"), tr("オフ", "off")
        # 繰り返し再生と基準音もログに残す。聞ける回数や基準音の有無で
        # 成績の比較条件が変わる
        repeat = on if self.repeat_enabled else off
        reference = on if self.reference_enabled else off
        if is_japanese():
            floor = "あり" if self.chk_floor.GetValue() else "なし"
            return (f"HRTF は{hrtf}、音は{sound}、首振りは{wobble}、"
                    f"首かしげは{roll}、"
                    f"上下の強調は{self.spn_tilt.GetValue()}パーセント、"
                    f"床の反射は{floor}、残響は{room}、空間は{space}"
                    f"（家具の量は{furnish}、人の数は{people}）"
                    f"{self._late_text()}、難易度は{diff}、"
                    f"前後の強調は{self.spn_depth.GetValue()}パーセント、"
                    f"繰り返し再生は{repeat}、"
                    f"基準音は{reference}（上下の聴き取りのみ）、"
                    f"いまの向きは{self._yaw:.0f}度、"
                    f"顔の向きは{self.pitch_text()}。")
        floor = "on" if self.chk_floor.GetValue() else "off"
        return (f"HRTF: {hrtf}. Sound: {sound}. Head wobble: {wobble}. "
                f"Head tilt: {roll}. "
                f"Up/down emphasis: {self.spn_tilt.GetValue()} percent. "
                f"Floor reflection: {floor}. Reverb: {room}. Space: {space} "
                f"(furniture amount: {furnish}, people: {people})"
                f"{self._late_text()}. Difficulty: {diff}. "
                f"Front/back emphasis: {self.spn_depth.GetValue()} percent. "
                f"Repeat: {repeat}. "
                f"Reference sound: {reference} (up/down tests only). "
                f"Facing {self._yaw:.0f} degrees, "
                f"face {self.pitch_text()}.")

    def _late_text(self) -> str:
        """空間の残響がどの作り方で鳴っているか（設定の読み上げとログ用）。"""
        if self.player is None or self.player.engine.space is None:
            return ""
        status = self.player.engine.late_status
        if not status:
            return ""
        return tr(f"（{status}）", f" ({status})")

    def apply_settings(self) -> None:
        if self.player is None:
            return
        engine = self.player.engine
        engine.front_back_depth = self.spn_depth.GetValue() / 100.0
        engine.elevation_emphasis = self.spn_tilt.GetValue() / 100.0
        space = self.space_keys[max(0, self.cmb_space.GetSelection())]
        # 空間は選択が変わったときだけ設定し直す。設定はモードを始めるたびに
        # 呼ばれるが、同じ空間を設定し直すと、切り替えのつなぎ（前の残響の
        # 尾と反射のクロスフェード）が毎回走ってしまう。自分で作った空間は
        # 中身（設計図）も比べる（同じ名前のまま直して保存したとき）
        precise = bool(self.chk_precise.GetValue())
        target = space
        applied_key = space
        if isinstance(space, tuple):
            spec = self.user_presets.spaces.get(space[1])
            if spec is None:                 # 消された（選択肢は作り直す）
                target, applied_key = None, None
            else:
                target = space_from_spec(spec)
                applied_key = (space, tuple(sorted(
                    (k, repr(v)) for k, v in spec.items())))
        elif space is not None:
            target = space_preset(space)
        if target is not None:
            extra = self.extra_contents(target)
            target = self.with_extra_contents(target, extra)
            applied_key = (applied_key, tuple(sorted(extra.items())))
        space_changed = (applied_key, precise) != self._applied_space
        if space_changed:
            # 空間は聞き手（原点、耳の高さ1.6m）の周りに置く。残響も空間の
            # 材質から決まるものに置き換わる。切り替えはなめらかにつなぐ
            self.player.set_space(target, late="precise" if precise else True)
            self._applied_space = (applied_key, precise)
        if target is None:
            # 残響は変わったときだけ作り直す（設定欄の数値を1つ動かすたびに
            # 作り直すと、鳴っている響きが途切れる）。空間を外した直後は
            # エンジンの残響が空なので、必ず設定する
            room = self.room_keys[max(0, self.cmb_room.GetSelection())]
            spec = (self.user_presets.reverbs.get(room[1])
                    if room is not None and room[0] == "user" else None)
            room_state = (room, repr(sorted(spec.items())) if spec else None)
            if space_changed or room_state != self._applied_room:
                if room is None:
                    self.player.set_reverb(None)
                elif spec is not None:
                    self.player.set_reverb(reverb_from_spec(
                        spec, engine.sample_rate))
                elif room[0] == "preset":
                    self.player.set_reverb(room[1])
                else:
                    self.player.set_reverb(None)
                self._applied_room = room_state
        else:
            self._applied_room = "unset"     # 残響は空間が持っている
        key = self.wobble_keys[max(0, self.cmb_wobble.GetSelection())]
        engine.set_head_wobble(WOBBLE_PRESETS[key])
        engine.set_roll_wobble(ROLL_PRESETS[self._roll_key()])
        engine.set_listener((0.0, 0.0, 0.0), yaw_deg=self._yaw,
                            pitch_deg=self._pitch)

    # -- 自分で作る空間と残響 -------------------------------------------
    def refresh_user_choices(self, room_key=None, space_key=None) -> None:
        """残響と空間の選択肢を作り直す（自分で作ったものを後ろに足す）。

        選んでいたものは選んだままにする。room_key / space_key を渡せば
        それを選ぶ（保存した直後に、作ったものを選ぶため）。
        """
        old_room = (self.room_keys[self.cmb_room.GetSelection()]
                    if getattr(self, "room_keys", None)
                    and self.cmb_room.GetSelection() >= 0 else None)
        old_space = (self.space_keys[self.cmb_space.GetSelection()]
                     if getattr(self, "space_keys", None)
                     and self.cmb_space.GetSelection() >= 0 else None)
        room_key = room_key if room_key is not None else old_room
        space_key = space_key if space_key is not None else old_space

        self.room_keys = ([None] + [("preset", n) for n in preset_names()]
                          + [("user", n) for n in self.user_presets.reverbs])
        self.cmb_room.Set([tr("なし", "None")] + preset_names()
                          + [f"{USER_PREFIX}{n}"
                             for n in self.user_presets.reverbs])
        self.cmb_room.SetSelection(self.room_keys.index(room_key)
                                   if room_key in self.room_keys else 0)

        self.space_keys = ([None] + [k for k, _ in space_labels()]
                           + [("user", n) for n in self.user_presets.spaces])
        self.cmb_space.Set([tr("なし", "None")] + [v for _, v in space_labels()]
                           + [f"{USER_PREFIX}{n}"
                              for n in self.user_presets.spaces])
        self.cmb_space.SetSelection(self.space_keys.index(space_key)
                                    if space_key in self.space_keys else 0)

    def extra_contents(self, space) -> dict:
        """設定欄の「家具の量」と「人の数」から、空間に足すものを決める。

        家具は閉じた部屋のときだけ足す（道や開けた場所は床がとても広く、
        床の広さに合わせると何十台ものソファになってしまう）。人はどこにでも
        足す。
        """
        extra = {}
        level = self.furnish_keys[max(0, self.cmb_furnish.GetSelection())]
        if level != "empty" and space.open_fraction == 0.0:
            extra.update(furnish_counts(space.size[0] * space.size[1], level))
        people = int(self.spn_people.GetValue())
        if people:
            extra["person"] = people
        return extra

    @staticmethod
    def with_extra_contents(space, extra: dict):
        """空間に、設定欄の家具と人（extra_contents）を足したもの。"""
        if not extra:
            return space
        merged = dict(space.contents)
        for k, n in extra.items():
            merged[k] = merged.get(k, 0) + n
        return space.with_contents(merged)

    def open_space_editor(self) -> None:
        if self._space_editor is None:
            self._space_editor = SpaceEditor(self)
            self._space_editor.Bind(
                wx.EVT_CLOSE, lambda e: self._close_editor("space"))
            # 設定欄で選んでいる空間を元にする（無ければ最初の空間）
            key = self.space_keys[max(0, self.cmb_space.GetSelection())]
            if isinstance(key, tuple):
                self._space_editor.select_source(key)
            elif key is not None:
                self._space_editor.select_source(("preset", key))
            else:
                self._space_editor.select_source(None)
        self._space_editor.Show()
        self._space_editor.Raise()
        self._space_editor.cmb_source.SetFocus()

    def open_reverb_editor(self) -> None:
        if self._reverb_editor is None:
            self._reverb_editor = ReverbEditor(self)
            self._reverb_editor.Bind(
                wx.EVT_CLOSE, lambda e: self._close_editor("reverb"))
            key = self.room_keys[max(0, self.cmb_room.GetSelection())]
            self._reverb_editor.select_source(key)
        self._reverb_editor.Show()
        self._reverb_editor.Raise()
        self._reverb_editor.cmb_source.SetFocus()

    def _close_editor(self, kind: str) -> None:
        if kind == "space":
            dlg, self._space_editor = self._space_editor, None
            button = self.btn_space_edit
        else:
            dlg, self._reverb_editor = self._reverb_editor, None
            button = self.btn_reverb_edit
        if dlg is not None:
            dlg.Destroy()
        self.end_preview()
        button.SetFocus()

    def preview_space(self, spec: dict) -> None:
        """編集中の空間で、正面3mの音を1回鳴らす（保存しない）。

        設定欄の「家具の量」と「人の数」も足す（保存した後に聞こえるのと
        同じ響きで聞けるように）。
        """
        if not self.ensure_player():
            return
        space = space_from_spec(spec)        # 問題があれば ValueError
        # 保存した後に聞こえるのと同じにする（設定欄の家具と人も足す）
        space = self.with_extra_contents(space, self.extra_contents(space))
        self._stop_repeat()
        self.stop_sound()
        self.player.set_space(space, late=True)
        self._applied_space = None           # 閉じたら選んでいるものに戻す
        self.play_at(self._yaw % 360.0, 3.0)  # いまの自分の正面

    def preview_reverb(self, spec: dict) -> None:
        """編集中の残響で、正面3mの音を1回鳴らす（空間は外して聞く）。"""
        if not self.ensure_player():
            return
        reverb = reverb_from_spec(spec, self.player.engine.sample_rate)
        self._stop_repeat()
        self.stop_sound()
        self.player.set_space(None)
        self.player.set_reverb(reverb)
        self._applied_space = None
        self.play_at(self._yaw % 360.0, 3.0)  # いまの自分の正面

    def end_preview(self) -> None:
        """試しに鳴らした響きから、設定欄で選んでいる響きに戻す。"""
        if self.player is not None and self._applied_space is None:
            self.apply_settings()

    def on_user_presets_changed(self, kind: str, name: str | None) -> None:
        """保存・削除の後。選択肢を作り直し、保存したものを選んで反映する。"""
        if kind == "space":
            self.refresh_user_choices(
                space_key=("user", name) if name else None)
        else:
            self.refresh_user_choices(
                room_key=("user", name) if name else None)
        if self.player is not None:
            self._applied_space = None       # 同じ名前を直したときも反映する
            self.apply_settings()


    def _roll_key(self) -> str:
        return self.roll_keys[max(0, self.cmb_roll.GetSelection())]

    def on_hrtf_changed(self, event: wx.CommandEvent) -> None:
        label, path = self.current_hrtf()
        if self.player is None:
            self.speech.say(tr(f"{label} に切り替えました。"
                               "次に音を鳴らすときに読み込みます。",
                               f"Switched to {label}. It will load when the "
                               "next sound plays."))
        elif self._active_tag is not None:
            # 耳介の選定で候補の耳介を読み込んでいる間。ここで差し替えると、
            # 残りの問題が選んだ HRTF で鳴るのに、成績は候補の耳介のものに
            # なる。終わったときに設定欄の HRTF へ戻す。まだ候補を読み込む
            # 前（案内の間）は、下の通常の切り替えを通す（ここで止めると、
            # そのまま別のモードへ移ったとき、戻す処理が働かず食い違う）
            self.speech.say(tr(f"{label} を選びました。耳介の選定中なので、"
                               "選定が終わってから使います。",
                               f"Chose {label}. A pinna choice is running, "
                               "so it will be used after it ends."))
        else:
            try:
                self.player.engine.set_hrir(self.load_hrir(path))
                self.speech.say(tr(f"{label} に切り替えました。",
                                   f"Switched to {label}."))
            except Exception as exc:  # noqa: BLE001
                self.speech.error(tr(f"HRTF を読み込めません。{exc}",
                                     f"Cannot load the HRTF. {exc}"))
        event.Skip()

    def on_setting_changed(self, event: wx.CommandEvent) -> None:
        self.apply_settings()
        event.Skip()

    # -- 音声エンジン --------------------------------------------------
    def load_hrir(self, path: Path | None) -> HrirSet:
        """HRTF を読んでキャッシュする。

        SOFA の読み込みは約200ミリ秒かかる。耳介の選定では何度も
        切り替えるので、一度読んだものは保持する。
        """
        rate = (self.player.engine.sample_rate if self.player else 48000)
        key = f"{path}|{rate}"
        if key not in self._hrir_cache:
            self._hrir_cache[key] = (
                HrirSet.from_sofa(path, sample_rate=rate) if path
                else HrirSet.synthetic(sample_rate=rate))
        return self._hrir_cache[key]

    def ensure_player(self) -> bool:
        if self.player is not None:
            return True
        label, sofa = self.current_hrtf()
        try:
            self.log(tr(f"{label} を読み込んでいます。少し待ってください。",
                        f"Loading {label}. Please wait a moment."))
            wx.SafeYield()
            # sofa_path=None は open() では「既定（実測アセット）を使う」の
            # 意味になる。合成HRIRを選んだときは明示的に渡さないと、
            # 「合成」と表示しながら実測で鳴らしてしまう。
            hrir = None if sofa is not None else self.load_hrir(None)
            self.player = SpatialPlayer.open(
                block=512, max_sources=8, hrir=hrir, sofa_path=sofa,
                reverb=None)
            self.bank = SoundBank(self.player.engine.sample_rate)
            self.apply_settings()
            hrir = self.player.engine.hrir
            # 無音モード（AUDIOSPACE_SILENT=1）ではデバイスを選ばないので
            # choice が None になる。ここで落ちるとテストでアプリを開けない。
            if self.player.choice is None:
                device = tr("無音モード（デバイスを開いていません）",
                            "Silent mode (no device is open)")
            else:
                device = self.player.choice.describe()
            self.log(tr(f"準備できました。{label}、{hrir.n_directions}方向。"
                        f"{device}",
                        f"Ready. {label}, {hrir.n_directions} directions. "
                        f"{device}"))
            return True
        except Exception as exc:  # noqa: BLE001
            self.speech.error(tr(f"音声の初期化に失敗しました。{exc}",
                                 f"Starting the audio failed. {exc}"))
            self.txt_log.SetValue(tr(f"[エラー] 音声の初期化に失敗しました。\n{exc}",
                                     f"[Error] Starting the audio failed.\n"
                                     f"{exc}"))
            return False

    def stop_sound(self) -> None:
        if self._orbit_timer is not None:
            self._orbit_timer.Stop()
            self._orbit_timer = None
        self._rot_timer.Stop()
        self._rot_trial = None
        if self.player is not None:
            for handle in (self._handle, self._ref_handle,
                           *self._extra_handles):
                if handle is None:
                    continue
                try:
                    self.player.stop_sound(handle)
                except Exception:  # noqa: BLE001
                    pass
        self._handle = None
        self._ref_handle = None
        self._extra_handles = []

    def _sound_kind(self) -> str:
        return SOUND_KINDS[max(0, self.cmb_sound.GetSelection())][1]

    def play_at(self, azimuth_deg: float, distance_m: float = 4.0,
                loop: bool = False, elevation_deg: float = 0.0,
                with_reference: bool = False) -> None:
        """その方向で音を鳴らす。

        `with_reference` なら、先に同じ方位・水平の基準音を鳴らし、
        REFERENCE_GAP_S の間を置いて本来の音を鳴らす。
        床の反射は、ループしない音（出題・聞き比べ）にだけ付ける。試聴
        （F2）の回る音は位置が動き続けるので、反射音の位置と遅れを固定
        できない。上下の強調はエンジンがかけるので、回る音にもかかる。
        """
        if not self.ensure_player():
            return
        self.stop_sound()
        kind = self._sound_kind()
        lead = 0.0
        if with_reference:
            self._ref_handle = self._play_one(
                kind, azimuth_deg, distance_m, 0.0, lead, loop=False)
            lead = (len(self.bank.get(kind)) / self.bank.sample_rate
                    + REFERENCE_GAP_S)
        self._handle = self._play_one(kind, azimuth_deg, distance_m,
                                      elevation_deg, lead, loop)

    def _play_one(self, kind: str, azimuth_deg: float, distance_m: float,
                  elevation_deg: float, lead_s: float, loop: bool) -> int:
        """1つの音（と、床の反射が有効ならその反射音）を鳴らす。

        返すのは直接音の音源。反射音は _extra_handles に積み、
        stop_sound() で一緒に止める。
        """
        position = position_for(azimuth_deg, distance_m, elevation_deg)
        handle = self.player.play(
            self.bank.variant(kind, delay_s=lead_s),
            position=position, loop=loop, reverb_send=0.3)
        if self.chk_floor.GetValue() and not loop:
            image, delay = floor_image(position)
            if image is not None:
                self._extra_handles.append(self.player.play(
                    self.bank.variant(kind, delay_s=lead_s + delay),
                    position=image, gain=FLOOR_REFLECTION_GAIN,
                    # 残響は直接音の分だけで足りる。反射音にも送ると
                    # 残響が二重になる
                    reverb_send=0.0))
        return handle

    # -- 基準音 ---------------------------------------------------------
    def _reference_applies(self) -> bool:
        """いまの出題に基準音を付けるか。上下の聴き取りだけが対象。"""
        return (self.reference_enabled
                and isinstance(self.session, ElevationSession))

    def toggle_reference(self) -> None:
        self.reference_enabled = not self.reference_enabled
        if self.reference_enabled:
            self.speech.say(tr("基準音をオンにしました。上下の聴き取りでは、"
                               "先に水平の音、続けて問題の音が鳴ります。",
                               "Reference sound on. In the up/down tests, a "
                               "level sound plays first, then the question "
                               "sound."))
            self.SetStatusText(tr("基準音 オン", "Reference sound on"))
        else:
            self.speech.say(tr("基準音をオフにしました。問題の音だけが鳴ります。",
                               "Reference sound off. Only the question sound "
                               "plays."))
            self.SetStatusText(tr("基準音 オフ", "Reference sound off"))
        # 1回分の長さが変わるので、繰り返しの予約を張り直す
        self._arm_repeat()

    # -- 繰り返し再生 ---------------------------------------------------
    def _current_sound_seconds(self) -> float:
        """1回分の出題の長さ。基準音つきなら基準音と間を含む。"""
        if self.bank is None:
            return 0.0
        if isinstance(self.session, RotationSession):
            return ROTATION_ARC_SECONDS
        one = len(self.bank.get(self._sound_kind())) / self.bank.sample_rate
        if self._reference_applies():
            return one * 2 + REFERENCE_GAP_S
        return one

    def repeat_interval_ms(self) -> int:
        """次に鳴らし直すまでの時間。音の長さに間を足す。

        間が短いと、鳴り終わりと次の鳴り始めがつながって1つの長い音に
        聞こえ、どこからが1回分か分からなくなる。
        """
        gap = (REFERENCE_REPEAT_GAP_S if self._reference_applies()
               else REPEAT_GAP_S)
        seconds = max(REPEAT_MIN_S, self._current_sound_seconds() + gap)
        return int(seconds * 1000)

    def _waiting_for_answer(self) -> bool:
        return (self._accepting and self.session is not None
                and not self.session.finished)

    def _arm_repeat(self) -> None:
        """出題音を鳴らした直後に呼ぶ。回答待ちの間だけ次を予約する。"""
        self._repeat_timer.Stop()
        if self.repeat_enabled and self._waiting_for_answer():
            self._repeat_timer.StartOnce(self.repeat_interval_ms())

    def _stop_repeat(self) -> None:
        self._repeat_timer.Stop()

    def on_repeat(self, event: wx.TimerEvent) -> None:
        # 予約してから鳴るまでの間に回答・中止・モード切替があり得るので、
        # 鳴らす直前にもう一度確かめる
        if not (self.repeat_enabled and self._waiting_for_answer()):
            return
        self._play_trial(self.session.current())
        self._arm_repeat()

    def toggle_repeat(self) -> None:
        self.repeat_enabled = not self.repeat_enabled
        if self.repeat_enabled:
            self.speech.say(tr("繰り返し再生をオンにしました。"
                               "答えるまで出題音を鳴らし続けます。",
                               "Repeat on. The question sound keeps playing "
                               "until you answer."))
            self._arm_repeat()
            self.SetStatusText(tr("繰り返し再生 オン", "Repeat on"))
        else:
            self._stop_repeat()
            self.speech.say(tr("繰り返し再生をオフにしました。"
                               "出題音は1回だけ鳴ります。F4 でもう一度聞けます。",
                               "Repeat off. The question sound plays once. "
                               "F4 plays it again."))
            self.SetStatusText(tr("繰り返し再生 オフ", "Repeat off"))

    # -- キー操作 ------------------------------------------------------
    def on_key(self, event: wx.KeyEvent) -> None:
        code = event.GetKeyCode()
        # 上下の聴き取りは Shift+F3（F3 のクイズの上下版）。以前の F12 は
        # PC-Talker のショートカットと衝突した。Ctrl+F3 は正面固定の上下。
        # 上下用の耳介の選定は Ctrl+F8（F8 は前後用。F3 と Ctrl+F3 の関係に揃えた）
        if code == wx.WXK_F8 and event.ControlDown():
            self.start_calibration_elev()
            return
        # 8方向の聞き比べは Ctrl+Shift+F2（利用者の指定。Shift+F2 の上下の
        # 聞き比べの水平版）。Ctrl+F2 と Shift+F2 より先に調べる
        if (code == wx.WXK_F2 and event.ControlDown()
                and event.ShiftDown()):
            self.start_listen8()
            return
        # 回る向きの聴き取りは Ctrl+F2（F2 の試聴の採点版。利用者が選んだ）
        if code == wx.WXK_F2 and event.ControlDown():
            self.start_rotation()
            return
        if code == wx.WXK_F3 and event.ControlDown():
            self.start_elevation_front()
            return
        if code == wx.WXK_F3 and event.ShiftDown():
            self.start_elevation()
            return
        # 聞き比べは Shift+F2（F2 の試聴の仲間。採点せずに音を聞く）
        if code == wx.WXK_F2 and event.ShiftDown():
            self.start_listen()
            return
        if code == wx.WXK_NUMPAD0:
            self.toggle_repeat()
            return
        if code == wx.WXK_NUMPAD_MULTIPLY:
            self.toggle_reference()
            return
        if self.mode == "listen":
            el = self._elevation_for_key(code)
            if el is not None:
                self.listen_at(el)
                return
        if self.mode == "listen8":
            az = self._azimuth_for_key(code)
            if az is not None:
                self.listen_direction(az)
                return
        # 顔の上下は利用者の指定で Page Up / Page Down。一覧や数値のスピンも
        # Page Up / Down を使うが、それらは上下の矢印でも動かせる（上下の
        # 矢印は横取りしていない）。テンキーの 9 と 3（NumLock オフで
        # Page Up / Down になる）は回答のキーなので対象にしない。
        if code in (wx.WXK_PAGEUP, wx.WXK_PAGEDOWN):
            self.tilt(code)
            return
        actions = {
            wx.WXK_ESCAPE: self.Close,
            wx.WXK_F1: self.show_help,
            wx.WXK_F2: self.toggle_orbit,
            wx.WXK_F3: self.start_quiz,
            wx.WXK_F4: self.replay,
            wx.WXK_F5: self.announce_result,
            wx.WXK_F6: self.stop_and_say,
            wx.WXK_F7: self.start_training,
            wx.WXK_F8: self.start_calibration,
            wx.WXK_F9: self.announce_settings,
            wx.WXK_F10: self.announce_records,
            wx.WXK_F11: self.start_ab,
        }
        if code in actions:
            actions[code]()
            return
        # 上下の矢印は横取りしない。以前は上矢印を「正面に向き直す」に
        # 使っていたが、パネルの CHAR_HOOK はフォーカス中のコントロールより
        # 先に届くので、選択肢を上下で選ぶたびに選択内容より先に「向き0度」と
        # 読み上げてしまった。上下の割り当ては今後決める。
        if code in (wx.WXK_LEFT, wx.WXK_RIGHT):
            self.turn(code)
            return
        if self.try_answer(code):
            return
        event.Skip()

    def try_answer(self, code: int) -> bool:
        """テンキー回答の受付判定。受理したら True。

        出題音が鳴る前（回答直後の読み上げ待ちを含む）は受け付けない。
        ここを開けておくと、キーの二度押しがまだ聞いていない次の問題への
        回答になり、画面の見えない利用者には起きたことに気づく手段がない。
        """
        if self.session is None or self.session.finished:
            return False
        if not self._accepting:
            return False
        if isinstance(self.session, ElevationSession):
            el = self._elevation_for_key(code)
            if el is None:
                return False
            self.submit_elevation(el)
            return True
        if isinstance(self.session, RotationSession):
            motion = self._rotation_for_key(code)
            if motion is None:
                return False
            if (motion in ("to_front", "to_back")
                    and not self.session.include_overhead):
                # 水平だけの回で頭上のキーを押しても、不正解にはしない
                self.speech.say(tr("この回は水平に回る音だけです。"
                                   "4で左回り、6で右回りと答えてください。",
                                   "This round has only level circles. "
                                   "Answer 4 for counterclockwise or 6 for "
                                   "clockwise."))
                return True
            self.submit_rotation(motion)
            return True
        az = self._azimuth_for_key(code)
        if az is None:
            return False
        self.submit(az)
        return True

    @staticmethod
    def _azimuth_for_key(code: int) -> float | None:
        """テンキーの並びを方向に対応させる。

        文字キーは IME と衝突するので使わない。矢印キーは向きを変える方に
        割り当てたので、回答はテンキー専用にしてある。
        """
        return {
            wx.WXK_NUMPAD8: 0.0,
            wx.WXK_NUMPAD7: 45.0,
            wx.WXK_NUMPAD4: 90.0,
            wx.WXK_NUMPAD1: 135.0,
            wx.WXK_NUMPAD2: 180.0,
            wx.WXK_NUMPAD3: 225.0,
            wx.WXK_NUMPAD6: 270.0,
            wx.WXK_NUMPAD9: 315.0,
        }.get(code)

    @staticmethod
    def _elevation_for_key(code: int) -> float | None:
        """上下の回答。テンキーの縦の並び（/ が真上、8 が上、5 が水平、
        2 が下）。ELEVATION_LEVELS は上から順なので、そのまま対応させる。"""
        keys = (wx.WXK_NUMPAD_DIVIDE, wx.WXK_NUMPAD8,
                wx.WXK_NUMPAD5, wx.WXK_NUMPAD2)
        table = {k: el for k, (el, _) in zip(keys, ELEVATION_LEVELS)}
        return table.get(code)

    @staticmethod
    def _rotation_for_key(code: int) -> str | None:
        """回る向きの回答。音が進む向きをキーの位置で表す（利用者が選んだ）。"""
        return {
            wx.WXK_NUMPAD4: "left",
            wx.WXK_NUMPAD6: "right",
            wx.WXK_NUMPAD8: "to_front",
            wx.WXK_NUMPAD2: "to_back",
        }.get(code)

    def turn(self, code: int) -> None:
        """自分の向きを変える。前後を見分ける最も確実な手段。"""
        if code == wx.WXK_LEFT:
            self._yaw = (self._yaw + TURN_STEP_DEG) % 360.0
        else:
            self._yaw = (self._yaw - TURN_STEP_DEG) % 360.0
        if self.player is not None:
            self.player.engine.set_listener((0.0, 0.0, 0.0),
                                            yaw_deg=self._yaw)
        self.SetStatusText(tr(f"向き {self._yaw:.0f}度",
                              f"Facing {self._yaw:.0f} degrees"))
        self.speech.say(tr(f"向き{self._yaw:.0f}度",
                           f"Facing {self._yaw:.0f} degrees"))

    def pitch_text(self) -> str:
        """顔の上下の読み上げ名。水平・上15度・真上・下30度・真下など。"""
        p = round(self._pitch)
        if p == 0:
            return tr("水平", "level")
        if p >= PITCH_LIMIT_DEG:
            return tr("真上", "straight up")
        if p <= -PITCH_LIMIT_DEG:
            return tr("真下", "straight down")
        if p > 0:
            return tr(f"上{p}度", f"up {p} degrees")
        return tr(f"下{-p}度", f"down {-p} degrees")

    def tilt(self, code: int) -> None:
        """顔を上下に向ける。Page Up で上、Page Down で下。音源は動かない。"""
        step = PITCH_STEP_DEG if code == wx.WXK_PAGEUP else -PITCH_STEP_DEG
        new = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG, self._pitch + step))
        if new == self._pitch:
            # 端で押し続けたとき、同じ角度を繰り返し読むと動いたのか
            # 分からない。端であることを伝える。
            self.speech.say(tr(f"顔の向き{self.pitch_text()}。"
                               "これ以上は向けません。",
                               f"Face {self.pitch_text()}. It cannot turn "
                               "further."))
            return
        self._pitch = new
        if self.player is not None:
            self.player.engine.set_listener(pitch_deg=self._pitch)
        self.SetStatusText(tr(f"顔の向き {self.pitch_text()}",
                              f"Face {self.pitch_text()}"))
        self.speech.say(tr(f"顔の向き{self.pitch_text()}",
                           f"Face {self.pitch_text()}"))

    def show_help(self) -> None:
        """キー操作の説明の窓を開く（開いていれば前に出す）。

        窓のタイトルと本文の行はスクリーンリーダーが読むので、ここでは
        読み上げない（重なって聞き取れなくなる）。閉じると元の場所に戻る。
        """
        self._stop_repeat()
        self.stop_sound()
        if self._help_dlg is None:
            self._help_return = wx.Window.FindFocus()
            self._help_dlg = HelpDialog(self, HELP_TEXT)
            self._help_dlg.Bind(wx.EVT_CLOSE, self._on_help_close)
        self._help_dlg.show_text()

    def _on_help_close(self, event: wx.CloseEvent) -> None:
        dlg, self._help_dlg = self._help_dlg, None
        if dlg is not None:
            dlg.Destroy()
        back = getattr(self, "_help_return", None)
        self._help_return = None
        # 元のコントロールへ戻す。無ければ状況の欄へ
        target = back if (back is not None and back.IsShownOnScreen()
                          and back.GetTopLevelParent() is self) else self.txt_log
        target.SetFocus()

    def stop_and_say(self) -> None:
        # 繰り返しも止める。止めないと、F6 の直後にまた鳴り出す
        repeating = self._repeat_timer.IsRunning()
        self._stop_repeat()
        self.stop_sound()
        if self.mode in ("listen", "listen8"):
            # 聞き比べは「F6 か別のモードのキーで終わる」と案内している。
            # 抜けないと、F6 の後もテンキーで音が鳴り続ける
            self.mode = "idle"
            self.speech.say(tr("聞き比べを終えました。",
                               "Finished comparing."))
            return
        if repeating:
            self.speech.say(tr("音を止めました。F4 でもう一度鳴らします。",
                               "Sound stopped. F4 plays it again."))
        else:
            self.speech.say(tr("音を止めました。", "Sound stopped."))

    def announce_settings(self) -> None:
        self.speech.say(self.settings_text())

    def _pinna_pool(self) -> list[tuple[str, str]]:
        return [(str(p), label)
                for label, p in self.hrtf_choices if p is not None]

    def announce_records(self) -> None:
        """耳介の通算成績。セッションをまたいで積み上がる。

        上下用の記録があれば続けて読む。キーを増やさないため F10 に
        まとめた（前後用が先、上下用が後）。
        """
        pool = self._pinna_pool()
        text = tr("前後の通算成績。", "Front/back totals. ") + \
            self.records.summary(pool)
        lines = self.records.detail_lines()
        if self.records_elev.records:
            text += tr(" 上下の通算成績。", " Up/down totals. ") + \
                self.records_elev.summary(pool)
            elev_lines = self.records_elev.detail_lines()
            if elev_lines:
                lines = lines + [tr("--- 上下 ---", "--- Up/down ---")] + \
                    elev_lines
        self.speech.say(text)
        if lines:
            self.txt_log.SetValue(text + "\n\n" + "\n".join(lines))

    # -- 聞き比べ -------------------------------------------------------
    def start_listen(self) -> None:
        """上下の聞き比べ。テンキーの / 8 5 2 でその高さの音が鳴るだけ。

        採点しないので、テスト前に4つの高さを好きな順に何度でも聞ける。
        音は正面（方位0度）に置く。左右の矢印で自分が回れば、横や後ろから
        聞いた上下も試せる。Page Up / Down で顔を上下に向けても試せる。
        """
        if not self.ensure_player():
            return
        if self._refuse_without_elevation("Shift+F2"):
            return
        self._stop_repeat()
        self.stop_sound()
        self.apply_settings()
        self.session = None
        self.mode = "listen"
        self._accepting = False
        self._begin_gen += 1        # 直前に始めたモードの遅延出題を無効にする
        self.lst_result.Set([])
        self.log(tr("聞き比べを開始しました。採点はしません。"
                    "テンキーのスラッシュで真上、8で上、5で水平、2で下の音が"
                    "正面で鳴ります。押したときは読み上げないので、音に集中して"
                    "ください。左右の矢印で回ると、横や後ろから聞いた上下も"
                    "試せます。終わるときは F6 か、別のモードのキーを押します。",
                    "Comparing heights. Not scored. Numpad slash plays a "
                    "sound straight up, 8 up, 5 level and 2 down, in front "
                    "of you. Keys do not speak, so you can focus on the "
                    "sound. Turn with the left and right arrows to try "
                    "heights from the side or behind. To finish, press F6 "
                    "or the key of another mode."))

    def listen_at(self, elevation_deg: float) -> None:
        """聞き比べで1つの高さを鳴らす。

        押すたびに高さを読み上げると、その声が音にかぶって聞き取りの
        妨げになる（繰り返し再生を作った理由と同じ）。押したキーで高さは
        分かっているので、読み上げずにステータスバーにだけ書く。
        """
        self.play_at(0.0, 4.0, elevation_deg=elevation_deg)
        self.SetStatusText(tr(f"聞き比べ {elevation_name(elevation_deg)}",
                              f"Comparing: {elevation_name(elevation_deg)}"))

    def start_listen8(self) -> None:
        """8方向の聞き比べ。テンキーの 1〜9（5 を除く）でその方向の音が鳴る。

        採点しない。クイズの前に、前と後ろ・斜めの聞こえ方の違いを好きな
        順に何度でも聞ける。HRTF を問わず使える（水平だけなので）。
        """
        if not self.ensure_player():
            return
        self._stop_repeat()
        self.stop_sound()
        self._restore_selected_hrir()
        self.apply_settings()
        self.session = None
        self.mode = "listen8"
        self._accepting = False
        self._begin_gen += 1        # 直前に始めたモードの遅延出題を無効にする
        self.lst_result.Set([])
        self.log(tr("8方向の聞き比べを開始しました。採点はしません。"
                    "テンキーの1から9で、その方向の音が鳴ります。5は使いません。"
                    "キーの並びがそのまま方向の配置で、8で前、2で後ろ、4で左、"
                    "6で右です。押したときは読み上げないので、音に集中して"
                    "ください。左右の矢印で回っても、方向はいまの自分から見た"
                    "向きです。終わるときは F6 か、別のモードのキーを押します。",
                    "Comparing 8 directions. Not scored. Numpad 1 to 9 play "
                    "a sound from that direction; 5 is not used. The keys "
                    "are laid out like the directions: 8 front, 2 back, "
                    "4 left, 6 right. Keys do not speak, so you can focus on "
                    "the sound. Even after turning with the left and right "
                    "arrows, directions are as seen from where you face now. "
                    "To finish, press F6 or the key of another mode."))

    def listen_direction(self, azimuth_deg: float) -> None:
        """8方向の聞き比べで1つの方向を鳴らす（いまの自分から見た向き）。

        listen_at と同じく、押すたびには読み上げない（声が音にかぶる）。
        """
        self.play_at((azimuth_deg + self._yaw) % 360.0, 4.0)
        self.SetStatusText(tr(f"聞き比べ {direction_name(azimuth_deg)}",
                              f"Comparing: {direction_name(azimuth_deg)}"))

    # -- 試聴 ---------------------------------------------------------
    def toggle_orbit(self) -> None:
        if self._orbit_timer is not None:
            self.stop_sound()
            self.log(tr("試聴を止めました。", "Preview stopped."))
            return
        if not self.ensure_player():
            return
        self._restore_selected_hrir()
        self.apply_settings()
        self.session = None
        self._begin_gen += 1        # 直前のモードの遅延出題を無効にする
        self.mode = "orbit"
        self._orbit_angle = 0.0
        self.play_at(0.0, loop=True)
        self._orbit_timer = self._orbit_obj
        self._orbit_timer.Start(50)
        self.log(tr("試聴を開始しました。音が正面から左回りに一周します。"
                    "もう一度 F2 で停止します。",
                    "Preview started. The sound circles counterclockwise "
                    "from the front. Press F2 again to stop."))

    def on_orbit(self, event: wx.TimerEvent) -> None:
        if self.player is None or self._handle is None:
            return
        self._orbit_angle = (self._orbit_angle + 1.5) % 360.0
        try:
            self.player.engine.set_source_position(
                self._handle, position_for(self._orbit_angle, 4.0))
        except KeyError:
            self.stop_sound()
            return
        if abs(self._orbit_angle % 45.0) < 1.5:
            self.SetStatusText(tr(f"いま {direction_name(self._orbit_angle)} "
                                  f"({self._orbit_angle:.0f}度)",
                                  f"Now {direction_name(self._orbit_angle)} "
                                  f"({self._orbit_angle:.0f} degrees)"))

    # -- 各モードの開始 -------------------------------------------------
    def _restore_selected_hrir(self) -> None:
        """耳介の選定で差し替えた HRTF を、設定欄で選んでいる HRTF に戻す。

        選定（F8 / Ctrl+F8）は問題ごとに耳介を切り替え、終わったときは
        最後に試した耳介のままになる。戻さないと、次のモードはその耳介で
        鳴るのに、設定の読み上げとログには設定欄の耳介（例: 耳介E）が
        書かれ、成績が別の耳介のものとして残ってしまう。途中でほかの
        モードへ移ったときも同じ。
        """
        if self._active_tag is None:
            return
        self._active_tag = None
        if self.player is None:
            return
        label, path = self.current_hrtf()
        try:
            self.player.engine.set_hrir(self.load_hrir(path))
        except Exception as exc:  # noqa: BLE001
            self.speech.error(tr(f"{label} に戻せませんでした。{exc}",
                                 f"Could not switch back to {label}. {exc}"))

    def _begin(self, mode: str, session: QuizSession, intro: str) -> None:
        self._stop_repeat()
        self.stop_sound()
        self._restore_selected_hrir()
        self.apply_settings()
        self.mode = mode
        self.session = session
        self._accepting = False
        self.lst_result.Set([])
        self.log(intro)
        # 開始をすぐ連打で切り替えたとき、前のモードの遅延出題が新しい
        # セッションを鳴らさないよう、世代番号で無効化する
        self._begin_gen += 1
        self._pending_next = False
        gen = self._begin_gen
        wx.CallLater(2800, lambda: self._play_if_current(gen))

    def _later(self, ms: int, fn) -> None:
        """遅らせて動かす。待つ間に別のモードが始まっていたら動かさない
        （前のモードの音や出題が、新しいモードにかぶらないように）。"""
        gen = self._begin_gen
        wx.CallLater(ms, lambda: fn() if gen == self._begin_gen else None)

    def _play_if_current(self, gen: int) -> None:
        if gen == self._begin_gen:
            self.play_current()

    def start_quiz(self) -> None:
        if not self.ensure_player():
            return
        diff = self.difficulties[max(0, self.cmb_diff.GetSelection())]
        # 各方向が最低2回は出るようにする。8方向で10問だと1回ずつしか
        # 出ず、たまたま当たった／外れたの影響が大きくなる。
        n = max(10, len(diff.angles) * 2)
        self._begin("quiz", QuizSession(difficulty=diff, n_trials=n),
                    tr(f"クイズを開始します。{diff.label}、全{n}問。"
                       f"{self.current_hrtf()[0]}を使います。"
                       "音が鳴ったら方向をテンキーで答えてください。"
                       "8が前、4が左、6が右、2が後ろです。",
                       f"Starting the quiz. {diff.label}, {n} questions. "
                       f"Using {self.current_hrtf()[0]}. When the sound "
                       "plays, answer the direction with the numeric "
                       "keypad: 8 front, 4 left, 6 right, 2 back."))

    def start_training(self) -> None:
        if not self.ensure_player():
            return
        diff = self.difficulties[max(0, self.cmb_diff.GetSelection())]
        streak = max(8, len(diff.angles))
        self._begin("training",
                    TrainingSession(difficulty=diff, target_streak=streak),
                    tr(f"訓練を開始します。{diff.label}。"
                       "間違えた方向を重点的に出し直します。"
                       "間違えたときは正解の方向をもう一度鳴らすので、"
                       f"違いをよく聞いてください。{streak}問連続で正解すると"
                       "終わりです。",
                       f"Starting training. {diff.label}. Directions you "
                       "miss come back more often. After a wrong answer the "
                       "right direction plays again, so listen for the "
                       f"difference. It ends after {streak} correct "
                       "answers in a row."))

    def start_calibration(self) -> None:
        if not self.ensure_player():
            return
        diff = self.difficulties[max(0, self.cmb_diff.GetSelection())]
        # 各耳介で全方向を最低2回。方向が増えるほど1耳介あたりが長く
        # なるので、枠数を減らして総問数をおよそ50問に抑える。
        per = max(8, len(diff.angles) * 2)
        slots = max(2, min(5, 48 // per))
        pool = [(str(p), label)
                for label, p in self.hrtf_choices if p is not None]
        if len(pool) < 2:
            self.speech.error(NOT_ENOUGH_HRTF)
            return

        lineup, anchor = self.records.build_lineup(pool, slots)
        self._anchor_tag = anchor
        session = CalibrationSession(lineup, difficulty=diff,
                                     trials_per_candidate=per)
        anchor_text = ""
        if anchor:
            name = dict(pool)[anchor]
            anchor_text = tr(f"うち{name}は現在の最良で、比較の基準として"
                             "毎回入れています。回を重ねると聞き手も上達するので、"
                             "同じ回の中で比べないと差が分かりません。",
                             f"{name} is the current best and is included "
                             "every time as the reference. Listeners improve "
                             "over time, so differences only show within "
                             "one round.")
        if len(diff.angles) > 4:
            warn = ""
        else:
            warn = tr("なお、前後左右の4方向では前後の取り違えを検出できません。"
                      "前後を試すなら8方向以上を選んでください。",
                      "Note: with the 4 directions front, back, left and "
                      "right, front/back confusions cannot be detected. "
                      "To test front and back, choose 8 directions or more.")
        self._begin("calib", session, join_sentences(
            tr(f"耳介の選定を開始します。{len(lineup)}種類を"
               f"それぞれ{per}問ずつ、全{session.n_trials}問。",
               f"Starting to choose a pinna. {len(lineup)} pinnae, {per} "
               f"questions each, {session.n_trials} in all."),
            anchor_text, self.records.progress_text(pool), warn))

    def start_calibration_elev(self) -> None:
        """上下用の耳介の選定（Ctrl+F8）。前後用（F8）の上下版。

        出題は正面固定の上下4択。耳介ごとに12問（4つの高さを3回ずつ）、
        4種類で48問。前後用と同じく、現在の最良（上下用の記録の）を
        基準として毎回同席させ、通算は別のファイルに積み上げる。
        """
        if not self.ensure_player():
            return
        pool = self._pinna_pool()
        if len(pool) < 2:
            self.speech.error(NOT_ENOUGH_HRTF)
            return
        per = len(ELEVATION_LEVELS) * 3
        slots = max(2, min(5, 48 // per))
        lineup, anchor = self.records_elev.build_lineup(pool, slots)
        self._anchor_tag = anchor
        session = ElevationCalibrationSession(lineup,
                                              trials_per_candidate=per)
        anchor_text = ""
        if anchor:
            anchor_text = tr(f"うち{dict(pool)[anchor]}は上下の現在の最良で、"
                             "比較の基準として毎回入れています。",
                             f"{dict(pool)[anchor]} is the current best for "
                             "up and down and is included every time as the "
                             "reference.")
        self._begin("calib_elev", session, join_sentences(
            tr(f"上下用の耳介の選定を開始します。{len(lineup)}種類を"
               f"それぞれ{per}問ずつ、全{session.n_trials}問。"
               "音は毎回正面から鳴り、変わるのは高さだけです。",
               f"Starting to choose a pinna for up and down. "
               f"{len(lineup)} pinnae, {per} questions each, "
               f"{session.n_trials} in all. The sound always comes from "
               "the front, and only the height changes."),
            anchor_text, self.records_elev.progress_text(pool),
            self._elevation_answer_text()))

    def start_ab(self) -> None:
        """首振りの効果測定。ありなしを1問ごとに交互に出して比べる。

        前後の取り違えが減ったのが首振りの効果か練習の効果かは、別々の
        セッションでは切り分けられない（聞き手は上達していくため）。
        同一セッション内で交互に混ぜれば、上達は両条件に等しく効く。
        """
        if not self.ensure_player():
            return
        diff = self.difficulties[max(0, self.cmb_diff.GetSelection())]
        n = max(16, len(diff.angles) * 4)   # 各条件で全方向がおよそ2回ずつ
        key = self.wobble_keys[max(0, self.cmb_wobble.GetSelection())]
        if not WOBBLE_PRESETS[key].enabled:
            key = "clear"                   # 「なし」対「なし」では測れない
        self._ab_wobble_key = key
        self._begin("ab", ABWobbleSession(difficulty=diff, n_trials=n),
                    tr(f"首振りの効果測定を開始します。{diff.label}、全{n}問。"
                       f"首振り{dict(wobble_labels())[key]}の問題と、"
                       "首振りなしの問題を1問ごとに交互に出します。"
                       "どちらの条件かは出題時には言いません。"
                       "実施中は矢印キーでの向きの変更を控えてください。"
                       "音が鳴ったら方向をテンキーで答えてください。",
                       f"Starting to measure the head wobble effect. "
                       f"{diff.label}, {n} questions. Questions with head "
                       f"wobble {dict(wobble_labels())[key]} and without "
                       "head wobble alternate. It does not say which is "
                       "which. Please avoid turning with the arrow keys "
                       "during the test. When the sound plays, answer the "
                       "direction with the numeric keypad."))

    def hrir_supports_elevation(self) -> bool:
        """いまの HRTF が上下の出題に使えるか。

        合成 HRIR は水平面だけ、既定アセットも水平面だけに絞ってある。
        それで上下を出題すると、エンジンは最も近い水平の方向で鳴らすので
        上も下も水平と同じ音になる。成績が当て推量に張り付くだけで、
        利用者には理由が分からないので、始める前に断る。
        """
        if self.player is None:
            return False
        el = self.player.engine.hrir.elevation
        need = [lvl for lvl, _ in ELEVATION_LEVELS]
        return all(float(np.min(np.abs(el - lvl))) <= 5.0 for lvl in need)

    def _refuse_without_elevation(self, key_name: str) -> bool:
        """上下の測定がない HRTF なら断って True を返す。

        耳介の選定の後なら、先に設定欄の HRTF へ戻してから調べる。戻す前に
        調べると、残っていた選定用の耳介（全天球）で通ってしまい、その後で
        戻した HRTF（例えば合成）には上下が無い、ということが起きる。
        """
        self._restore_selected_hrir()
        if self.hrir_supports_elevation():
            return False
        self._stop_repeat()
        self.stop_sound()
        self.session = None
        self.mode = "idle"
        self._begin_gen += 1        # 直前のモードの遅延出題を無効にする
        self.log(tr(f"{self.current_hrtf()[0]}には上下の測定がないため、"
                    "上下の聴き取りはできません。"
                    f"HRTF で実測 Viking の耳介を選んでから、もう一度 {key_name} を"
                    "押してください。",
                    f"{self.current_hrtf()[0]} has no up/down measurements, "
                    "so the up/down tests cannot run. Choose a measured "
                    f"Viking pinna in HRTF, then press {key_name} again."))
        return True

    def _elevation_answer_text(self) -> str:
        parts = [tr("音が鳴ったら、テンキーのスラッシュで真上、8で上、5で水平、"
                    "2で下と答えてください。",
                    "When the sound plays, answer with numeric keypad slash "
                    "for straight up, 8 for up, 5 for level or 2 for down.")]
        if self.reference_enabled:
            parts.append(tr("基準音がオンです。先に水平の音、続けて問題の音が鳴ります。"
                            "2つ目の音が1つ目より上か下かを聞いてください。",
                            "The reference sound is on. A level sound plays "
                            "first, then the question sound. Listen for "
                            "whether the second is higher or lower than the "
                            "first."))
        if self._pitch != 0.0:
            # 顔を傾けたまま始めると、聞こえ方が普段と違う。気づけるように
            parts.append(tr(f"いま顔は{self.pitch_text()}を向いています。"
                            "答えは顔ではなく体から見た高さです。",
                            f"Your face is turned {self.pitch_text()} now. "
                            "Answer with the height as seen from your body, "
                            "not your face."))
        return join_sentences(*parts)

    def start_elevation(self) -> None:
        """上下（仰角）の聴き取りテスト。方位は難易度の方向から混ぜる。"""
        if not self.ensure_player():
            return
        if self._refuse_without_elevation("Shift+F3"):
            return
        diff = self.difficulties[max(0, self.cmb_diff.GetSelection())]
        # 高さ4つ×方位を2巡。8方向以上では48問で打ち切る（長すぎると
        # 集中が切れて、後半の成績が聴覚ではなく疲労を測ってしまう）。
        n = min(48, max(24, len(ELEVATION_LEVELS) * len(diff.angles) * 2))
        self._begin("elev", ElevationSession(difficulty=diff, n_trials=n),
                    join_sentences(
                        tr(f"上下の聴き取りを開始します。全{n}問。"
                           f"{self.current_hrtf()[0]}を使います。"
                           f"方位は{diff.label}から選びますが、答えるのは高さだけです。",
                           f"Starting the up/down test. {n} questions. "
                           f"Using {self.current_hrtf()[0]}. Directions "
                           f"come from {diff.label}, but you answer only "
                           "the height."),
                        self._elevation_answer_text()))

    def start_elevation_front(self) -> None:
        """正面固定の上下の聴き取り。方位は正面だけで、変わるのは高さだけ。"""
        if not self.ensure_player():
            return
        if self._refuse_without_elevation("Ctrl+F3"):
            return
        # 高さ4つを6回ずつ。方位が1つなので難易度は使わない
        n = len(ELEVATION_LEVELS) * 6
        self._begin("elev_front", FrontElevationSession(n_trials=n),
                    join_sentences(
                        tr(f"正面固定の上下の聴き取りを開始します。全{n}問。"
                           f"{self.current_hrtf()[0]}を使います。"
                           "音は毎回正面から鳴り、変わるのは高さだけです。",
                           f"Starting the up/down test from the front. {n} "
                           f"questions. Using {self.current_hrtf()[0]}. The "
                           "sound always comes from the front, and only "
                           "the height changes."),
                        self._elevation_answer_text()))

    def start_rotation(self) -> None:
        """回る向きの聴き取り。音が半周する向きを当てる。"""
        if not self.ensure_player():
            return
        self._restore_selected_hrir()
        overhead = self.hrir_supports_elevation()
        session = RotationSession(n_trials=24, include_overhead=overhead)
        if overhead:
            kinds = tr("水平に回る音と、頭上を越える音が出ます。"
                       "水平は、テンキーの4で左回り、6で右回り。"
                       "頭上は、8で後ろから前へ、2で前から後ろへ答えてください。",
                       "There are level circles and sounds passing "
                       "overhead. For level ones, answer numpad 4 for "
                       "counterclockwise or 6 for clockwise. For overhead "
                       "ones, answer 8 for back to front or 2 for front to "
                       "back.")
        else:
            kinds = tr(f"{self.current_hrtf()[0]}には上下の測定がないため、"
                       "水平に回る音だけを出します。"
                       "テンキーの4で左回り、6で右回りと答えてください。",
                       f"{self.current_hrtf()[0]} has no up/down "
                       "measurements, so only level circles are asked. "
                       "Answer numpad 4 for counterclockwise or 6 for "
                       "clockwise.")
        self._begin("rotation", session, join_sentences(
            tr(f"回る向きの聴き取りを開始します。全{session.n_trials}問。"
               "音が自分の周りを半周します。回り始めの方向は毎回変わります。"
               "左回りは、上から見て反時計回り、前から左、後ろ、右の順です。",
               f"Starting the rotation test. {session.n_trials} questions. "
               "A sound travels half a circle around you, starting from a "
               "different direction each time. Counterclockwise is as seen "
               "from above: front, then left, back, right."),
            kinds))

    def play_rotation(self, trial: RotationTrial) -> None:
        """音を回り始めの位置で鳴らし、ROTATION_ARC_SECONDS かけて半周させる。

        位置は経過時間から決める（タイマーの間隔の揺れで速さが変わらない
        ように）。回り始めたときの向きを基準にし、その人の前から回す。
        途中で向きを変えても音は付いてこない（向きを変えて前後を確かめる
        やり方が使える）。
        """
        if not self.ensure_player():
            return
        self.stop_sound()
        self._rot_trial = trial
        self._rot_t0 = time.perf_counter()
        self._rot_yaw = self._yaw
        self._handle = self.player.play(
            self.bank.get(self._sound_kind()),
            position=trial.position(0.0, self._rot_yaw), loop=True,
            reverb_send=0.3)
        self._rot_timer.Start(20)

    def on_rotation_tick(self, event: wx.TimerEvent | None = None) -> None:
        trial = self._rot_trial
        if self.player is None or self._handle is None or trial is None:
            return
        fraction = (time.perf_counter() - self._rot_t0) / ROTATION_ARC_SECONDS
        try:
            self.player.engine.set_source_position(
                self._handle, trial.position(fraction, self._rot_yaw))
        except KeyError:
            self.stop_sound()
            return
        if fraction >= 1.0:
            self._fade_out_rotation()        # 回り終えた

    def _fade_out_rotation(self) -> None:
        """回している音を短く消して止める。

        ループする音をいきなり切ると「プツッ」と鳴る。回り終えたときも、
        途中で答えたときも、40ms で消す。
        """
        self._rot_timer.Stop()
        self._rot_trial = None
        if self.player is not None and self._handle is not None:
            try:
                self.player.stop_sound(self._handle, fade_ms=40.0)
            except Exception:  # noqa: BLE001
                pass
        self._handle = None

    # -- 出題と回答 ----------------------------------------------------
    def play_current(self) -> None:
        if self.session is None or self.session.finished:
            return
        self._pending_next = False     # 次の問題を鳴らす（F4 も受け付ける）
        trial = self.session.current()
        if isinstance(self.session, (CalibrationSession,
                                     ElevationCalibrationSession)):
            # ラウンドの先頭でだけ耳介を切り替える。_active_tag を見るのは、
            # 同じ問題の再出題（F4）で切替をやり直さないためと、切替に
            # 失敗したとき F4 でもう一度試せるようにするため。
            if (self.session.is_round_start()
                    and self.session.current_tag != self._active_tag):
                self._switch_pinna()
                return
        if isinstance(self.session, ABWobbleSession):
            key = (self._ab_wobble_key
                   if self.session.wobble_enabled_for(trial) else "off")
            self.player.engine.set_head_wobble(WOBBLE_PRESETS[key])
        self.SetStatusText(self._progress_text())
        self._play_trial(trial)
        self._accepting = True
        self._arm_repeat()

    def _play_trial(self, trial) -> None:
        """その問題の音を鳴らす。上下の問題なら仰角と基準音も付ける。"""
        if isinstance(trial, RotationTrial):
            self.play_rotation(trial)
            return
        self.play_at(trial.truth_deg, trial.distance_m,
                     elevation_deg=getattr(trial, "truth_el", 0.0),
                     with_reference=self._reference_applies())

    def _switch_pinna(self) -> None:
        """耳介を切り替えてから出題する。読み込みは数百ミリ秒かかる。"""
        session = self.session
        label = dict(session.candidates)[session.current_tag]
        self.log(tr(f"ここから{label}です。", f"Now {label}."), speak=True)
        try:
            self.player.engine.set_hrir(self.load_hrir(Path(session.current_tag)))
        except Exception as exc:  # noqa: BLE001
            self.speech.error(tr(f"{label} を読み込めません。{exc} "
                                 "F4 でもう一度試すか、Escape で終了できます。",
                                 f"Cannot load {label}. {exc} Press F4 to "
                                 "try again, or Escape to quit."))
            return
        self._active_tag = session.current_tag
        self._later(1500, self._play_after_switch)

    def _play_after_switch(self) -> None:
        if self.session is None or self.session.finished:
            return
        self._pending_next = False
        trial = self.session.current()
        self.SetStatusText(self._progress_text())
        self._play_trial(trial)
        self._accepting = True
        self._arm_repeat()

    def _progress_text(self) -> str:
        s = self.session
        if isinstance(s, TrainingSession):
            return tr(f"訓練 連続{s.streak}問 / 目標{s.target_streak}問"
                      f"（最長{s.best_streak}）",
                      f"Training: {s.streak} in a row / target "
                      f"{s.target_streak} (best {s.best_streak})")
        trial = s.current()
        return tr(f"{trial.index + 1}問目 / {s.n_trials}",
                  f"Question {trial.index + 1} of {s.n_trials}")

    def replay(self) -> None:
        if self.session is None or self.session.finished:
            self.speech.say(tr("再生できる問題がありません。"
                               "F3 でクイズ、F7 で訓練、Shift+F3 で上下の聴き取りを"
                               "始めます。",
                               "There is no question to play. F3 starts the "
                               "quiz, F7 training, Shift+F3 the up/down "
                               "test."))
            return
        if self._pending_next:
            # 答えた直後の読み上げの間。次の問題はもうすぐ鳴る（ここで鳴らすと
            # 読み上げにかぶり、その問題の聞き直しが1回数えられる）
            return
        self.session.note_replay()
        # play_at を直接呼ばず play_current を通す。耳介の切替に失敗した
        # 直後なら切替からやり直し、通常時はいまの問題を鳴らし直す。
        self.play_current()

    def submit(self, azimuth_deg: float) -> None:
        session = self.session
        if session is None:
            return
        # 次の出題音が鳴るまで、続けてのキー入力は受け付けない
        self._accepting = False
        self._pending_next = True
        self._stop_repeat()
        trial = session.answer(azimuth_deg, listener_yaw_deg=self._yaw)

        if isinstance(session, TrainingSession):
            self.speech.say(session.feedback(trial))
            self.lst_result.Set(session.detail_lines()[-12:])
            if not trial.correct:
                # 正解方向をもう一度鳴らして違いを聞かせる
                self._later(600, lambda: self.play_at(trial.truth_deg,
                                                      trial.distance_m))
                self._later(2600, self._after_answer)
            else:
                self._later(900, self._after_answer)
            return

        if trial.front_back:
            extra = tr("。前後の取り違えです", ". Front and back were confused")
        else:
            extra = ""
        self.speech.say(answer_text(trial.index + 1, trial.correct, extra,
                                    direction_name(trial.truth_deg)))
        self.lst_result.Set(session.detail_lines()[-12:])
        self._later(1200, self._after_answer)

    def submit_elevation(self, elevation_deg: float) -> None:
        session = self.session
        if not isinstance(session, ElevationSession):
            return
        self._accepting = False
        self._pending_next = True
        self._stop_repeat()
        trial = session.answer_elevation(elevation_deg)
        if trial.up_down:
            extra = tr("。上と下の取り違えです", ". Up and down were confused")
        else:
            extra = ""
        self.speech.say(answer_text(trial.index + 1, trial.correct, extra,
                                    elevation_name(trial.truth_el)))
        self.lst_result.Set(session.detail_lines()[-12:])
        self._later(1200, self._after_answer)

    def submit_rotation(self, motion: str) -> None:
        session = self.session
        if not isinstance(session, RotationSession):
            return
        self._accepting = False
        self._pending_next = True
        self._stop_repeat()
        # 回っている途中で答えたら、音は切らずに短く消す
        self._fade_out_rotation()
        self.stop_sound()
        trial = session.answer_rotation(motion)
        if trial.reversed:
            extra = tr("。回る向きの取り違えです",
                       ". The direction was reversed")
        else:
            extra = ""
        self.speech.say(answer_text(trial.index + 1, trial.correct, extra,
                                    rotation_name(trial.truth_motion)))
        self.lst_result.Set(session.detail_lines()[-12:])
        self._later(1200, self._after_answer)

    def _after_answer(self) -> None:
        self._pending_next = False
        session = self.session
        if session is None:
            return
        if session.finished:
            self.stop_sound()
            self.finish_session()
        else:
            self.play_current()

    def finish_session(self) -> None:
        self._stop_repeat()
        session = self.session
        summary = session.summary()
        self.lst_result.Set(session.detail_lines())
        # 耳介の選定が終わったら、最後に試した耳介から設定欄の HRTF へ戻す
        self._restore_selected_hrir()

        if isinstance(session, ABWobbleSession):
            # 測定中は首振りを問題ごとに切り替えていた。設定値に戻す
            self.apply_settings()

        # 耳介の選定は通算成績にも積み上げる。1回48問では足りないので、
        # 何回かに分けて母数を稼ぐ。前後用と上下用で別々に積む。
        store = None
        if isinstance(session, CalibrationSession):
            store = self.records
            n_directions = len(session.difficulty.angles)
        elif isinstance(session, ElevationCalibrationSession):
            store = self.records_elev
            # 記録は「方向数が8以上の回だけを順位に使う」作り。上下の4択は
            # 前後左右の4方向と違い全員が満点に張り付くほど易しくないので、
            # 判別力のある回として数える。
            n_directions = MIN_DIRECTIONS_FOR_RANKING
        if store is not None:
            pool = self._pinna_pool()
            store.update(session.scores(), self._anchor_tag,
                         n_directions=n_directions)
            try:
                store.save()
            except OSError as exc:
                self.speech.warn(tr(f"戦績を保存できませんでした。{exc}",
                                    f"Could not save the totals. {exc}"))
            summary = summary + " " + store.summary(pool)
            self.lst_result.Set(session.detail_lines()
                                + [tr("--- 通算 ---", "--- Totals ---")]
                                + store.detail_lines())

        path, saved = self.save_log(session, summary)
        if saved and isinstance(session, CalibrationSession):
            self._mark_recorded(path)
        # 保存に失敗したときに「保存しました」と読み上げない。矛盾した
        # 2つの報告が続くと、どちらが本当か分からなくなる。
        if saved:
            tail = tr(f" 結果を {path.name} に保存しました。",
                      f" Results saved to {path.name}.")
        else:
            tail = tr(" 結果のファイル保存に失敗しました。結果は F5 で"
                      "読み上げられます。",
                      " Saving the results file failed. F5 reads the "
                      "results.")
        self.log(summary + tail)
        self.SetStatusText(tr("終了。F5 で結果の読み上げ",
                              "Finished. F5 reads the results"))

    def _mark_recorded(self, path: Path) -> None:
        """このログは通算に反映済みだと印を付ける。

        古いログを通算に取り込むときは、この印で取り込み済みかを判断
        する。印がないと、アプリが記録した分をもう一度取り込んで
        二重集計になる。
        """
        marker = LOG_DIR / "imported_calib.txt"
        try:
            names = set()
            if marker.exists():
                names = {x.strip() for x in
                         marker.read_text(encoding="utf-8").splitlines()
                         if x.strip()}
            names.add(path.name)
            marker.write_text("\n".join(sorted(names)), encoding="utf-8")
        except OSError:
            # 印付けに失敗しても結果自体は保存済みなので続行する
            pass

    def announce_result(self) -> None:
        if self.session is None:
            self.speech.say(tr("まだ実施していません。"
                               "F3 でクイズ、F7 で訓練、F8 で耳介の選定を始めます。",
                               "Nothing has been done yet. F3 starts the "
                               "quiz, F7 training, F8 choosing a pinna."))
            return
        self.speech.say(self.session.summary())

    def save_log(self, session: QuizSession, summary: str) -> tuple[Path, bool]:
        """結果をファイルに残す。(パス, 保存できたか) を返す。

        どの設定で試したかを後から確認できないと、条件を変えた比較が
        できない。自動保存にしておく。成否の報告は呼び出し側がまとめて
        読み上げる（ここで警告を出すと成功報告と二重になる）。
        """
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"{self.mode}_{stamp}.txt"
        lines = [tr(f"モード: {self.mode}", f"Mode: {self.mode}"),
                 tr(f"日時: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                    f"Date: {dt.datetime.now():%Y-%m-%d %H:%M:%S}"),
                 tr(f"設定: {self.settings_text()}",
                    f"Settings: {self.settings_text()}"),
                 "",
                 tr(f"結果: {summary}", f"Results: {summary}"),
                 "",
                 tr("詳細:", "Details:")]
        lines.extend(f"  {line}" for line in session.detail_lines())
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            return path, False
        return path, True

    def on_close(self, event: wx.CloseEvent) -> None:
        self._stop_repeat()
        self.stop_sound()
        if self.player is not None:
            self.player.stop()
            self.player = None
        self.speech.stop()
        event.Skip()


class DemoApp(wx.App):
    def __init__(self, start_elevation: bool = False) -> None:
        self.start_elevation = start_elevation
        super().__init__(False)

    def OnInit(self) -> bool:
        frame = MainFrame()
        frame.Show()
        self.SetTopWindow(frame)
        if self.start_elevation:
            # 画面が出てから始める。OnInit の中で始めると、ウィンドウが
            # まだ前面に来ておらず、スクリーンリーダーの読み上げより先に
            # 案内が流れて聞き逃す。
            wx.CallLater(500, frame.start_elevation)
        return True


def main() -> int:
    app = DemoApp(start_elevation="--elevation" in sys.argv[1:])
    app.MainLoop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
