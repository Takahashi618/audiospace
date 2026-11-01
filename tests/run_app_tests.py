"""GUI を表示せずに localization_app の組み立てを検証する。

wx.App と Frame は生成するが Show() せず、ウィジェットのラベル・名前・
キー対応・音の生成だけを確認してから破棄する。読み上げと音声出力は
AUDIOSPACE_SILENT=1 で止めるので、実行しても音は出ない。

Out: tests/app_test_result.txt (UTF-8)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
# 表示の言語を日本語に固定する（テストが期待する文は日本語）
os.environ["AUDIOSPACE_LANG"] = "ja"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wx  # noqa: E402

from demo.localization_app import (SOUND_KINDS, MainFrame,  # noqa: E402
                                   SoundBank, list_hrtf_choices)

OUT = Path(__file__).with_name("app_test_result.txt")
_log: list[str] = []
_passed = 0
_failed = 0


def log(msg: str = "") -> None:
    _log.append(msg)


def test(name: str):
    def deco(fn):
        def run(frame: MainFrame) -> None:
            global _passed, _failed
            try:
                detail = fn(frame)
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


@test("すべてのウィジェットに SetName がある")
def t_names(frame: MainFrame) -> str:
    checked = []
    missing = []

    def walk(win: wx.Window) -> None:
        for child in win.GetChildren():
            cls = type(child).__name__
            if cls in ("Choice", "TextCtrl", "ListBox", "SpinCtrl",
                       "StaticText", "Button", "CheckBox"):
                name = child.GetName()
                checked.append(f"{cls}:{name}")
                if not name or name in ("choice", "text", "listBox",
                                        "wxSpinCtrl", "staticText", "button",
                                        "check", "panel", "control"):
                    missing.append(f"{cls} にわかりやすい名前がありません")
            walk(child)

    walk(frame)
    assert not missing, "\n".join(missing)
    return f"{len(checked)}個のウィジェットを確認\n" + "\n".join(
        f"  {c}" for c in checked)


@test("ラベルは対応するコントロールより先に生成されている")
def t_label_order(frame: MainFrame) -> str:
    """スクリーンリーダーは生成順で関連付けを推測することがあるため、
    StaticText が対応コントロールの直前にある並びを保つ。"""
    order = []

    def walk(win: wx.Window) -> None:
        for child in win.GetChildren():
            order.append(child)
            walk(child)

    walk(frame)
    pairs = [(frame.cmb_hrtf, "HRTF"), (frame.cmb_sound, "音の種類"),
             (frame.cmb_wobble, "首振り"), (frame.cmb_roll, "首かしげ"),
             (frame.cmb_room, "残響"), (frame.cmb_space, "空間"),
             (frame.chk_precise, "精密な残響"),
             (frame.cmb_diff, "難易度"),
             (frame.spn_depth, "前後の強調"), (frame.spn_tilt, "上下の強調"),
             (frame.chk_floor, "床の反射"), (frame.txt_log, "状況"),
             (frame.lst_result, "結果の一覧")]
    lines = []
    for ctrl, label_name in pairs:
        labels = [w for w in order
                  if isinstance(w, wx.StaticText) and w.GetName() == label_name]
        assert labels, f"「{label_name}」のラベルが見つかりません"
        assert order.index(labels[0]) < order.index(ctrl), \
            f"「{label_name}」のラベルがコントロールより後に生成されています"
        lines.append(f"{label_name} → {type(ctrl).__name__}")
    return "、".join(lines)


@test("動作キーに文字キーを使っていない")
def t_no_letter_keys(frame: MainFrame) -> str:
    """IME と衝突するため、動作は F キー・テンキー・矢印・Escape のみ。"""
    for code in range(ord("A"), ord("Z") + 1):
        got = frame._azimuth_for_key(code)
        assert got is None, f"文字キー {chr(code)} に方向が割り当てられています"
    for code in range(ord("0"), ord("9") + 1):
        assert frame._azimuth_for_key(code) is None, \
            f"通常の数字キー {chr(code)} に方向が割り当てられています"
    numpad = [wx.WXK_NUMPAD1, wx.WXK_NUMPAD2, wx.WXK_NUMPAD3,
              wx.WXK_NUMPAD4, wx.WXK_NUMPAD6, wx.WXK_NUMPAD7,
              wx.WXK_NUMPAD8, wx.WXK_NUMPAD9]
    mapped = {n: frame._azimuth_for_key(n) for n in numpad}
    assert all(v is not None for v in mapped.values()), mapped
    return (f"文字キー26個・数字キー10個すべて未割り当て、"
            f"テンキー8方向のみ有効 (8=前 4=左 6=右 2=後ろ)")


@test("矢印キーは回答ではなく向きの変更に使う")
def t_arrows_turn(frame: MainFrame) -> str:
    """矢印を回答に兼用すると、向きを変えたいのに回答してしまう。"""
    for code in (wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_UP, wx.WXK_DOWN):
        assert frame._azimuth_for_key(code) is None, \
            "矢印キーに方向が割り当てられています"
    frame._yaw = 0.0
    frame.turn(wx.WXK_LEFT)
    left_once = frame._yaw
    frame.turn(wx.WXK_LEFT)
    left_twice = frame._yaw
    frame.turn(wx.WXK_RIGHT)
    back = frame._yaw
    assert left_once == 15.0 and left_twice == 30.0, (left_once, left_twice)
    assert back == 15.0, back
    # 左に回ると yaw が増える = 反時計回り。方位規約と一致する
    frame._yaw = 0.0
    return (f"左で+15度ずつ ({left_once:g}→{left_twice:g})、"
            f"右で戻り ({back:g})")


@test("上下の矢印は横取りせず、選択肢や一覧の操作に渡す")
def t_updown_passthrough(frame: MainFrame) -> str:
    """以前は上矢印を「正面に向き直す」に使っていた。パネルの CHAR_HOOK は
    フォーカス中のコントロールより先に届くので、選択肢を上下で選ぶたびに
    選択内容より先に「向き0度」と読み上げていた。"""
    frame._yaw = 30.0
    lines = []
    for code, name in ((wx.WXK_UP, "上"), (wx.WXK_DOWN, "下")):
        said_before = len(frame.speech.log)
        event = _key(code)
        frame.on_key(event)
        assert event.GetSkipped(), \
            f"{name}矢印がコントロールに渡されていません"
        assert frame._yaw == 30.0, f"{name}矢印で向きが {frame._yaw} 度に変わりました"
        said = frame.speech.log[said_before:]
        assert not any("向き" in s for s in said), \
            f"{name}矢印で向きを読み上げました: {said}"
        lines.append(f"{name}矢印: コントロールへ渡す、向きは変えない、読み上げなし")
    event = _key(wx.WXK_LEFT)
    frame.on_key(event)
    assert not event.GetSkipped() and frame._yaw == 45.0, \
        "左矢印で向きが変わりません"
    frame._yaw = 0.0
    return "\n".join(lines + ["左矢印は今までどおり向きを変える"])


@test("首振りの選択肢がプリセットと対応している")
def t_wobble_choices(frame: MainFrame) -> str:
    from audiospace import WOBBLE_PRESETS

    assert frame.cmb_wobble.GetCount() == len(frame.wobble_keys)
    for key in frame.wobble_keys:
        assert key in WOBBLE_PRESETS, f"{key} がプリセットにありません"
    selected = frame.wobble_keys[frame.cmb_wobble.GetSelection()]
    assert WOBBLE_PRESETS[selected].enabled, \
        "既定で首振りが無効になっています。前後対策として既定で有効にすべきです"
    return (f"{frame.cmb_wobble.GetCount()}段階、既定は"
            f"{frame.cmb_wobble.GetStringSelection()} "
            f"({WOBBLE_PRESETS[selected].describe()})")


@test("設定の読み上げ文にすべての設定が含まれる")
def t_settings_text(frame: MainFrame) -> str:
    """どの条件で試したか後から分からなくなるのを防ぐための機能。"""
    text = frame.settings_text()
    for token in ("HRTF", "音は", "首振り", "残響", "難易度",
                  "前後の強調", "向き", "首かしげ", "上下の強調", "床の反射",
                  "空間は"):
        assert token in text, f"設定文に「{token}」がありません: {text}"
    return text


@test("テンキーの配置と方向が幾何的に一致する")
def t_numpad_geometry(frame: MainFrame) -> str:
    """7 8 9 / 4 5 6 / 1 2 3 の並びが、そのまま方向の配置になっているか。
    ずれていると触覚的な直感と食い違って使いにくい。"""
    layout = {
        wx.WXK_NUMPAD7: (-1, 1), wx.WXK_NUMPAD8: (0, 1),
        wx.WXK_NUMPAD9: (1, 1),
        wx.WXK_NUMPAD4: (-1, 0), wx.WXK_NUMPAD6: (1, 0),
        wx.WXK_NUMPAD1: (-1, -1), wx.WXK_NUMPAD2: (0, -1),
        wx.WXK_NUMPAD3: (1, -1),
    }
    lines = []
    for key, (dx, dy) in layout.items():
        az = frame._azimuth_for_key(key)
        # 方位 -> 画面上の向き (左が -x、前が +y)
        rad = np.radians(az)
        want_x = -np.sin(rad)
        want_y = np.cos(rad)
        norm = np.hypot(dx, dy)
        assert abs(want_x - dx / norm) < 1e-6 and \
               abs(want_y - dy / norm) < 1e-6, \
               f"キー配置({dx},{dy})と方位{az}度が一致しません"
        lines.append(f"({dx:+d},{dy:+d})={az:g}度")
    return "、".join(lines)


@test("音の種類がすべて生成できる")
def t_sound_bank(frame: MainFrame) -> str:
    bank = SoundBank(48000)
    lines = []
    for label, kind in SOUND_KINDS:
        data = bank.get(kind)
        assert data.ndim == 1 and data.dtype == np.float32, data.shape
        peak = float(np.max(np.abs(data)))
        assert 0.05 < peak <= 1.0, f"{kind} のピークが {peak}"
        assert len(data) > 1000, f"{kind} が短すぎます"
        lines.append(f"{kind}: {len(data)}サンプル "
                     f"({len(data)/48000:.2f}秒) ピーク{peak:.2f}")
    # キャッシュが効いていること
    assert bank.get("click") is bank.get("click"), "キャッシュが効いていません"
    return "\n".join(lines)


@test("HRTF の選択肢に合成と実測の両方が並ぶ")
def t_hrtf_choices(frame: MainFrame) -> str:
    items = list_hrtf_choices()
    assert items[0][1] is None, "先頭は合成HRIRのはず"
    measured = [n for n, p in items if p is not None]
    labels = [frame.cmb_hrtf.GetString(i)
              for i in range(frame.cmb_hrtf.GetCount())]
    assert len(labels) == len(items)
    return (f"合計{len(items)}件 (実測{len(measured)}件)。"
            f"既定の選択={frame.cmb_hrtf.GetStringSelection()}")


@test("初期状態で操作説明が表示されている")
def t_welcome(frame: MainFrame) -> str:
    text = frame.txt_log.GetValue()
    for key in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                "F10", "F11", "Shift+F3", "Ctrl+F3", "Shift+F2",
                "テンキーの0", "テンキーの*",
                "Page Up", "Page Down", "Escape",
                "テンキー", "矢印"):
        assert key in text, f"説明に {key} がありません"
    assert "ヘッドホン" in text, "ヘッドホン前提の注意がありません"
    assert frame.txt_log.IsEditable() is False, \
        "状況表示は読み取り専用であるべきです"
    return f"{len(text.splitlines())}行の説明を表示、読み取り専用を確認"


@test("初期選択は聴取テストで採用した耳介E")
def t_default_pinna(frame: MainFrame) -> str:
    """ライブラリの既定（耳介E）とアプリの既定がずれていると、
    アプリで確かめた印象と実際のゲームの音が食い違う。"""
    selected = frame.cmb_hrtf.GetStringSelection()
    _, path = frame.hrtf_choices[frame.cmb_hrtf.GetSelection()]
    if path is None or "subj_E" not in path.name:
        # E が未取得の環境では実測の先頭で妥協するが、
        # E がある限りは E でなければならない
        has_e = any(p is not None and p.stem == "subj_E"
                    for _, p in frame.hrtf_choices)
        assert not has_e, f"subj_E があるのに初期選択が {selected}"
        return f"subj_E 未取得のため {selected} で開始（許容）"
    return f"初期選択 = {selected}"


@test("出題音が鳴るまでテンキー回答を受け付けない")
def t_answer_gate(frame: MainFrame) -> str:
    """回答直後の読み上げ待ちにキーを二度押しすると、まだ鳴っていない
    次の問題に回答してしまう。画面が見えないと気づけない誤操作なので、
    出題音の開始でだけ受付を開く。"""
    from demo.quiz import Difficulty, QuizSession

    frame.session = QuizSession(difficulty=Difficulty.FOUR, n_trials=4,
                                seed=1)
    frame._accepting = False
    taken = frame.try_answer(wx.WXK_NUMPAD8)
    assert not taken, "出題前の回答が受理されました"
    assert frame.session.position == 0

    frame._accepting = True
    taken = frame.try_answer(wx.WXK_NUMPAD8)
    assert taken, "出題後の回答が拒否されました"
    assert frame.session.position == 1
    assert frame._accepting is False, \
        "回答後は次の出題まで受付を閉じるべきです"
    again = frame.try_answer(wx.WXK_NUMPAD8)
    assert not again, "読み上げ待ちの間の二度押しが次の問題に入りました"
    assert frame.session.position == 1

    frame.session = None            # 後続テストへ影響させない
    frame._accepting = False
    return "出題前は拒否、出題後は受理、回答直後の二度押しは拒否"


@test("F11 に首振りの効果測定が割り当てられている")
def t_ab_mode(frame: MainFrame) -> str:
    from demo.quiz import ABWobbleSession

    assert hasattr(frame, "start_ab"), "start_ab がありません"
    s = ABWobbleSession(n_trials=4, seed=1)
    assert s.wobble_enabled_for(s.trials[0]) is True
    assert s.wobble_enabled_for(s.trials[1]) is False
    assert "F11" in frame.txt_log.GetValue(), "説明に F11 がありません"
    return "start_ab あり、条件は交互、説明に F11 を記載"


@test("上下の聴き取り: テンキー / 8 5 2 が真上・上・水平・下に対応する")
def t_elev_keys(frame: MainFrame) -> str:
    from demo.quiz import ELEVATION_LEVELS

    want = {wx.WXK_NUMPAD_DIVIDE: 90.0, wx.WXK_NUMPAD8: 45.0,
            wx.WXK_NUMPAD5: 0.0, wx.WXK_NUMPAD2: -45.0}
    for code, el in want.items():
        got = frame._elevation_for_key(code)
        assert got == el, f"キー{code}が{got}度になりました（期待{el}度）"
    assert [el for el, _ in ELEVATION_LEVELS] == [90.0, 45.0, 0.0, -45.0]
    others = [wx.WXK_NUMPAD0, wx.WXK_NUMPAD1, wx.WXK_NUMPAD3, wx.WXK_NUMPAD4,
              wx.WXK_NUMPAD6, wx.WXK_NUMPAD7, wx.WXK_NUMPAD9,
              wx.WXK_NUMPAD_MULTIPLY, wx.WXK_UP, wx.WXK_DOWN,
              wx.WXK_PAGEUP, wx.WXK_PAGEDOWN]
    others += list(range(ord("A"), ord("Z") + 1))
    others += list(range(ord("0"), ord("9") + 1))
    for code in others:
        assert frame._elevation_for_key(code) is None, \
            f"キー{code}に高さが割り当てられています"
    return ("/=真上 8=上 5=水平 2=下、ほかのテンキー・矢印・Page Up/Down・"
            "文字・数字は未割り当て")


@test("上下の聴き取り: 回答は高さとして受理され、方位キーは無視される")
def t_elev_answer(frame: MainFrame) -> str:
    from demo.quiz import ElevationSession

    frame.session = ElevationSession(n_trials=4, seed=1)
    frame._accepting = True
    assert not frame.try_answer(wx.WXK_NUMPAD4), \
        "上下のテスト中に方位のキー（4=左）が回答として受理されました"
    assert frame.session.position == 0
    assert frame.try_answer(wx.WXK_NUMPAD8), "8（上）が受理されません"
    first = frame.session.trials[0]
    assert first.answer_el == 45.0, first.answer_el
    assert frame._accepting is False, "回答後は次の出題まで受付を閉じるべきです"
    assert not frame.try_answer(wx.WXK_NUMPAD2), "二度押しが受理されました"
    frame.session = None
    frame._accepting = False
    return "4（左）は無視、8で上と回答、直後の二度押しは拒否"


@test("上下の聴き取り: 上下の測定がない HRTF では始めずに理由を伝える")
def t_elev_guard(frame: MainFrame) -> str:
    """合成 HRIR や水平面だけの HRTF で出題すると、上も下も水平と同じ音に
    なる。成績が当て推量に張り付くだけで理由が分からないので、断る。"""
    from audiospace import HrirSet

    assert frame.ensure_player(), "無音モードでプレイヤーを開けません"
    engine = frame.player.engine
    original = engine.hrir
    rate = engine.sample_rate
    try:
        engine.set_hrir(HrirSet.synthetic(sample_rate=rate))
        assert not frame.hrir_supports_elevation()
        frame.start_elevation()
        assert frame.session is None and frame.mode == "idle", \
            "合成 HRIR で上下のテストが始まりました"
        said = frame.txt_log.GetValue()
        assert "上下の測定がない" in said and "Shift+F3" in said, said
        refused = said

        engine.set_hrir(original.horizontal())
        assert not frame.hrir_supports_elevation(), \
            "水平面だけの HRTF を上下に使えると判定しました"

        engine.set_hrir(original)
        full = original.elevation.max() >= 40 and original.elevation.min() <= -40
        assert frame.hrir_supports_elevation() == bool(full), \
            (f"仰角 {original.elevation.min():.0f}〜"
             f"{original.elevation.max():.0f}度の HRTF で判定が逆です")
    finally:
        engine.set_hrir(original)
        frame.session = None
        frame.mode = "idle"
    return (f"合成と水平面のみは拒否、既定の{original.name}"
            f"（{original.n_directions}方向）は"
            f"{'使用可' if full else '上下なしのため拒否'}\n断りの文: {refused}")


def _key(code: int, shift: bool = False, ctrl: bool = False) -> wx.KeyEvent:
    """パネルの EVT_CHAR_HOOK に届くのと同じ形のキーイベントを作る。"""
    event = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    event.SetKeyCode(code)
    event.SetShiftDown(shift)
    event.SetControlDown(ctrl)
    return event


def _reset(frame: MainFrame) -> None:
    frame._stop_repeat()
    frame.stop_sound()
    frame.session = None
    frame.mode = "idle"
    frame._accepting = False
    frame._begin_gen += 1          # 予約済みの遅延出題を無効にする


@test("上下の聴き取りは Shift+F3 で始まり、F12 では何も起きない")
def t_shift_f3(frame: MainFrame) -> str:
    """F12 は PC-Talker のショートカットと衝突したので外した。"""
    from demo.quiz import ElevationSession, QuizSession

    assert frame.ensure_player(), "無音モードでプレイヤーを開けません"
    try:
        frame.on_key(_key(wx.WXK_F12))
        assert frame.mode == "idle" and frame.session is None, \
            "F12 でモードが始まりました（PC-Talker と衝突します）"
        frame.on_key(_key(wx.WXK_F3, shift=True))
        assert isinstance(frame.session, ElevationSession), \
            f"Shift+F3 で上下の聴き取りが始まりません: {frame.mode}"
        _reset(frame)
        frame.on_key(_key(wx.WXK_F3))
        assert type(frame.session) is QuizSession, \
            f"Shift なしの F3 がクイズになりません: {frame.mode}"
    finally:
        _reset(frame)
    return "F12 は無反応、Shift+F3 で上下、F3 単独はクイズのまま"


@test("繰り返し再生: 回答待ちの間だけ間を空けて鳴らし直す")
def t_repeat(frame: MainFrame) -> str:
    """読み上げに紛れて出題音を聞き逃しても、待てばまた聞ける。"""
    from demo.localization_app import REPEAT_GAP_S, REPEAT_MIN_S
    from demo.quiz import ElevationSession

    assert frame.ensure_player(), "無音モードでプレイヤーを開けません"
    assert frame.repeat_enabled, "起動時は繰り返し再生がオンのはずです"
    frame.cmb_sound.SetSelection(0)                  # ノイズバースト3連
    seconds = len(frame.bank.get("noise_bursts")) / frame.bank.sample_rate
    want = int(max(REPEAT_MIN_S, seconds + REPEAT_GAP_S) * 1000)
    assert frame.repeat_interval_ms() == want, frame.repeat_interval_ms()
    try:
        frame.session = ElevationSession(n_trials=4, seed=1)
        frame.mode = "elev"
        frame.play_current()
        assert frame._repeat_timer.IsRunning(), "出題後に次の再生が予約されません"
        first = frame._handle

        frame.on_repeat(None)                        # タイマーが来たときの動き
        assert frame._handle is not None and frame._handle != first, \
            "繰り返しで出題音が鳴り直していません"
        assert frame._repeat_timer.IsRunning(), "鳴らした後に次が予約されません"
        assert frame.session.trials[0].replays == 0, \
            "自動の繰り返しを F4 の聞き直しとして数えています"

        assert frame.try_answer(wx.WXK_NUMPAD8)
        assert not frame._repeat_timer.IsRunning(), \
            "回答した後も繰り返しが続いています"
        frame.on_repeat(None)                        # 止め損ねた予約が来ても
        assert frame.session.position == 1           # 次の問題は鳴らさない

        frame.play_current()
        frame.stop_and_say()
        assert not frame._repeat_timer.IsRunning(), "F6 で繰り返しが止まりません"
        frame.replay()
        assert frame._repeat_timer.IsRunning(), "F4 で繰り返しが再開しません"
    finally:
        _reset(frame)
    return (f"間隔{want}ミリ秒（音{seconds:.1f}秒＋{REPEAT_GAP_S:g}秒）、"
            "回答・F6 で停止、F4 で再開、自動分は聞き直し回数に数えない")


@test("繰り返し再生: テンキーの0でオンとオフが切り替わる")
def t_repeat_toggle(frame: MainFrame) -> str:
    from demo.quiz import QuizSession

    assert frame.ensure_player()
    assert frame._azimuth_for_key(wx.WXK_NUMPAD0) is None
    assert frame._elevation_for_key(wx.WXK_NUMPAD0) is None
    try:
        frame.session = QuizSession(n_trials=4, seed=1)
        frame.mode = "quiz"
        frame.play_current()
        assert frame._repeat_timer.IsRunning()
        frame.on_key(_key(wx.WXK_NUMPAD0))
        assert not frame.repeat_enabled, "オフになりません"
        assert not frame._repeat_timer.IsRunning(), "オフにしても予約が残っています"
        assert frame.session.position == 0, "0 キーが回答として扱われました"
        assert "繰り返し再生はオフ" in frame.settings_text()
        frame.play_current()
        assert not frame._repeat_timer.IsRunning(), "オフなのに予約されました"
        frame.on_key(_key(wx.WXK_NUMPAD0))
        assert frame.repeat_enabled, "オンに戻りません"
        assert frame._repeat_timer.IsRunning(), \
            "回答待ちの途中でオンにしたのに予約されません"
        assert "繰り返し再生はオン" in frame.settings_text()
    finally:
        frame.repeat_enabled = True
        _reset(frame)
    return "0 でオフ（予約も解除）、もう一度でオン（すぐ予約）、回答には使わない"


@test("Ctrl+F3 で正面固定の上下の聴き取りが始まる")
def t_ctrl_f3(frame: MainFrame) -> str:
    from demo.quiz import ElevationSession, FrontElevationSession

    assert frame.ensure_player(), "無音モードでプレイヤーを開けません"
    try:
        frame.on_key(_key(wx.WXK_F3, ctrl=True))
        s = frame.session
        assert isinstance(s, FrontElevationSession), \
            f"Ctrl+F3 で正面固定の上下が始まりません: {frame.mode}"
        assert frame.mode == "elev_front", frame.mode
        assert {t.truth_deg for t in s.trials} == {0.0}
        text = frame.settings_text()
        assert "正面固定のため使わない" in text, \
            f"ログに難易度が残り、前後左右で試したと読めます: {text}"
        intro = frame.txt_log.GetValue()
        assert "スラッシュで真上" in intro, intro
        n = s.n_trials
        _reset(frame)
        frame.on_key(_key(wx.WXK_F3, shift=True))
        assert type(frame.session) is ElevationSession, \
            "Shift+F3 が方位を混ぜる上下のままになっていません"
    finally:
        _reset(frame)
    return f"Ctrl+F3 で正面固定（{n}問、方位は正面のみ）、Shift+F3 は従来どおり"


@test("Page Up で上、Page Down で下を15度ずつ向き、真上・真下で止まる")
def t_tilt(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    frame._pitch = 0.0
    frame._yaw = 30.0
    frame.apply_settings()
    try:
        ev = _key(wx.WXK_PAGEUP)
        frame.on_key(ev)
        assert not ev.GetSkipped(), "Page Up がコントロールへ流れました"
        frame.on_key(_key(wx.WXK_PAGEUP))
        assert frame._pitch == 30.0, frame._pitch
        assert engine.listener.pitch_deg == 30.0, engine.listener
        assert engine.listener.yaw_deg == 30.0, \
            "顔を上げたら左右の向きが変わりました"
        assert frame.speech.log[-1] == "顔の向き上30度", frame.speech.log[-1]
        assert "顔の向きは上30度" in frame.settings_text()

        for _ in range(10):
            frame.on_key(_key(wx.WXK_PAGEUP))
        assert frame._pitch == 90.0, f"真上で止まりません: {frame._pitch}"
        assert "真上" in frame.speech.log[-1] and "これ以上" in frame.speech.log[-1], \
            frame.speech.log[-1]
        for _ in range(20):
            frame.on_key(_key(wx.WXK_PAGEDOWN))
        assert frame._pitch == -90.0, f"真下で止まりません: {frame._pitch}"
        assert engine.listener.pitch_deg == -90.0
        top_said = "真上、これ以上の読み上げあり"

        # 左右の向きを変えても顔の上下は保たれる
        frame.on_key(_key(wx.WXK_LEFT))
        assert engine.listener.pitch_deg == -90.0, \
            "左右に回ったら顔の上下が水平に戻りました"
        # テンキーの 9 と 3 は回答のキー。顔を動かさない
        before = frame._pitch
        frame.on_key(_key(wx.WXK_NUMPAD9))
        frame.on_key(_key(wx.WXK_NUMPAD3))
        assert frame._pitch == before, "テンキーの 9 / 3 で顔が動きました"
    finally:
        frame._pitch = 0.0
        frame._yaw = 0.0
        frame.apply_settings()
    return f"15度刻み、上30度で読み上げ「顔の向き上30度」、{top_said}、真下で停止、左右の向きと独立"


def _slot(frame: MainFrame, handle: int):
    return frame.player.engine._slots[handle]


@test("基準音: 上下の問題の前に、同じ方位・水平の音を鳴らす")
def t_reference(frame: MainFrame) -> str:
    """「どの段か」の絶対判断を「基準より上か下か」の比較に変える。"""
    from audiospace.geometry import Listener, relative_spherical
    from demo.localization_app import (REFERENCE_GAP_S,
                                       REFERENCE_REPEAT_GAP_S)
    from demo.quiz import ElevationSession, QuizSession

    assert frame.ensure_player()
    assert frame.reference_enabled, "起動時は基準音がオンのはずです"
    frame.cmb_sound.SetSelection(0)
    one = len(frame.bank.get("noise_bursts")) / frame.bank.sample_rate
    try:
        s = ElevationSession(n_trials=8, seed=4)
        # 真上以外の問題を選ぶ（真上は方位がない）
        while s.current().truth_el >= 90.0:
            s.answer_elevation(0.0)
        frame.session = s
        frame.mode = "elev"
        frame.play_current()
        trial = s.current()
        assert frame._ref_handle is not None, "基準音が鳴っていません"
        assert frame._handle is not None
        ref = _slot(frame, frame._ref_handle)
        tgt = _slot(frame, frame._handle)
        listener = Listener()
        r_az, r_el, _ = relative_spherical(listener, ref.position)
        t_az, t_el, _ = relative_spherical(listener, tgt.position)
        assert abs(r_el) < 1e-6, f"基準音が水平ではありません: {r_el}"
        assert abs(((r_az - trial.truth_deg + 180) % 360) - 180) < 1e-6, \
            "基準音の方位が問題と違います"
        assert abs(t_el - trial.truth_el) < 1e-6, (t_el, trial.truth_el)
        lead = len(tgt.source.data) - len(ref.source.data) \
            if hasattr(tgt.source, "data") else None
        want_ms = int((one * 2 + REFERENCE_GAP_S
                       + REFERENCE_REPEAT_GAP_S) * 1000)
        assert frame.repeat_interval_ms() == want_ms, \
            (frame.repeat_interval_ms(), want_ms)
        assert "基準音はオン" in frame.settings_text()

        frame.stop_and_say()
        assert frame._ref_handle is None and frame._handle is None, \
            "F6 で基準音か問題の音が残りました"

        frame.on_key(_key(wx.WXK_NUMPAD_MULTIPLY))
        assert not frame.reference_enabled, "テンキーの * でオフになりません"
        assert s.position == s.current().index, "* が回答として扱われました"
        frame.play_current()
        assert frame._ref_handle is None, "オフなのに基準音が鳴りました"
        assert frame.repeat_interval_ms() == int((one + 1.0) * 1000)
        frame.on_key(_key(wx.WXK_NUMPAD_MULTIPLY))
        assert frame.reference_enabled, "もう一度押してもオンに戻りません"

        # 方位のクイズには付けない
        _reset(frame)
        frame.session = QuizSession(n_trials=4, seed=1)
        frame.mode = "quiz"
        frame.play_current()
        assert frame._ref_handle is None, "方位のクイズに基準音が付きました"
    finally:
        frame.reference_enabled = True
        _reset(frame)
    return (f"基準音は同じ方位・水平、問題は{trial.truth_el:g}度。"
            f"1回分{want_ms}ミリ秒ごとに繰り返し（音{one:.1f}秒×2＋間"
            f"{REFERENCE_GAP_S:g}秒＋{REFERENCE_REPEAT_GAP_S:g}秒）。"
            f"F6 で両方停止、* でオンオフ、方位のクイズには付けない"
            + (f"、遅れ{lead}サンプル" if lead is not None else ""))


@test("聞き比べ: Shift+F2 で始まり、/ 8 5 2 でその高さが正面で鳴る")
def t_listen(frame: MainFrame) -> str:
    from audiospace.geometry import Listener, relative_spherical
    from demo.quiz import QuizSession

    assert frame.ensure_player()
    try:
        frame.on_key(_key(wx.WXK_F2, shift=True))
        assert frame.mode == "listen" and frame.session is None, frame.mode
        assert frame._orbit_timer is None, "Shift+F2 で試聴（F2）が始まりました"
        lines = []
        for code, want in ((wx.WXK_NUMPAD_DIVIDE, 90.0),
                           (wx.WXK_NUMPAD8, 45.0), (wx.WXK_NUMPAD5, 0.0),
                           (wx.WXK_NUMPAD2, -45.0)):
            said = len(frame.speech.log)
            frame.on_key(_key(code))
            assert frame._handle is not None, f"キー{code}で鳴りません"
            az, el, _ = relative_spherical(
                Listener(), _slot(frame, frame._handle).position)
            assert abs(el - want) < 1e-6, f"{want}度のはずが{el}度"
            if want < 90.0:
                assert min(az, 360 - az) < 1e-6, f"正面ではありません: {az}"
            assert len(frame.speech.log) == said, \
                f"押したときに読み上げました: {frame.speech.log[said:]}"
            assert frame._ref_handle is None, "聞き比べに基準音が付きました"
            lines.append(frame.GetStatusBar().GetStatusText())
        frame.on_key(_key(wx.WXK_NUMPAD4))
        assert frame.mode == "listen", "方位のキーで聞き比べが終わりました"

        frame.on_key(_key(wx.WXK_F3))
        assert type(frame.session) is QuizSession, "別のモードに移れません"
        frame.on_key(_key(wx.WXK_NUMPAD8))
        assert frame.mode == "quiz"
    finally:
        _reset(frame)
    return ("、".join(lines) + "。押しても読み上げず、"
            "方位のキーは無視、F3 でクイズへ移れる")


@test("上下の手がかり3つ（首かしげ・上下の強調・床の反射）は起動時オフ")
def t_cues_default_off(frame: MainFrame) -> str:
    """利用者の指定。今までと同じ聞こえ方で始め、1つずつ効果を確かめる。"""
    assert frame._roll_key() == "off", frame._roll_key()
    assert frame.spn_tilt.GetValue() == 0
    assert frame.chk_floor.GetValue() is False
    assert frame.ensure_player()
    frame.apply_settings()
    assert not frame.player.engine.roll_wobble.enabled
    assert frame.player.engine.elevation_emphasis == 0.0
    text = frame.settings_text()
    for token in ("首かしげはなし", "上下の強調は0パーセント", "床の反射はなし"):
        assert token in text, f"「{token}」がありません: {text}"
    return "首かしげなし、上下の強調0、床の反射なし。設定文にも記録"


@test("首かしげ: 選択がエンジンの揺れに届き、上下の音を逆向きに動かす")
def t_roll_setting(frame: MainFrame) -> str:
    from audiospace import ROLL_PRESETS
    from audiospace.geometry import Listener, relative_spherical
    from demo.quiz import position_for

    assert frame.ensure_player()
    engine = frame.player.engine
    try:
        frame.cmb_roll.SetSelection(frame.roll_keys.index("clear"))
        frame.apply_settings()
        assert engine.roll_wobble == ROLL_PRESETS["clear"], engine.roll_wobble
        assert "首かしげは標準" in frame.settings_text()
        # 左耳を下げたとき: 上は右へ、下は左へ、正面水平は動かない
        tilt = Listener(roll_deg=12.0)
        up = relative_spherical(tilt, position_for(0.0, 4.0, 45.0))[0]
        down = relative_spherical(tilt, position_for(0.0, 4.0, -45.0))[0]
        level = relative_spherical(tilt, position_for(0.0, 4.0, 0.0))[0]
        assert 180 < up < 360, f"上の音が右へ動きません: {up}"
        assert 0 < down < 180, f"下の音が左へ動きません: {down}"
        assert min(level, 360 - level) < 1e-9, f"正面水平が動きました: {level}"
    finally:
        frame.cmb_roll.SetSelection(0)
        frame.apply_settings()
    assert not engine.roll_wobble.enabled
    return (f"標準で振り幅{ROLL_PRESETS['clear'].amplitude_deg:g}度。"
            f"左耳を12度下げると上45度は方位{up:.1f}度（右）、"
            f"下45度は{down:.1f}度（左）、正面水平は動かない")


@test("床の反射: 鏡像の位置と遅れ、方向は測定範囲内に寄せる")
def t_floor(frame: MainFrame) -> str:
    from audiospace.geometry import Listener, relative_spherical
    from demo.localization_app import (FLOOR_DEPTH_M,
                                       FLOOR_REFLECTION_GAIN, floor_image)
    from demo.quiz import FrontElevationSession, position_for

    delays = {}
    for el in (90.0, 45.0, 0.0, -45.0):
        image, delay = floor_image(position_for(0.0, 4.0, el))
        assert image is not None
        _, img_el, _ = relative_spherical(Listener(), image)
        assert img_el >= -45.0 - 1e-6, f"反射音が測定範囲の外です: {img_el}"
        delays[el] = delay
    assert delays[90.0] > delays[45.0] > delays[0.0] > delays[-45.0] > 0, \
        f"高い音ほど遅れが長くなっていません: {delays}"
    # 真上の反射は真下から来る。方位が定まらないので正面に寄せる（左右対称）
    image, _ = floor_image(position_for(0.0, 4.0, 90.0))
    az, _, _ = relative_spherical(Listener(), image)
    assert min(az, 360 - az) < 1e-6, f"真上の反射が正面ではありません: {az}"
    # 床より下の音には反射がない
    assert floor_image((0.0, 1.0, -FLOOR_DEPTH_M - 0.5))[0] is None

    assert frame.ensure_player()
    try:
        frame.reference_enabled = False
        frame.chk_floor.SetValue(True)
        frame.session = FrontElevationSession(n_trials=4, seed=2)
        frame.mode = "elev_front"
        frame.play_current()
        assert len(frame._extra_handles) == 1, frame._extra_handles
        refl = _slot(frame, frame._extra_handles[0])
        assert abs(refl.gain - FLOOR_REFLECTION_GAIN) < 1e-9, refl.gain
        frame.reference_enabled = True
        frame.play_current()
        assert len(frame._extra_handles) == 2, \
            "基準音にも反射が付いていません"
        frame.stop_and_say()
        assert frame._extra_handles == [], "F6 で反射音が残りました"
        frame.chk_floor.SetValue(False)
        frame.play_current()
        assert frame._extra_handles == [], "オフなのに反射音が鳴りました"
    finally:
        frame.chk_floor.SetValue(False)
        frame.reference_enabled = True
        _reset(frame)
    ms = "、".join(f"{el:g}度 {d * 1000:.1f}ms" for el, d in delays.items())
    return f"遅れ: {ms}。反射音は下45度以内、真上の反射は正面、F6 で停止"


@test("上下の強調はエンジンの機能で、F2 の試聴の回る音にもかかる")
def t_tilt_emphasis(frame: MainFrame) -> str:
    """以前はアプリが音を作るときに焼き込んでいて、回る音にはかけられな
    かった。いまはエンジンの elevation_emphasis で、音源ごとにその場で効く。"""
    assert frame.ensure_player()
    engine = frame.player.engine
    try:
        frame.spn_tilt.SetValue(35)
        event = wx.SpinEvent(wx.wxEVT_SPINCTRL, frame.spn_tilt.GetId())
        event.SetEventObject(frame.spn_tilt)
        frame.spn_tilt.GetEventHandler().ProcessEvent(event)
        assert abs(engine.elevation_emphasis - 0.35) < 1e-9,             engine.elevation_emphasis
        assert abs(engine.emphasis_db(90.0) - 4.2) < 1e-9
        assert abs(engine.emphasis_db(-45.0) + 2.1) < 1e-9
        # 鳴らす音は作り変えない（明るさはエンジンが変える）
        bank = frame.bank
        assert bank.variant(frame._sound_kind()) is bank.get(
            frame._sound_kind())
        # 試聴の回る音も同じエンジンを通るので、強調がかかる
        frame.toggle_orbit()
        assert frame.mode == "orbit"
        assert engine.effective_elevation_emphasis > 0.0
    finally:
        if frame._orbit_timer is not None:
            frame.toggle_orbit()
        frame.spn_tilt.SetValue(0)
        frame.apply_settings()
    assert engine.elevation_emphasis == 0.0
    return ("35パーセントでエンジンの強さ 0.35（真上 +4.2dB、下45度 -2.1dB）。"
            "音は作り変えず、試聴中もエンジンがかける")


@test("Ctrl+F8 で上下用の耳介の選定が始まり、記録は別ファイルに積む")
def t_calib_elev(frame: MainFrame) -> str:
    from demo.quiz import ElevationCalibrationSession

    assert frame.ensure_player()
    assert frame.records_elev.path.name == "pinna_records_elev.json"
    assert frame.records_elev.path != frame.records.path
    assert frame.records_elev.fb_name == "上下"
    try:
        frame.on_key(_key(wx.WXK_F8, ctrl=True))
        s = frame.session
        assert isinstance(s, ElevationCalibrationSession), frame.mode
        assert frame.mode == "calib_elev"
        assert s.trials_per_candidate == 12 and s.n_trials == 48, \
            (s.trials_per_candidate, s.n_trials)
        text = frame.settings_text()
        assert "耳介の選定中のため問題ごとに切替" in text, text
        assert "正面固定のため使わない" in text, text
        # F8 単独は前後用のまま
        _reset(frame)
        frame.on_key(_key(wx.WXK_F8))
        assert frame.mode == "calib", frame.mode
    finally:
        _reset(frame)
    return f"Ctrl+F8 で{s.n_trials}問（12問×4種類）、F8 は前後用のまま"


@test("耳介の選定の後や途中でほかのモードへ移ると、設定欄の HRTF に戻る")
def t_restore_hrir(frame: MainFrame) -> str:
    """戻さないと、次のモードは最後に試した耳介で鳴るのに、設定の
    読み上げとログには設定欄の耳介が書かれる（成績の取り違え）。"""
    import tempfile

    from demo.quiz import ElevationCalibrationSession
    from demo.records import RecordStore

    assert frame.ensure_player()
    engine = frame.player.engine
    label, selected = frame.current_hrtf()
    chosen = frame.load_hrir(selected)
    other = next(p for _, p in frame.hrtf_choices
                 if p is not None and p != selected)
    other_hrir = frame.load_hrir(other)

    def calibrating() -> ElevationCalibrationSession:
        s = ElevationCalibrationSession([(str(other), "他の耳介"),
                                         (str(selected), label)],
                                        trials_per_candidate=4, seed=1)
        frame.session = s
        frame.mode = "calib_elev"
        frame.play_current()                 # 1問目で耳介を切り替える
        assert engine.hrir is other_hrir, "選定で耳介が切り替わりません"
        return s

    lines = []
    saved_store, saved_log = frame.records_elev, frame.save_log
    try:
        calibrating()
        frame.start_elevation_front()
        assert engine.hrir is chosen, "Ctrl+F3 へ移っても戻りません"
        lines.append("途中で Ctrl+F3")

        calibrating()
        frame.start_listen()
        assert engine.hrir is chosen, "Shift+F2 へ移っても戻りません"
        lines.append("途中で Shift+F2")

        calibrating()
        frame.toggle_orbit()
        assert engine.hrir is chosen, "F2 の試聴へ移っても戻りません"
        frame.toggle_orbit()
        lines.append("途中で F2")

        # 最後まで終えたとき。記録は一時ファイルへ、ログは書かない
        with tempfile.TemporaryDirectory() as tmp:
            frame.records_elev = RecordStore(Path(tmp) / "elev.json",
                                             fb_name="上下")
            frame.save_log = lambda session, summary: (Path(tmp) / "x", False)
            s = calibrating()
            for t in s.trials:
                s.answer_elevation(t.truth_el)
            frame.finish_session()
            assert engine.hrir is chosen, "選定を終えても戻りません"
        lines.append("最後まで終えたとき")
    finally:
        frame.records_elev, frame.save_log = saved_store, saved_log
        _reset(frame)
        frame._active_tag = None
    return "すべて設定欄の HRTF に戻る: " + "、".join(lines)


@test("聞き比べは F6 で終わり、その後テンキーで鳴らない")
def t_listen_f6(frame: MainFrame) -> str:
    assert frame.ensure_player()
    try:
        frame.on_key(_key(wx.WXK_F2, shift=True))
        assert frame.mode == "listen"
        frame.on_key(_key(wx.WXK_F6))
        assert frame.mode == "idle", f"F6 の後も {frame.mode} のままです"
        assert frame.speech.log[-1] == "聞き比べを終えました。", \
            frame.speech.log[-1]
        frame.on_key(_key(wx.WXK_NUMPAD8))
        assert frame._handle is None, "F6 の後もテンキーで音が鳴りました"
    finally:
        _reset(frame)
    return "F6 で終了を読み上げ、その後のテンキー8では鳴らない"


@test("作り変えた音の保持は上限までで、元の音はそのまま返す")
def t_variant_cache_bound(frame: MainFrame) -> str:
    bank = SoundBank(48000)
    base = bank.get("noise_bursts")
    assert bank.variant("noise_bursts") is base
    cap = SoundBank.VARIANT_CACHE_MAX
    for k in range(cap + 30):
        bank.variant("noise_bursts", delay_s=0.001 * (k + 1))
    assert len(bank._variants) == cap, len(bank._variants)
    last = bank.variant("noise_bursts", delay_s=0.001 * (cap + 30))
    assert bank.variant("noise_bursts", delay_s=0.001 * (cap + 30)) is last
    return f"{cap + 30}種類作っても保持は{cap}個、同じ指定は同じ配列を返す"


@test("空間を選ぶと材質の響きになり、なしに戻すと残響の設定に戻る")
def t_space_choice(frame: MainFrame) -> str:
    from audiospace import BandFdnReverb

    assert frame.ensure_player()
    engine = frame.player.engine
    names = [frame.cmb_space.GetString(i)
             for i in range(frame.cmb_space.GetCount())]
    assert names[0] == "なし" and "コンクリートの部屋" in names, names
    try:
        frame.cmb_space.SetSelection(names.index("コンクリートの部屋"))
        frame.apply_settings()
        assert engine.space is not None and engine.space.name == \
            "コンクリートの部屋", engine.space
        assert isinstance(engine.reverb, BandFdnReverb), engine.reverb
        assert "空間はコンクリートの部屋" in frame.settings_text()
        frame.cmb_space.SetSelection(names.index("何もない道"))
        frame.apply_settings()
        assert engine.reverb is None, "何もない道に拡散した残響が付きました"
        assert engine.space.name == "何もない道"
    finally:
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    assert engine.space is None, "なしに戻しても空間が残りました"
    # 精密な残響: 閉じた部屋は焼き込み済みの尾、開いた空間は速い版のまま
    from audiospace import ConvolutionReverb
    try:
        frame.chk_precise.SetValue(True)
        frame.cmb_space.SetSelection(names.index("コンクリートの部屋"))
        frame.apply_settings()
        assert isinstance(engine.reverb, ConvolutionReverb), engine.reverb
        assert "精密版の残響" in frame.settings_text(), frame.settings_text()
        frame.cmb_space.SetSelection(names.index("建物の間の道"))
        frame.apply_settings()
        assert engine.late_status.startswith("開いた面のある空間"), \
            engine.late_status
    finally:
        frame.chk_precise.SetValue(False)
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    return (f"{len(names) - 1}種類。コンクリートの部屋で材質の残響、"
            "何もない道で残響なし、なしで元に戻る")


def _msaa_name_value(hwnd: int) -> tuple[str | None, str | None] | None:
    """スクリーンリーダーが読む MSAA の名前と値。取れなければ None。"""
    try:
        import ctypes
        from ctypes import POINTER, byref

        import comtypes.client
        comtypes.client.GetModule("oleacc.dll")
        from comtypes.gen.Accessibility import IAccessible
    except Exception:  # noqa: BLE001
        return None
    pointer = POINTER(IAccessible)()
    result = ctypes.oledll.oleacc.AccessibleObjectFromWindow(
        ctypes.c_void_p(hwnd), ctypes.c_ulong(0xFFFFFFFC),   # OBJID_CLIENT
        byref(IAccessible._iid_), byref(pointer))
    if result != 0 or not pointer:
        return None
    return pointer.accName(0), pointer.accValue(0)


def _focus_hwnd(ctrl: wx.Window) -> int:
    """フォーカスが入る子窓。SpinCtrl は内側の Edit に入る。"""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found: list[int] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _):
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, buf, 64)
        if buf.value == "Edit":
            found.append(hwnd)
        return True
    # wx の SpinCtrl は Edit を兄弟として作るので、親の子から探して
    # 生成順で SpinCtrl 本体の直前（または直後）にあるものを選ぶ
    user32.EnumChildWindows(ctrl.GetParent().GetHandle(), proc(callback), 0)
    own = ctrl.GetHandle()
    for hwnd in found:
        if user32.GetDlgCtrlID(hwnd) == user32.GetDlgCtrlID(own) or \
                user32.GetWindow(hwnd, 2) == own:          # GW_HWNDNEXT
            return hwnd
    return own


@test("数値の設定は整数のスピンで、名前と値がそのまま読める")
def t_numeric_spins(frame: MainFrame) -> str:
    """wx.Slider に SL_LABELS を付けると、目盛りの数字が Static として
    スライダーの直前に作られ、名前が「35」になっていた（値も範囲に対する
    割合で、0〜90 の 35 が「38」）。整数の SpinCtrl なら直前のラベルが
    名前になる。"""
    sliders = []

    def walk(win: wx.Window) -> None:
        for child in win.GetChildren():
            if isinstance(child, wx.Slider):
                sliders.append(child.GetName())
            walk(child)
    walk(frame)
    assert not sliders, f"wx.Slider が残っています: {sliders}"

    lines = []
    children = list(frame.panel.GetChildren())
    for ctrl, lo, hi, text in ((frame.spn_depth, 0, 90, "前後の強調"),
                               (frame.spn_tilt, 0, 100, "上下の強調")):
        assert isinstance(ctrl, wx.SpinCtrl), type(ctrl).__name__
        assert (ctrl.GetMin(), ctrl.GetMax()) == (lo, hi), \
            (ctrl.GetMin(), ctrl.GetMax())
        before = children[children.index(ctrl) - 1]
        assert isinstance(before, wx.StaticText), type(before).__name__
        label = before.GetLabel()
        assert text in label and "パーセント" in label, label
        msaa = _msaa_name_value(_focus_hwnd(ctrl))
        if msaa is not None:
            name, value = msaa
            assert name and text in name and "パーセント" in name, \
                f"MSAA の名前が「{name}」"
            assert value == str(ctrl.GetValue()), \
                f"MSAA の値が「{value}」、実際は {ctrl.GetValue()}"
            lines.append(f"{name}、{value}")
        else:
            lines.append(f"{label}（MSAA は確認できず）")

    # 矢印での増減（EVT_SPINCTRL）がエンジンに届く
    assert frame.ensure_player()
    try:
        frame.spn_depth.SetValue(60)
        event = wx.SpinEvent(wx.wxEVT_SPINCTRL, frame.spn_depth.GetId())
        event.SetEventObject(frame.spn_depth)
        frame.spn_depth.GetEventHandler().ProcessEvent(event)
        depth = frame.player.engine.front_back_depth
        assert abs(depth - 0.60) < 1e-9, depth
    finally:
        frame.spn_depth.SetValue(35)
        frame.apply_settings()
    return " / ".join(lines) + "。60 にするとエンジンの前後の強調が 0.6"


@test("回る向きの聴き取りは Ctrl+F2 で始まり、音が半周して止まる")
def t_rotation_mode(frame: MainFrame) -> str:
    from demo.quiz import RotationSession

    assert frame.ensure_player(), "無音モードでプレイヤーを開けません"
    try:
        frame.on_key(_key(wx.WXK_F2, ctrl=True))
        s = frame.session
        assert isinstance(s, RotationSession), \
            f"Ctrl+F2 で回る向きの聴き取りが始まりません: {frame.mode}"
        assert frame.mode == "rotation", frame.mode
        assert frame._orbit_timer is None, "F2 の試聴まで始まりました"
        assert s.include_overhead, "実測 Viking なのに頭上を出しません"
        intro = frame.txt_log.GetValue()
        assert "4で左回り" in intro and "8で後ろから前へ" in intro, intro
        assert "回る向きの聴き取りでは使わない" in frame.settings_text()
        assert frame.repeat_interval_ms() == 4000, frame.repeat_interval_ms()

        # 出題: 回り始めの位置で鳴り、時間とともに半周して止まる
        frame.play_current()
        trial = s.current()
        handle = frame._handle
        assert handle is not None and frame._rot_timer.IsRunning()
        engine = frame.player.engine
        start = engine._slots[handle].position
        assert np.allclose(start, trial.position(0.0, frame._yaw)), start
        frame._rot_t0 -= 1.5                     # 半分の時間が過ぎたことにする
        frame.on_rotation_tick()
        mid = engine._slots[handle].position
        assert np.allclose(mid, trial.position(0.5, frame._yaw), atol=0.2), \
            (mid, trial.position(0.5, frame._yaw))
        frame._rot_t0 -= 2.0
        frame.on_rotation_tick()
        assert frame._handle is None and not frame._rot_timer.IsRunning(), \
            "回り終えても音が止まりません"

        # 回答: テンキーの 4 6 8 2
        keys = {"left": wx.WXK_NUMPAD4, "right": wx.WXK_NUMPAD6,
                "to_front": wx.WXK_NUMPAD8, "to_back": wx.WXK_NUMPAD2}
        frame._accepting = True
        frame.on_key(_key(keys[trial.truth_motion]))
        assert trial.correct, (trial.truth_motion, trial.answer_motion)
        assert frame._accepting is False
    finally:
        _reset(frame)
    return (f"Ctrl+F2 で{s.n_trials}問（頭上{sum(t.kind == 'overhead' for t in s.trials)}問）。"
            "回り始めの位置で鳴り、3秒で半周して止まる。テンキーで正解を記録")


@test("回る向き: 上下の測定がない HRTF では水平だけにし、頭上のキーは不正解にしない")
def t_rotation_flat(frame: MainFrame) -> str:
    from demo.quiz import RotationSession

    assert frame.ensure_player()
    frame.hrir_supports_elevation = lambda: False
    try:
        frame.on_key(_key(wx.WXK_F2, ctrl=True))
        s = frame.session
        assert isinstance(s, RotationSession) and not s.include_overhead
        assert all(t.kind == "horizontal" for t in s.trials)
        assert "水平に回る音だけ" in frame.txt_log.GetValue()
        frame._accepting = True
        frame.on_key(_key(wx.WXK_NUMPAD8))
        assert not s.trials[0].answered, "頭上のキーで回答になりました"
        assert frame._accepting, "頭上のキーで受付が閉じました"
    finally:
        del frame.hrir_supports_elevation
        _reset(frame)
    return "水平だけの24問。8を押しても回答にならず、4か6を案内する"


@test("F1 でキー操作の説明の窓が開き、本文にフォーカスが移り、Escape で閉じる")
def t_help_dialog(frame: MainFrame) -> str:
    """以前は「状況」の欄に書き込むだけでフォーカスが動かず、押しても
    何も起きないように聞こえた。"""
    from demo.localization_app import HELP_TEXT, HelpDialog

    def dialogs() -> list:
        return [w for w in wx.GetTopLevelWindows()
                if isinstance(w, HelpDialog)]

    try:
        frame.on_key(_key(wx.WXK_F1))
        dlg = frame._help_dlg
        assert isinstance(dlg, HelpDialog) and dlg.IsShown(), "窓が開きません"
        assert dlg.text.GetValue() == HELP_TEXT
        assert dlg.text.IsEditable() is False, "本文が書き換えられます"
        assert dlg.text.GetInsertionPoint() == 0, "先頭から読めません"
        # 読み上げ名は直前の StaticText（操作まで含めた文）
        children = list(dlg.text.GetParent().GetChildren())
        before = children[children.index(dlg.text) - 1]
        assert isinstance(before, wx.StaticText)
        assert "Escape で閉じます" in before.GetLabel(), before.GetLabel()
        # F1 はキーとは別に「ヘルプの要求」でも届く。両方来ても窓は1つ
        frame.ProcessEvent(wx.HelpEvent(wx.wxEVT_HELP, frame.GetId()))
        frame.on_key(_key(wx.WXK_F1))
        assert len(dialogs()) == 1, f"窓が{len(dialogs())}個開きました"
        dlg.on_key(_key(wx.WXK_ESCAPE))
        wx.GetApp().ProcessPendingEvents()
        assert frame._help_dlg is None, "Escape で閉じません"
        # ヘルプの要求だけでも開く（スクリーンリーダーがキーを横取りした場合）
        frame.ProcessEvent(wx.HelpEvent(wx.wxEVT_HELP, frame.GetId()))
        assert isinstance(frame._help_dlg, HelpDialog), \
            "ヘルプの要求で窓が開きません"
        frame._help_dlg.Close()
        wx.GetApp().ProcessPendingEvents()
        assert frame._help_dlg is None
    finally:
        if frame._help_dlg is not None:
            frame._help_dlg.Close()
        _reset(frame)
    return ("F1 で窓が1つ開き、本文の先頭にフォーカス。読み取り専用。"
            "ヘルプの要求でも開き、Escape で閉じる")


@test("8方向の聞き比べは Ctrl+Shift+F2 で始まり、テンキーの方向で鳴る")
def t_listen8(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    try:
        frame.on_key(_key(wx.WXK_F2, shift=True, ctrl=True))
        assert frame.mode == "listen8", frame.mode
        assert frame.session is None, "採点のモードが始まりました"
        assert "8で前、2で後ろ" in frame.txt_log.GetValue()

        def heard() -> tuple:
            pos = engine._slots[frame._handle].position
            from audiospace.geometry import relative_spherical
            from audiospace import Listener
            az = relative_spherical(Listener(yaw_deg=frame._yaw), pos)[0]
            return round(az % 360.0, 3)

        lines = []
        for code, want in ((wx.WXK_NUMPAD8, 0.0), (wx.WXK_NUMPAD1, 135.0),
                           (wx.WXK_NUMPAD6, 270.0)):
            frame.on_key(_key(code))
            assert heard() == want, (code, heard(), want)
            lines.append(f"{want:g}度")
        # 向きを変えても、いまの自分から見た方向で鳴る
        frame._yaw = 90.0
        frame.on_key(_key(wx.WXK_NUMPAD2))
        assert heard() == 180.0, heard()
        frame._yaw = 0.0
        frame.on_key(_key(wx.WXK_NUMPAD5))          # 5 は使わない
        frame.on_key(_key(wx.WXK_F6))
        assert frame.mode == "idle", "F6 で聞き比べが終わりません"
        frame.on_key(_key(wx.WXK_F2, ctrl=True))
        assert frame.mode == "rotation", "Ctrl+F2 が回る向きのままではありません"
    finally:
        _reset(frame)
    return ("8・1・6 で " + "、".join(lines) + "。向きを変えても自分から見た方向。"
            "F6 で終了、Ctrl+F2 は回る向きのまま")


def _walk(win: wx.Window):
    for child in win.GetChildren():
        yield child
        yield from _walk(child)


@test("自分で作る窓: 欄はラベルの後に作られ、名前があり、数値は整数のスピン")
def t_editor_a11y(frame: MainFrame) -> str:
    try:
        frame.open_space_editor()
        frame.open_reverb_editor()
        lines = []
        for dlg in (frame._space_editor, frame._reverb_editor):
            ctrls = [w for w in _walk(dlg)
                     if isinstance(w, (wx.Choice, wx.TextCtrl, wx.SpinCtrl,
                                       wx.Button, wx.CheckBox))]
            assert not [w for w in _walk(dlg)
                        if isinstance(w, (wx.Slider, wx.SpinCtrlDouble))]
            for w in ctrls:
                assert w.GetName() and w.GetName() not in (
                    "choice", "text", "wxSpinCtrl", "button"), w
                if isinstance(w, wx.Button):
                    continue
                siblings = list(w.GetParent().GetChildren())
                before = siblings[siblings.index(w) - 1]
                assert isinstance(before, wx.StaticText), \
                    f"{w.GetName()} の前にラベルがありません"
            lines.append(f"{dlg.GetTitle()}: {len(ctrls)}個")
    finally:
        for dlg in (frame._space_editor, frame._reverb_editor):
            if dlg is not None:
                dlg.Close()
        wx.GetApp().ProcessPendingEvents()
    assert frame._space_editor is None and frame._reverb_editor is None
    return "、".join(lines)


@test("自分で作る空間: 保存すると選択肢に並んで選ばれ、エンジンに反映される")
def t_user_space(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    store = frame.user_presets
    try:
        frame.btn_space_edit.GetEventHandler().ProcessEvent(
            wx.CommandEvent(wx.wxEVT_BUTTON, frame.btn_space_edit.GetId()))
        dlg = frame._space_editor
        assert dlg is not None, "ボタンで窓が開きません"
        # 元にする: コンクリートの部屋
        idx = [k for k, _ in dlg.sources].index(("preset", "concrete_room"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        assert dlg.txt_name.GetValue() == ""
        assert "残響時間は" in dlg.txt_info.GetValue()
        dlg.spn_w.SetValue(800)
        floor = dlg.material_keys["floor"].index("carpet_soft_10mm")
        dlg.cmb_mat["floor"].SetSelection(floor)
        dlg.on_material("floor")
        dlg.save()                             # 名前が空なので断る
        assert store.spaces == {}, "名前なしで保存しました"
        dlg.txt_name.SetValue("じゅうたんの広い部屋")
        dlg.save()
        assert "じゅうたんの広い部屋" in store.spaces
        assert frame.cmb_space.GetStringSelection() == "自作: じゅうたんの広い部屋"
        assert engine.space is not None and engine.space.size[0] == 8.0
        assert engine.space.materials["floor"] == "carpet_soft_10mm"
        # 同じ名前で直して保存すると、エンジンにも反映される
        dlg.cmb_mat["floor"].SetSelection(dlg.material_keys["floor"].index("marble_floor"))
        dlg.on_material("floor")
        dlg.save()
        assert engine.space.materials["floor"] == "marble_floor"
        # 試しに鳴らすと一時的に変わり、閉じると選んでいる空間に戻る
        dlg.spn_h.SetValue(900)
        dlg.preview()
        assert engine.space.size[2] == 9.0
        assert frame._handle is not None, "試しの音が鳴りません"
        assert "自作: じゅうたんの広い部屋" in frame.settings_text()
        dlg.Close()
        wx.GetApp().ProcessPendingEvents()
        assert engine.space.size[2] == 2.7, engine.space.size   # 保存した高さ
        assert engine.space.materials["floor"] == "marble_floor"
        # 削除は2回押す
        frame.open_space_editor()
        dlg = frame._space_editor
        assert dlg.txt_name.GetValue() == "じゅうたんの広い部屋"
        dlg.delete()
        assert "じゅうたんの広い部屋" in store.spaces, "1回で消えました"
        dlg.delete()
        assert store.spaces == {}
        assert frame.cmb_space.GetSelection() == 0
        assert engine.space is None, "消した空間が残っています"
    finally:
        if frame._space_editor is not None:
            frame._space_editor.Close()
        wx.GetApp().ProcessPendingEvents()
        for name in list(store.spaces):
            store.delete_space(name)
        frame.refresh_user_choices(space_key=None)
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    return ("コンクリートの部屋を元に、幅8m・じゅうたんの床で保存→選ばれて反映。"
            "直して上書き、試しに鳴らして閉じると元に戻る、削除は2回")


@test("自分で作る残響: 保存すると残響の選択肢に並び、空間がなしのとき効く")
def t_user_reverb(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    store = frame.user_presets
    try:
        frame.open_reverb_editor()
        dlg = frame._reverb_editor
        idx = [k for k, _ in dlg.sources].index(("preset", "hall"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        from audiospace import preset_reverb_spec
        want = int(round(preset_reverb_spec("hall")["rt60"] * 1000))
        assert dlg.spn_rt.GetValue() == want, (dlg.spn_rt.GetValue(), want)
        dlg.spn_rt.SetValue(3200)
        dlg.spn_level.SetValue(150)
        dlg.txt_name.SetValue("広い石の響き")
        dlg.save()
        assert frame.cmb_room.GetStringSelection() == "自作: 広い石の響き"
        assert abs(engine.reverb.rt60 - 3.2) < 1e-9, engine.reverb.rt60
        assert "広い石の響き" in frame.settings_text()
        dlg.Close()
        wx.GetApp().ProcessPendingEvents()
        # 読み直しても選べる（ファイルに書けている）
        from demo.user_presets import UserPresetStore
        assert UserPresetStore(store.path).reverbs["広い石の響き"]["level"] == 1.5
    finally:
        if frame._reverb_editor is not None:
            frame._reverb_editor.Close()
        wx.GetApp().ProcessPendingEvents()
        for name in list(store.reverbs):
            store.delete_reverb(name)
        frame.refresh_user_choices(room_key=None)
        frame.cmb_room.SetSelection(0)
        frame.apply_settings()
    return "hall を元に残響時間3.2秒・大きさ150%で保存→選ばれて反映、読み直しても残る"


@test("設定欄の家具の量と人の数が、選んでいる空間に足される")
def t_furnish_setting(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    names = [frame.cmb_space.GetString(i)
             for i in range(frame.cmb_space.GetCount())]
    try:
        frame.cmb_space.SetSelection(names.index("コンクリートの部屋"))
        frame.apply_settings()
        assert engine.space.contents == {}, engine.space.contents
        base_rt = engine.space.rt60_bands()[3]
        frame.cmb_furnish.SetSelection(frame.furnish_keys.index("normal"))
        frame.spn_people.SetValue(5)
        frame.apply_settings()
        got = engine.space.contents
        assert got.get("person") == 5 and got.get("sofa") == 1, got
        assert engine.space.rt60_bands()[3] < base_rt
        text = frame.settings_text()
        assert "家具の量は普通" in text and "人の数は5" in text, text
        # 道では家具は足さず、人だけ足す
        frame.cmb_space.SetSelection(names.index("建物の間の道"))
        frame.apply_settings()
        assert engine.space.contents == {"person": 5}, engine.space.contents
    finally:
        frame.cmb_furnish.SetSelection(0)
        frame.spn_people.SetValue(0)
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    return (f"コンクリートの部屋に普通の家具と5人: 1kHz の残響時間 "
            f"{base_rt:.2f}秒から短く。道では人だけ")


@test("自分で作る空間: 面ごとに合う材質が上、家具の量で数が入り保存される")
def t_editor_contents(frame: MainFrame) -> str:
    store = frame.user_presets
    try:
        frame.open_space_editor()
        dlg = frame._space_editor
        floor_first = dlg.cmb_mat["floor"].GetString(0)
        ceil_first = dlg.cmb_mat["ceiling"].GetString(0)
        assert "床" in floor_first, floor_first
        assert dlg.material_keys["floor"][0] == "concrete_floor"
        assert dlg.material_keys["ceiling"][0] == "plasterboard"
        idx = [k for k, _ in dlg.sources].index(("preset", "wood_room"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        dlg.spn_contents["person"].SetValue(3)
        dlg.cmb_furnish.SetSelection(dlg.furnish_keys.index("full"))
        dlg.on_furnish()
        assert dlg.spn_contents["sofa"].GetValue() == 1
        assert dlg.spn_contents["person"].GetValue() == 3, "人の数が変わった"
        info = dlg.txt_info.GetValue()
        assert "置いてあるものが無ければ" in info and "混み具合" in info, info
        dlg.txt_name.SetValue("物の多い木の部屋")
        dlg.save()
        saved = store.spaces["物の多い木の部屋"]["contents"]
        assert saved["person"] == 3 and saved["bed"] == 1, saved
        assert frame.player is None or frame.player.engine.space.contents \
            .get("bed") == 1
    finally:
        if frame._space_editor is not None:
            frame._space_editor.Close()
        wx.GetApp().ProcessPendingEvents()
        for name in list(store.spaces):
            store.delete_space(name)
        frame.refresh_user_choices(space_key=None)
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    return (f"床の一覧の先頭は「{floor_first}」、天井は「{ceil_first}」。"
            "家具の量「多い」で数が入り、人の数はそのまま、保存にも残る")


@test("review: 回る音を途中で答えたら、切らずに短く消す。タイマーは使い回す")
def t_review_rotation_fade(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    timer = frame._rot_timer
    try:
        frame.on_key(_key(wx.WXK_F2, ctrl=True))
        frame.play_current()
        assert frame._rot_timer is timer and timer.IsRunning()
        handle = frame._handle
        trial = frame.session.current()
        keys = {"left": wx.WXK_NUMPAD4, "right": wx.WXK_NUMPAD6,
                "to_front": wx.WXK_NUMPAD8, "to_back": wx.WXK_NUMPAD2}
        frame._accepting = True
        frame.on_key(_key(keys[trial.truth_motion]))
        assert trial.answered
        slot = engine._slots.get(handle)
        assert slot is not None and slot._fade_step > 0.0, \
            "回る音をいきなり切りました"
        assert not timer.IsRunning() and frame._handle is None
        frame.play_current()                       # 次の問題
        assert frame._rot_timer is timer and timer.IsRunning(), \
            "タイマーを作り直しました"
    finally:
        _reset(frame)
    return "答えると 40ms で消え、次の問題も同じタイマーで回る"


@test("review: 空間が無いとき、残響は変わったときだけ作り直す")
def t_review_reverb_kept(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    try:
        frame.cmb_space.SetSelection(0)
        frame.cmb_room.SetSelection(frame.room_keys.index(("preset", "hall")))
        frame.apply_settings()
        first = engine.reverb
        assert first is not None
        frame.spn_depth.SetValue(50)
        frame.apply_settings()
        frame.spn_people.SetValue(3)
        frame.apply_settings()
        assert engine.reverb is first, "同じ残響を作り直しました"
        frame.cmb_room.SetSelection(frame.room_keys.index(("preset", "cave")))
        frame.apply_settings()
        assert engine.reverb is not first, "残響を変えても替わりません"
        second = engine.reverb
        # 空間を選んでから外すと、残響を設定し直す
        names = [frame.cmb_space.GetString(i)
                 for i in range(frame.cmb_space.GetCount())]
        frame.cmb_space.SetSelection(names.index("コンクリートの部屋"))
        frame.apply_settings()
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
        assert engine.reverb is not None and engine.reverb is not second
        assert engine.space is None
    finally:
        frame.spn_depth.SetValue(35)
        frame.spn_people.SetValue(0)
        frame.cmb_room.SetSelection(0)
        frame.cmb_space.SetSelection(0)
        frame.apply_settings()
    return "数値を変えても同じ残響のまま、残響を替えたときと空間を外したときだけ作り直す"


@test("review: 「この設定で鳴らす」は、設定欄の家具と人を足し、自分の正面で鳴らす")
def t_review_preview_extras(frame: MainFrame) -> str:
    assert frame.ensure_player()
    engine = frame.player.engine
    try:
        frame.spn_people.SetValue(4)
        frame._yaw = 90.0
        frame.open_space_editor()
        dlg = frame._space_editor
        idx = [k for k, _ in dlg.sources].index(("preset", "wood_room"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        dlg.preview()
        assert engine.space.contents.get("person") == 4, engine.space.contents
        pos = engine._slots[frame._handle].position
        from audiospace import Listener
        from audiospace.geometry import relative_spherical
        az = relative_spherical(Listener(yaw_deg=90.0), pos)[0]
        assert abs((az + 180.0) % 360.0 - 180.0) < 1e-6, az
    finally:
        if frame._space_editor is not None:
            frame._space_editor.Close()
        wx.GetApp().ProcessPendingEvents()
        frame._yaw = 0.0
        frame.spn_people.SetValue(0)
        _reset(frame)
        frame.apply_settings()
    return "人4人を足した響きで、向きを変えていても自分の正面から鳴る"


@test("review: 空間を作る窓の家具の量は、部屋のときだけ選べる")
def t_review_editor_furnish_room_only(frame: MainFrame) -> str:
    try:
        frame.open_space_editor()
        dlg = frame._space_editor
        idx = [k for k, _ in dlg.sources].index(("preset", "street"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        assert not dlg.cmb_furnish.IsEnabled()
        dlg.cmb_furnish.SetSelection(dlg.furnish_keys.index("normal"))
        dlg.on_furnish()
        assert dlg.spn_contents["sofa"].GetValue() == 0, "道に家具を入れました"
        idx = [k for k, _ in dlg.sources].index(("preset", "wood_room"))
        dlg.cmb_source.SetSelection(idx)
        dlg.on_source()
        assert dlg.cmb_furnish.IsEnabled()
    finally:
        if frame._space_editor is not None:
            frame._space_editor.Close()
        wx.GetApp().ProcessPendingEvents()
    return "道では家具の量を選べず、選んでも数は入らない。部屋では選べる"


@test("英語の表示: 画面・案内・読み上げ・結果に日本語が残っていない")
def t_english_ui(frame: MainFrame) -> str:
    # 言語は起動したときに決まるので、英語の画面は別のプロセスで作る
    out = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "tests" / "en_probe.py")],
        cwd=ROOT, env=dict(os.environ, AUDIOSPACE_SILENT="1",
                           AUDIOSPACE_LANG="en"),
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    assert out.returncode == 0, " / ".join(lines[-8:]) + out.stderr[-800:]
    return lines[0] if lines else ""


@test("マニュアルとはじめにお読みくださいが、今のコードと合っている")
def t_manual_current(frame: MainFrame) -> str:
    # 画面をもう1つ作るので、別のプロセスで調べる（この試験の画面を乱さない）
    out = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "tools" / "build_manual.py"),
         "--check"], cwd=ROOT, env=dict(os.environ, AUDIOSPACE_SILENT="1"),
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    assert out.returncode == 0, (
        "python -X utf8 tools/build_manual.py で作り直してください: "
        + " / ".join(lines[-4:]) + out.stderr[-800:])
    return lines[-1] if lines else ""


TESTS = [t_names, t_label_order, t_no_letter_keys, t_arrows_turn,
         t_updown_passthrough,
         t_wobble_choices, t_settings_text, t_numpad_geometry,
         t_sound_bank, t_hrtf_choices, t_welcome,
         t_default_pinna, t_answer_gate, t_ab_mode,
         t_elev_keys, t_elev_answer, t_elev_guard,
         t_shift_f3, t_repeat, t_repeat_toggle,
         # 状況表示を書き換えるので、初期表示を見る t_welcome より後に置く
         t_ctrl_f3, t_tilt, t_reference, t_listen,
         t_cues_default_off, t_roll_setting, t_floor, t_tilt_emphasis,
         t_calib_elev, t_restore_hrir, t_listen_f6, t_variant_cache_bound,
         t_space_choice, t_numeric_spins, t_rotation_mode, t_rotation_flat,
         t_help_dialog, t_listen8, t_editor_a11y, t_user_space,
         t_user_reverb, t_furnish_setting, t_editor_contents,
         t_review_rotation_fade, t_review_reverb_kept,
         t_review_preview_extras, t_review_editor_furnish_room_only,
         t_english_ui, t_manual_current]


def main() -> int:
    app = wx.App(False)
    # 自作の空間と残響は一時フォルダに保存する（利用者のファイルを書き換えない）
    presets_dir = tempfile.TemporaryDirectory()
    frame = MainFrame(preset_path=Path(presets_dir.name) / "user_presets.json")
    log(f"localization_app 構成テスト — wxPython {wx.version()}")
    log("=" * 74)
    for fn in TESTS:
        fn(frame)
        log()
    log("=" * 74)
    log(f"passed {_passed} / {_passed + _failed}"
        + (f", FAILED {_failed}" if _failed else ""))
    frame.Destroy()
    app.Destroy()
    OUT.write_text("\n".join(_log), encoding="utf-8")
    print(f"passed {_passed}/{_passed + _failed}; written: {OUT}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
