"""英語の表示に日本語が残っていないかを調べる（GUI構成のテストが別のプロセスで呼ぶ）。

    python -X utf8 tests\\en_probe.py

表示の言語を英語にして（AUDIOSPACE_LANG=en）本物の画面を作り、利用者に
見える文を集める: 画面の部品の名前・ラベル・選択肢、F1 の説明、各モードを
始めたときの案内、設定の読み上げ、答えた後の読み上げ、結果の文、自分で
作る窓の欄と説明、ライブラリの状態の文。集めた文に日本語の文字（かな・
漢字・全角の記号）があれば、その文を書き出して 1 で終わる。

音は出さない（AUDIOSPACE_SILENT=1、デバイスを開かない）。自作の空間と
残響の保存先と、耳介の通算成績の保存先は一時フォルダにする（利用者の
ファイルを読み書きしない）。結果のファイル（logs/）も書かない。
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["AUDIOSPACE_SILENT"] = "1"
os.environ["AUDIOSPACE_LANG"] = "en"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wx  # noqa: E402

import audiospace  # noqa: E402
from audiospace import (FURNISH_LEVELS, SpatialPlayer, describe_presets,  # noqa: E402
                        furnish_counts, material_choices, reverb_text,
                        space_from_spec, space_labels, space_preset,
                        space_spec_problems, reverb_spec_problems)
from demo import localization_app as app_mod  # noqa: E402
from demo.localization_app import MainFrame, answer_text  # noqa: E402
from demo.preset_editor import ReverbEditor, SpaceEditor  # noqa: E402
from demo.quiz import (ABWobbleSession, CalibrationSession,  # noqa: E402
                       Difficulty, ElevationCalibrationSession,
                       ElevationSession, FrontElevationSession, QuizSession,
                       RotationSession, TrainingSession)
from demo.records import RecordStore  # noqa: E402

#: 日本語の文字（かな・漢字・全角の記号と句読点）
JAPANESE = re.compile(r"[　-ヿ㐀-鿿＀-￯]")

_seen: list[tuple[str, str]] = []


def take(where: str, text) -> None:
    if text is None:
        return
    if isinstance(text, (list, tuple)):
        for item in text:
            take(where, item)
        return
    _seen.append((where, str(text)))


def widgets(where: str, window: wx.Window) -> None:
    """窓の中の部品の名前・ラベル・選択肢・値を集める。"""
    take(f"{where} title", window.GetTitle() if hasattr(window, "GetTitle")
         else "")
    take(f"{where} name", window.GetName())
    for child in window.GetChildren():
        name = f"{where}/{type(child).__name__}"
        take(name, child.GetName())
        if isinstance(child, (wx.StaticText, wx.Button, wx.CheckBox)):
            take(name, child.GetLabel())
        if isinstance(child, (wx.Choice, wx.ListBox)):
            take(name, [child.GetString(i) for i in range(child.GetCount())])
        if isinstance(child, wx.TextCtrl):
            take(name, child.GetValue())
        widgets(name, child)


def answer_all(session, pick) -> None:
    """全問に答える（pick(trial) が答えを返す）。"""
    while not session.finished:
        trial = session.current()
        pick(session, trial)


def sessions() -> None:
    """各モードのセッションに答えて、結果の文を集める。"""
    def az(s, t):
        s.answer((t.truth_deg + (0 if t.index % 3 else 180)) % 360)

    for diff in Difficulty:
        take("difficulty", diff.label)
        quiz = QuizSession(difficulty=diff, n_trials=12, seed=1)
        take("quiz detail", quiz.detail_lines())
        take("quiz empty", quiz.summary())
        answer_all(quiz, az)
        take("quiz", [quiz.summary(), quiz.detail_lines()])

    ab = ABWobbleSession(difficulty=Difficulty.EIGHT, n_trials=16, seed=2)
    answer_all(ab, az)
    take("ab", [ab.summary(), ab.detail_lines()])

    training = TrainingSession(difficulty=Difficulty.EIGHT, target_streak=3,
                               max_trials=12, seed=3)
    while not training.finished:
        t = training.current()
        wrong = t.index % 2 == 0
        t2 = training.answer((t.truth_deg + (180 if wrong else 0)) % 360)
        take("training feedback", training.feedback(t2))
    take("training", [training.summary(), training.detail_lines()])

    pool = [("subj_A.sofa", "Measured Viking pinna A"),
            ("subj_B.sofa", "Measured Viking pinna B")]
    calib = CalibrationSession(pool, difficulty=Difficulty.EIGHT,
                               trials_per_candidate=8, seed=4)
    answer_all(calib, az)
    take("calib", [calib.summary(), calib.detail_lines()])

    def el(s, t):
        s.answer_elevation(-t.truth_el if t.index % 2 else t.truth_el)

    for session in (ElevationSession(difficulty=Difficulty.EIGHT,
                                     n_trials=12, seed=5),
                    FrontElevationSession(n_trials=8, seed=6),
                    ElevationCalibrationSession(pool, trials_per_candidate=8,
                                                seed=7)):
        take("elevation detail", session.detail_lines())
        answer_all(session, el)
        take("elevation", [session.summary(), session.detail_lines()])

    from demo.quiz import ROTATION_REVERSE

    def rot(s, t):
        s.answer_rotation(ROTATION_REVERSE[t.truth_motion] if t.index % 2
                          else t.truth_motion)

    rotation = RotationSession(n_trials=12, seed=8)
    take("rotation detail", rotation.detail_lines())
    answer_all(rotation, rot)
    take("rotation", [rotation.summary(), rotation.detail_lines()])

    with tempfile.TemporaryDirectory() as tmp:
        store = RecordStore(Path(tmp) / "records.json")
        take("records empty", [store.summary(pool), store.progress_text(pool)])
        for _ in range(2):
            store.update(calib.scores(), "subj_A.sofa", n_directions=8)
        take("records", [store.summary(pool), store.detail_lines()])
        elev_store = RecordStore(Path(tmp) / "records_elev.json",
                                 fb_name="up/down")
        elev_store.update(calib.scores(), None, n_directions=8)
        take("records elev", [elev_store.summary(pool),
                              elev_store.detail_lines()])

    for correct in (True, False):
        take("answer", answer_text(3, correct, "", "front"))


def library() -> None:
    take("spaces", [v for _, v in space_labels()])
    take("furnish levels", [v for _, v in FURNISH_LEVELS])
    take("furnishings", [v for _, v in audiospace.furnishing_labels()])
    take("face labels", list(audiospace.FACE_LABELS.values()))
    take("space kinds", [v for _, v in audiospace.SPACE_KINDS])
    take("materials", [v for _, v in material_choices("floor")])
    take("wobble", [v for _, v in audiospace.wobble_labels()])
    take("roll", [v for _, v in audiospace.roll_labels()])
    take("wobble describe", [w.describe() for w in
                             audiospace.WOBBLE_PRESETS.values()])
    take("roll describe", [w.describe() for w in
                           audiospace.ROLL_PRESETS.values()])
    take("presets", describe_presets())
    for key, _ in space_labels():
        space = space_preset(key)
        take("reverb text", reverb_text(space))
        full = space.with_contents(furnish_counts(20.0, "full"))
        take("reverb text full", [reverb_text(full),
                                  audiospace.contents_text(full.contents)])
    take("contents none", audiospace.contents_text({}))
    take("space problems", space_spec_problems(
        {"kind": "room", "size": [0, 1], "materials": {"floor": "x"},
         "at": [2, 2], "contents": {"sofa": -1, "ghost": 1},
         "early_order": 99, "early_gain": 9, "late_gain": -1,
         "name": "x" * 99}))
    take("space problems", space_spec_problems({"kind": "?"}))
    take("space problems", space_spec_problems([]))
    take("reverb problems", reverb_spec_problems(
        {"rt60": "x", "width": 9, "name": "x" * 99}))
    take("reverb problems", reverb_spec_problems({}))
    try:
        space_from_spec({"kind": "room"})
    except ValueError as exc:
        take("space_from_spec", str(exc))
    player = SpatialPlayer.open(block=512, max_sources=4)
    try:
        take("player status", player.status())
        for late in (False, True, "precise"):
            for key in ("wood_room", "street", "open_road"):
                player.set_space(space_preset(key), late=late)
                take("late status", player.engine.late_status)
            player.set_space(space_preset("wood_room").with_contents(
                {"sofa": 1}), late=late)
            take("late status", player.engine.late_status)
    finally:
        player.stop()


def frame_texts(app: wx.App) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # 耳介の通算成績を一時フォルダに向ける（利用者の記録を読み書きしない）
        app_mod.LOG_DIR = Path(tmp) / "logs"
        frame = MainFrame(preset_path=Path(tmp) / "user_presets.json")
        frame.records = RecordStore(Path(tmp) / "records.json")
        frame.records_elev = RecordStore(Path(tmp) / "records_elev.json",
                                         fb_name="up/down")
        try:
            widgets("frame", frame)
            take("status bar", frame.GetStatusBar().GetStatusText())
            take("help", app_mod.HELP_TEXT)
            frame.show_help()
            widgets("help dialog", frame._help_dlg)
            frame._help_dlg.Close()
            assert frame.ensure_player()
            take("ready", frame.txt_log.GetValue())
            starts = [frame.toggle_orbit, frame.toggle_orbit,
                      frame.start_listen, frame.start_listen8,
                      frame.start_quiz, frame.start_training,
                      frame.start_calibration, frame.start_ab,
                      frame.start_elevation, frame.start_elevation_front,
                      frame.start_calibration_elev, frame.start_rotation]
            for start in starts:
                frame.speech.log.clear()
                start()
                take(start.__name__, [frame.txt_log.GetValue(),
                                      frame.speech.log,
                                      frame.settings_text(),
                                      frame.GetStatusBar().GetStatusText()])
                if frame.session is not None:
                    frame.session.current()
                    take("progress", frame._progress_text())
            frame.speech.log.clear()
            for action in (frame.toggle_repeat, frame.toggle_repeat,
                           frame.toggle_reference, frame.toggle_reference,
                           frame.stop_and_say, frame.announce_settings,
                           frame.announce_records, frame.announce_result):
                action()
            frame.turn(wx.WXK_LEFT)
            for _ in range(8):
                frame.tilt(wx.WXK_PAGEUP)
            frame.tilt(wx.WXK_PAGEDOWN)
            frame._elevation_answer_text()
            take("actions", [frame.speech.log, frame.settings_text(),
                             frame._elevation_answer_text(),
                             frame.GetStatusBar().GetStatusText()])
            frame._stop_repeat()
            frame.stop_sound()
            frame.session = None
            frame.speech.log.clear()
            frame.replay()
            frame.announce_result()
            # 上下の測定の無い HRTF（合成）で上下の聴き取りを断る文
            frame.cmb_hrtf.SetSelection(0)
            frame.on_hrtf_changed(wx.CommandEvent())
            frame.start_elevation()
            frame.start_rotation()
            take("refuse", [frame.speech.log, frame.txt_log.GetValue()])
            frame._stop_repeat()
            frame.stop_sound()

            for editor_cls in (SpaceEditor, ReverbEditor):
                dlg = editor_cls(frame)
                try:
                    dlg.select_source(dlg.sources[0][0])
                    widgets(editor_cls.__name__, dlg)
                    frame.speech.log.clear()
                    dlg.txt_name.SetValue("")
                    dlg.save()
                    dlg.delete()
                    dlg.txt_name.SetValue("probe")
                    dlg.save()
                    dlg.save()
                    dlg.delete()
                    dlg.delete()
                    dlg.preview()
                    take(f"{editor_cls.__name__} speech", frame.speech.log)
                    if editor_cls is SpaceEditor:
                        for i, kind in enumerate(dlg.kind_keys):
                            dlg.cmb_kind.SetSelection(i)
                            dlg.on_kind()
                            take("space info", dlg.txt_info.GetValue())
                        dlg.cmb_kind.SetSelection(0)
                        dlg.cmb_furnish.SetSelection(3)
                        dlg.on_furnish()
                        take("space info furnished", dlg.txt_info.GetValue())
                        dlg.spn_w.SetValue(50)
                        dlg.spn_h.SetValue(50)
                        dlg.update_info()
                        take("space info bad", dlg.txt_info.GetValue())
                    else:
                        take("reverb info", dlg.txt_info.GetValue())
                finally:
                    dlg.Destroy()
            frame.end_preview()
        finally:
            frame._stop_repeat()
            frame.stop_sound()
            if frame.player is not None:
                frame.player.stop()
            frame.Destroy()


def main() -> int:
    app = wx.App(False)
    assert audiospace.language() == "en", audiospace.language()
    library()
    sessions()
    frame_texts(app)
    bad = [(w, t) for w, t in _seen if JAPANESE.search(t)]
    print(f"checked {len(_seen)} texts in English")
    for where, text in bad[:40]:
        print(f"JAPANESE in {where}: {text[:200]!r}")
    if bad:
        print(f"found {len(bad)} texts with Japanese")
        return 1
    print("no Japanese found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
