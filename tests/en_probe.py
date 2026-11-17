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
import time
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
from audiospace.quiz import (ABWobbleSession, CalibrationSession,  # noqa: E402
                       Difficulty, ElevationCalibrationSession,
                       ElevationSession, FrontElevationSession, QuizSession,
                       RotationSession, TrainingSession)
from audiospace.records import RecordStore  # noqa: E402

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

    from audiospace.quiz import ROTATION_REVERSE

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
        # ライブラリの耳介の選定（読み込みは合成の HRTF で代える）
        import numpy as np

        from audiospace import (MODE_ELEVATION, HrirSet, PinnaSelection,
                                SpatialPlayer)
        from audiospace.pinna import pinna_label
        take("pinna labels", [pinna_label("subj_E.sofa"),
                              pinna_label("subj_Z.sofa")])
        player = SpatialPlayer.open(block=512, max_sources=4, reverb=None,
                                    hrir=HrirSet.synthetic(sample_rate=48000))
        try:
            fake = [("a.sofa", "pinna A"), ("b.sofa", "pinna B")]
            sel_store = RecordStore(Path(tmp) / "sel.json")
            sel = PinnaSelection(player, fake, sel_store, loader=lambda t, r:
                                 HrirSet.synthetic(sample_rate=r))
            take("pinna intro", sel.intro())
            sel.prepare()
            take("pinna round", sel.round_text())
            try:
                sel.finish()
            except RuntimeError as exc:
                take("pinna not finished", str(exc))
            while not sel.finished:
                sel.play(np.zeros(48, np.float32))
                sel.answer(sel.current().truth_deg)
            take("pinna finish", sel.finish(save=False))
            flat = PinnaSelection(player, fake, sel_store, mode=MODE_ELEVATION,
                                  loader=lambda t, r:
                                  HrirSet.synthetic(sample_rate=r))
            take("pinna elevation intro", flat.intro())
            try:
                flat.prepare()
            except ValueError as exc:
                take("pinna no elevation", str(exc))
            try:
                PinnaSelection(player, fake[:1], sel_store)
            except ValueError as exc:
                take("pinna too few", str(exc))
        finally:
            player.stop()

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
    try:
        audiospace.reverb_from_spec({"rt60": 99, "width": 5}, 48000)
    except ValueError as exc:
        take("reverb_from_spec", str(exc))
    # 読めない音声ファイルの誤り（ファイルの名前は英語にしておく）
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "broken.wav"
        bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
        try:
            audiospace.load_mono(bad)
        except audiospace.DecodeError as exc:
            # soundfile の文は一時フォルダのパスを含む（利用者名が日本語の
            # PC もある）ので、パスを除いてから調べる
            take("decode error", re.sub(r"'[^']*'", "'<path>'", str(exc)))
    player = SpatialPlayer.open(block=512, max_sources=4)
    try:
        take("player status", player.status())
        player.engine.source_errors = 1
        player.engine.last_source_error = "RuntimeError: x"
        player.block_mismatches = 1
        player.last_mismatch_frames = 256
        # 鳴り終わりの関数の誤りと、届けていない知らせ
        player.callback_errors = 1
        player.last_callback_error = "ValueError: x"
        import numpy as np
        player.play(np.zeros(480, np.float32), on_end=lambda e: None)
        for _ in range(200):
            if player.engine.pending_ends():
                break
            time.sleep(0.01)
        with player.engine._lock:
            player.engine._elapsed += 5.0
        # 長いファイルを少しずつ読む音: 読み込みが間に合わなかった数と、誤り。
        # 一時フォルダは終わったら消す（少しずつ読む音を止めてから）
        import soundfile as sf
        stream_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        wav = Path(stream_dir.name) / "stream.wav"
        sf.write(wav, np.zeros((4800, 2), np.float32), 44100)
        streamed = player.play(wav, spatialise=False, stream=True, loop=True)
        source = player.engine.get_source(streamed).source
        source._state.underruns = 2
        # 空間の無い自動の遮蔽も、状態の文に出る
        player.set_auto_occlusion()
        take("player status errors", player.status())
        player.set_auto_occlusion(False)
        player.stop_sound(streamed)
        source.close()
        try:
            player.play(wav, stream=True, rate=1.5)
        except ValueError as exc:
            take("stream and rate", str(exc))
        try:
            player.play(np.zeros(48, np.float32), stream=True)
        except ValueError as exc:
            take("stream not a file", str(exc))
        source._thread.join(timeout=2.0)
        stream_dir.cleanup()
        try:
            audiospace.FileStreamSource(Path(__file__), 48000)
        except audiospace.DecodeError as exc:
            take("stream decode", re.sub(r"'[^']*'", "'<path>'", str(exc)))
        player.poll_ends()
        player.callback_errors = 0
        player.reset_stats()
        # 出力デバイスの見張りの知らせ（実機を模す。デバイスは開かない）
        from audiospace.device import DeviceChoice
        state = {"lost": True, "fail": None}

        def reopen(*a, **k):
            if state["fail"] is not None:
                raise state["fail"]
            state["lost"] = False
            return True

        player.silent = False
        player.check_device = lambda: state["lost"]
        player.reopen = reopen
        player.choice = DeviceChoice(0, "Dev", "Windows WASAPI", 48000, 10.0)
        events = [player.poll_device()]
        state.update(lost=True, fail=RuntimeError("no device"))
        events.append(player.poll_device())
        player._retry_at = 0.0
        state["fail"] = audiospace.DeviceRateError("44100 Hz")
        events.append(player.poll_device())
        take("device events", [e.message for e in events if e is not None])
        player._device_lost = True
        player.last_error = "x"
        take("status lost", player.status())
        player._device_lost = False
        del player.check_device, player.reopen
        player.choice = None
        player.silent = True
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
        # 前回の設定の読み込みの文も英語で出るように、使えない値を混ぜて置く
        settings = Path(tmp) / "app_settings.json"
        settings.write_text('{"version": 1, "settings": {"hrtf": "nope.sofa", '
                            '"difficulty": "EIGHT"}}', encoding="utf-8")
        frame = MainFrame(preset_path=Path(tmp) / "user_presets.json",
                          settings_path=settings)
        take("restored settings", frame.txt_log.GetValue().split("\n\n")[0])
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

            # 出力デバイスが失われたときの読み上げ（つなぎ直せた・できない・
            # レートが違うので作り直した）
            player = frame.player
            state = {"lost": True, "fail": None}

            def reopen(*a, **k):
                if state["fail"] is not None:
                    raise state["fail"]
                state["lost"] = False
                return True

            player.check_device = lambda: state["lost"]
            player.reopen = reopen
            player.silent = False       # 実機を模す（デバイスは開かない）
            frame.speech.log.clear()
            frame.check_output()
            state.update(lost=True, fail=RuntimeError("no device"))
            frame.check_output()
            state["fail"] = audiospace.DeviceRateError("44100 Hz")
            player._retry_at = 0.0
            frame.check_output()
            take("device lost", [frame.speech.log, frame.txt_log.GetValue()])
            player.silent = True
            frame._stop_repeat()
            frame.stop_sound()

            # 保存できない理由（区切りも英語の「; 」になる）
            for put in (frame.user_presets.put_space,
                        frame.user_presets.put_reverb):
                try:
                    put("", {"kind": "?", "rt60": 99})
                except ValueError as exc:
                    take("put problems", str(exc))

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
