"""Two-camera desktop interface for the official SO-101 following loop."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from so101_camera_session import CameraWorker, Session

JOINT_LABELS = (
    ("shoulder_pan", "台座の旋回", "°"),
    ("shoulder_lift", "肩", "°"),
    ("elbow_flex", "肘", "°"),
    ("wrist_flex", "手首の曲げ", "°"),
    ("wrist_roll", "手首の回転", "°"),
    ("gripper", "グリッパー", "%"),
)


def ask_save(window, session, error_text=""):
    """Ask independently about both recordings after motor/camera shutdown."""
    dialog = tk.Toplevel(window)
    dialog.title("保存するものを選んでください")
    dialog.transient(window)
    dialog.grab_set()
    body = ttk.Frame(dialog, padding=20)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="追従と録画を終了しました。保存するものを選んでください。").pack(anchor="w")
    if error_text:
        ttk.Label(body, text=error_text, wraplength=640, foreground="#a02020").pack(anchor="w", pady=10)
    video = tk.BooleanVar(value=True)
    log = tk.BooleanVar(value=True)
    ttk.Checkbutton(body, text="2台のカメラ動画を保存する", variable=video).pack(anchor="w", pady=(12, 4))
    ttk.Checkbutton(body, text="パラメータ・実行ログを保存する", variable=log).pack(anchor="w", pady=4)
    ttk.Label(body, text=f"保存先：{session.directory}", wraplength=640).pack(anchor="w", pady=12)
    result = [True, True]

    def accept():
        result[:] = [video.get(), log.get()]
        dialog.destroy()

    # Closing the choice dialog retains the files rather than discarding data.
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    ttk.Button(body, text="選択を確定", command=accept).pack(anchor="e")
    dialog.wait_window()
    return tuple(result)


def run_ui(args, build_arms, run_follow, root_path):
    """Run independent sessions in one window; restart only after save and joins."""
    window = tk.Tk()
    window.title("SO-101 追従・2台のカメラ録画")
    window.geometry("1320x840")
    window.minsize(800, 640)
    global_stop = getattr(args, "stop_event", None) or threading.Event()
    active = None
    mode = "starting"
    app_exit_code = 0
    images = {}
    status = tk.StringVar(value="カメラを準備しています…")
    rate = tk.StringVar(value="追従速度：未計測 Hz")
    recording_path = tk.StringVar(value="")
    saved_path = tk.StringVar(value="保存先：まだ保存していません")

    outer = ttk.Frame(window, padding=10)
    outer.pack(fill="both", expand=True)
    status_label = ttk.Label(outer, textvariable=status, font=("Yu Gothic UI", 12), wraplength=1260)
    status_label.pack(anchor="w")
    ttk.Label(outer, text="Q / Esc キー・停止ボタンで停止して保存を選択", wraplength=1260).pack(anchor="w", pady=(3, 0))
    ttk.Label(outer, textvariable=recording_path, wraplength=1260).pack(anchor="w", pady=(0, 3))
    ttk.Label(outer, textvariable=saved_path, wraplength=1260).pack(anchor="w", pady=(0, 8))
    previews = ttk.Frame(outer)
    previews.pack(fill="both", expand=True)
    previews.columnconfigure(0, weight=1)
    previews.columnconfigure(1, weight=1)
    previews.rowconfigure(0, weight=1)
    canvases = []
    for column, index in enumerate(args.camera_indices):
        panel = ttk.LabelFrame(previews, text=f"カメラ {index}", padding=3)
        panel.grid(row=0, column=column, sticky="nsew", padx=3)
        canvas = tk.Canvas(panel, bg="#171717", highlightthickness=0, width=640, height=480)
        canvas.pack(fill="both", expand=True)
        canvas.create_text(320, 240, text="準備中", fill="white", tags="waiting")
        canvases.append(canvas)
    numbers = ttk.Frame(outer, padding=(8, 10))
    numbers.pack(fill="x")
    ttk.Label(numbers, textvariable=rate).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    for column, text in enumerate(("関節", "黒い側の入力", "白い側へ送った目標")):
        ttk.Label(numbers, text=text, font=("Yu Gothic UI", 10, "bold")).grid(row=1, column=column, sticky="w", padx=12)
        numbers.columnconfigure(column, weight=1)
    values = {}
    for row, (joint, label, unit) in enumerate(JOINT_LABELS, 2):
        ttk.Label(numbers, text=f"{label} ({unit})").grid(row=row, column=0, sticky="w", padx=12)
        for column in (1, 2):
            value = tk.StringVar(value="—")
            values[joint, column] = value
            ttk.Label(numbers, textvariable=value, font=("Consolas", 12)).grid(row=row, column=column, sticky="w", padx=12)
    footer = ttk.Frame(outer)
    footer.pack(fill="x")

    def set_status(text, *, error=False):
        status.set(text)
        status_label.configure(foreground="#a02020" if error else "#202020")

    def request_stop(_event=None):
        nonlocal mode
        if active is None or mode not in ("running", "stopping"):
            return
        if mode == "stopping":
            return
        mode = "stopping"
        active["stop"].set()
        set_status("停止処理中… モーターのOFFと録画の終了を待っています。")
        stop_button.configure(state="disabled")

    def request_close():
        if mode in ("saved", "save_failed", "start_failed"):
            window.destroy()
        elif mode in ("running", "stopping"):
            request_stop()
        # Saving is modal: ignore repeated main-window close/stop requests.

    stop_button = ttk.Button(footer, text="停止して保存を選択 (Q)", command=request_stop)
    stop_button.pack(side="right", padx=3)
    close_button = ttk.Button(footer, text="閉じる", state="disabled", command=request_close)
    close_button.pack(side="right", padx=3)
    next_button = ttk.Button(footer, text="次のセッションを開始", state="disabled")
    next_button.pack(side="right", padx=3)
    ttk.Label(footer, text="白い側の数値は送信目標です。実際の位置ではありません。").pack(side="left")
    window.bind_all("<KeyPress-q>", request_stop)
    window.bind_all("<KeyPress-Q>", request_stop)
    window.bind_all("<Escape>", request_stop)
    window.protocol("WM_DELETE_WINDOW", request_close)

    def start_session(*, restart=False):
        nonlocal active, mode, app_exit_code
        if restart:
            if mode != "saved" or active is None or not active.get("restart_allowed", False):
                return
            if active["worker"].is_alive() or any(camera.is_alive() for camera in active["cameras"]):
                return
            # Only a user's explicit restart releases a previous signal request.
            global_stop.clear()
        elif active is not None:
            return
        mode = "starting"
        next_button.configure(state="disabled")
        close_button.configure(state="disabled")
        stop_button.configure(state="disabled")
        set_status("新しいセッションのカメラを準備しています…")
        rate.set("追従速度：未計測 Hz")
        for value in values.values():
            value.set("—")
        for canvas in canvases:
            canvas.delete("all")
            canvas.create_text(320, 240, text="準備中", fill="white", tags="waiting")
        images.clear()
        context = {"stop": threading.Event(), "events": queue.Queue(),
                   "latest_lock": threading.Lock(), "latest": {},
                   "finished": threading.Event(), "save_requested": False,
                   "restart_allowed": False, "cameras": [], "worker": None,
                   "result": {"exit_code": 0, "error": "", "off_confirmed": False,
                              "drive_started": False, "release_needed": False}}
        active = context
        try:
            session = Session(args.output_root, script_name="follow-so101")
            context["session"] = session
            recording_path.set(f"今回の記録先：{session.directory}")
            context["cameras"] = [CameraWorker(index, session, context["stop"], fps=args.camera_fps)
                                  for index in args.camera_indices]
        except Exception as exc:
            app_exit_code = 1
            mode = "start_failed"
            set_status(f"セッションの準備に失敗しました：{exc}", error=True)
            if "session" in context:
                context["session"].close_log()
            close_button.configure(state="normal")
            return
        stop = context["stop"]
        if global_stop.is_set():
            stop.set()
        events = context["events"]
        result = context["result"]
        cameras = context["cameras"]

        def frame_update(row):
            with context["latest_lock"]:
                context["latest"].update(row)

        def on_frame(row):
            session.emit({"phase": "parameters", **row})
            frame_update(row)

        def emit(row):
            session.emit(row)
            if "leader_action" in row:
                frame_update(row)
            if row.get("phase") == "stopped":
                result["off_confirmed"] = row.get("off_confirmed", False)
                result["drive_started"] = row.get("drive_started", False)
                result["release_needed"] = row.get("release_needed", False)
            events.put(row)

        context["emit"] = emit

        def controller():
            try:
                emit({"phase": "starting", "source": "https://huggingface.co/docs/lerobot/il_robots",
                      "preview_only": args.preview_only, "fps_target": args.fps,
                      "camera_indices": args.camera_indices, "camera_fps": args.camera_fps})
                for camera in cameras:
                    if stop.is_set():
                        break
                    camera.start()
                    ready = camera.wait_ready(timeout=8)
                    if stop.is_set():
                        break
                    if not ready:
                        raise RuntimeError(f"カメラ {camera.index} の初回映像を取得できませんでした。")
                if not stop.is_set():
                    if args.preview_only:
                        emit({"phase": "camera_preview"})
                        stop.wait()
                    else:
                        emit({"phase": "connecting"})
                        leader, follower = build_arms(args.leader_port, args.follower_port)
                        emit({"phase": "ports_resolved", "leader_port": leader.bus.port,
                              "follower_port": follower.bus.port})
                        run_follow(leader, follower, fps=args.fps, stop_event=stop,
                                   emit=emit, frame_emit=on_frame)
            except Exception as exc:
                result["exit_code"] = 1
                result["error"] = str(exc)
                try:
                    emit({"phase": "ui_fault", "error": repr(exc)})
                except Exception:
                    pass
            finally:
                stop.set()
                for camera in cameras:
                    try:
                        camera.stop()
                    except Exception as exc:
                        result["exit_code"] = 1
                        result["error"] += f"\nカメラ {camera.index} の停止：{exc}"
                for camera in cameras:
                    try:
                        camera.join(timeout=8)
                        while camera.is_alive():
                            events.put({"phase": "camera_closing"})
                            camera.join(timeout=1)
                    except Exception as exc:
                        result["exit_code"] = 1
                        result["error"] += f"\nカメラ {camera.index} の終了待ち：{exc}"
                    if camera.error:
                        result["exit_code"] = 1
                        result["error"] += f"\nカメラ {camera.index}: {camera.error}"
                context["finished"].set()

        context["worker"] = threading.Thread(target=controller, name="so101-follow-controller", daemon=False)
        mode = "running"
        stop_button.configure(text="停止して保存を選択 (Q)", state="normal")
        try:
            context["worker"].start()
        except Exception as exc:
            mode = "start_failed"
            app_exit_code = 1
            session.close_log()
            set_status(f"セッションの開始に失敗しました：{exc}", error=True)
            stop_button.configure(state="disabled")
            close_button.configure(state="normal")

    def save_finished_session(context):
        nonlocal mode, app_exit_code
        context["save_requested"] = True
        mode = "saving"
        result, session = context["result"], context["session"]
        context["worker"].join()
        if any(camera.is_alive() for camera in context["cameras"]):
            mode = "save_failed"
            result["exit_code"] = app_exit_code = 1
            set_status(f"カメラの終了を確認できませんでした。再開を止めています。\n記録の確認先：{session.directory}", error=True)
            close_button.configure(state="normal")
            return
        off_missing = (not args.preview_only and
                       (result["drive_started"] or result["release_needed"]) and
                       not result["off_confirmed"])
        if off_missing:
            result["error"] += "\n全モーターのOFFを確認できませんでした。再開前に接続状態を確認してください。"
            result["exit_code"] = 1
        set_status("追従・録画は終了しました。保存するものを選んでください。")
        stop_button.configure(state="disabled")
        try:
            save_video, save_log = ask_save(window, session, result["error"])
            context["emit"]({"phase": "save_selection", "save_video": save_video, "save_log": save_log})
            session.close_log()
            paths = session.finish(save_video=save_video, save_log=save_log)
            videos = sum(path.suffix.lower() == ".avi" for path in paths)
            logs = sum(path.name == "session.jsonl" for path in paths)
            summary = f"保存完了：動画{videos}本／ログ{logs}件" if paths else "保存せず終了"
            saved_path.set(f"保存先：{session.directory}" if paths else "直前のセッションは保存していません")
            if result["error"]:
                summary += f"\n{result['error']}"
            set_status(summary, error=bool(result["error"]))
            mode = "saved"
            context["restart_allowed"] = not off_missing
            if context["restart_allowed"]:
                next_button.configure(state="normal")
        except Exception as exc:
            result["exit_code"] = 1
            mode = "save_failed"
            context["restart_allowed"] = False
            try:
                session.close_log()
            except Exception:
                pass
            set_status(f"保存処理に失敗しました：{exc}\n記録は {session.directory} に残しています。", error=True)
            saved_path.set(f"記録の確認先：{session.directory}")
        app_exit_code = max(app_exit_code, result["exit_code"])
        close_button.configure(state="normal")

    next_button.configure(command=lambda: start_session(restart=True))

    def refresh():
        context = active
        if context is not None and context.get("worker") is not None:
            if mode in ("running", "stopping") and global_stop.is_set():
                request_stop()
            while True:
                try:
                    row = context["events"].get_nowait()
                except queue.Empty:
                    break
                phase = row.get("phase")
                if phase == "following" and mode == "running" and not context["stop"].is_set():
                    set_status("追従・2台録画中　画面を選択して Q キーで停止")
                elif phase == "camera_preview" and mode == "running" and not context["stop"].is_set():
                    set_status("カメラ2台の表示・録画テスト中（アームには接続していません）")
                elif phase == "connecting" and mode == "running" and not context["stop"].is_set():
                    set_status("アームへ接続しています…")
                elif phase == "camera_closing" and mode in ("running", "stopping"):
                    set_status("追従終了。カメラのファイルを閉じています…")
            if mode in ("running", "stopping"):
                with context["latest_lock"]:
                    row = dict(context["latest"])
                for joint, _label, _unit in JOINT_LABELS:
                    for column, field in ((1, "leader_action"), (2, "sent_action")):
                        value = row.get(field, {}).get(f"{joint}.pos")
                        if value is not None:
                            values[joint, column].set(f"{value:.2f}")
                if row.get("actual_hz") is not None:
                    rate.set(f"追従速度：{row['actual_hz']:.1f} Hz　（設定：{args.fps:g} Hz）")
                for column, camera in enumerate(context["cameras"]):
                    frame = camera.latest_frame()
                    if frame is None:
                        continue
                    canvas = canvases[column]
                    im = Image.fromarray(frame[:, :, ::-1])
                    im.thumbnail((max(1, canvas.winfo_width()), max(1, canvas.winfo_height())))
                    images[column] = ImageTk.PhotoImage(im)
                    canvas.delete("waiting")
                    if canvas.find_withtag("video"):
                        canvas.itemconfigure("video", image=images[column])
                        canvas.coords("video", canvas.winfo_width() / 2, canvas.winfo_height() / 2)
                    else:
                        canvas.create_image(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                                            image=images[column], tags="video")
            if (context["finished"].is_set() and not context["save_requested"]
                    and not context["worker"].is_alive()):
                save_finished_session(context)
        window.after(33, refresh)

    start_session()
    window.after(33, refresh)
    try:
        window.mainloop()
    finally:
        if active is not None:
            active["stop"].set()
            for camera in active.get("cameras", []):
                camera.stop()
            worker = active.get("worker")
            if worker is not None and worker.ident is not None:
                worker.join()
            for camera in active.get("cameras", []):
                camera.join()
            if "session" in active:
                active["session"].close_log()
    return app_exit_code

