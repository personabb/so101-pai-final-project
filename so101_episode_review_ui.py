"""Review saved AVI or sealed RGB attempts; no camera/robot connections."""
from __future__ import annotations

import importlib
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
STATUS_LABELS = {"active":"録画中", "recording":"録画中", "pending":"保存待ち",
    "exporting":"保存中", "saved":"保存済み", "discarded":"学習対象外",
    "delete_requested":"削除待ち／学習不可（反映待ち）", "deleted":"削除済み", "error":"学習対象外",
    "failed":"保存失敗・学習対象外"}


class FramePairReader:
    """File-only reader. RGB spool pairs are camera0 then camera1, 640x480."""
    def __init__(self, attempt):
        import cv2
        self.cv2 = cv2
        self.attempt = dict(attempt)
        self.directory = Path(attempt["directory"])
        self.fps = float(attempt.get("fps",30))
        if not 0 < self.fps <= 1000:
            raise ValueError("録画のfpsが不正です。")
        manifest = attempt.get("raw_manifest")
        self.raw_path = Path(manifest).parent / "frames.rgb" if manifest else self.directory / "pending_dataset/frames.rgb"
        paths = attempt.get("camera_paths") or {}
        self.paths = [Path(paths.get(f"camera{i}") or self.directory / f"camera_{i}.avi") for i in (0,1)]
        self.caps = []
        self.use_raw = False
        try:
            # A partial AVI being rendered is not ready; sealed raw is complete.
            if self.raw_path.is_file():
                self.use_raw = True
                self.frames = self.raw_path.stat().st_size // 1843200
                self.source = "保存待ち RGB 原本"
            elif all(path.is_file() for path in self.paths):
                for path in self.paths:
                    cap = cv2.VideoCapture(str(path))
                    self.caps.append(cap)
                    if not cap.isOpened():
                        raise RuntimeError(f"動画を開けません: {path.name}")
                self.frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.caps)
                fps = self.caps[0].get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.fps = fps
                self.source = "保存済み AVI"
            else:
                raise FileNotFoundError("再生できる動画または RGB 原本がまだありません。")
            declared = int(attempt.get("frames") or self.frames)
            self.frames = min(self.frames,declared)
            if self.frames <= 0:
                raise ValueError("この試行には再生フレームがありません。")
        except BaseException:
            self.close()
            raise

    def read_pair(self,index):
        import numpy as np
        index = int(index)
        if not 0 <= index < self.frames:
            raise IndexError(index)
        if self.use_raw:
            # Do not keep a Windows file handle that would block export cleanup.
            try:
                with self.raw_path.open("rb") as raw:
                    raw.seek(index*1843200)
                    data = raw.read(1843200)
            except FileNotFoundError:
                if all(path.is_file() for path in self.paths):
                    self.close()
                    self.__init__(self.attempt)
                    return self.read_pair(index)
                raise
            if len(data) != 1843200:
                raise RuntimeError("RGB 原本の読み取りが途中で終わりました。")
            pair = np.frombuffer(data,np.uint8).reshape(2,480,640,3)
            return pair[0],pair[1]
        result = []
        for cap in self.caps:
            position = int(cap.get(self.cv2.CAP_PROP_POS_FRAMES))
            if position != index:
                cap.set(self.cv2.CAP_PROP_POS_FRAMES,index)
            ok,image = cap.read()
            if not ok:
                raise RuntimeError(f"動画のフレーム {index} を読み取れません。")
            result.append(self.cv2.cvtColor(image,self.cv2.COLOR_BGR2RGB))
        return tuple(result)

    def close(self):
        for cap in self.caps:
            cap.release()
        self.caps = []


class _PreviewWorker:
    """One decoder owner; seeking replaces pending requests, never adds a queue."""
    def __init__(self,attempt,messages,generation):
        self.attempt,self.messages,self.generation = attempt,messages,generation
        self.condition = threading.Condition()
        self.requested = None
        self.stopped = False
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def request(self,index):
        with self.condition:
            self.requested = index
            self.condition.notify()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify()

    def run(self):
        reader = None
        try:
            reader = FramePairReader(self.attempt)
            self.messages.put(("reader_ready",self.generation,(reader.frames,reader.fps,reader.source)))
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopped or self.requested is not None)
                    if self.stopped:
                        break
                    index,self.requested = self.requested,None
                pair = reader.read_pair(index)
                self.messages.put(("frame",self.generation,(index,pair)))
        except Exception as exc:
            self.messages.put(("preview_error",self.generation,str(exc)))
        finally:
            if reader is not None:
                reader.close()


def show_episode_review(parent,root_path=ROOT,active_controller=None,*,store=None):
    """Open a Toplevel, returning it immediately. All deletion work is asynchronous."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image,ImageTk
    root_path = Path(root_path).resolve()
    if store is None:
        sys.path.insert(0,str(root_path/"training"))
        store = importlib.import_module("episode_review_store")
    window = tk.Toplevel(parent)
    window.title("SO-101 録画の確認と削除")
    window.geometry("1120x850")
    messages = queue.Queue()
    state = {"rows":{},"selected":None,"worker":None,"generation":0,"index":0,"frames":0,
        "fps":30.,"playing":False,"next_tick":0.,"closing":False,"busy":False,"updating_seek":False}
    applied_signatures = set()
    busy_collections = set()
    photos = []
    status = tk.StringVar(value="試行番号はフォルダー番号です。学習用 episode 番号と区別して表示します。")
    ttk.Label(window,textvariable=status,wraplength=1080).pack(padx=8,pady=6)
    tree = ttk.Treeview(window,columns=("collection","attempt","dataset","frames","status"),show="headings",height=8)
    for key,label,width in (("collection","収集",270),("attempt","試行",65),("dataset","学習 episode（0開始）",170),("frames","フレーム",95),("status","状態",180)):
        tree.heading(key,text=label); tree.column(key,width=width)
    tree.pack(fill="x",padx=8)
    source_text = tk.StringVar(value="保存待ちは削除予約できます。現在の録画は収集画面から破棄できます。")
    ttk.Label(window,textvariable=source_text,wraplength=1080).pack(pady=6)
    panels = ttk.Frame(window); panels.pack()
    image_labels = []
    for name in ("camera0：全体","camera1：手先"):
        frame = ttk.Frame(panels); frame.pack(side="left",padx=4)
        ttk.Label(frame,text=name).pack()
        label = ttk.Label(frame,text="未選択",width=66); label.pack()
        image_labels.append(label)
    timeline = tk.StringVar(value="0 / 0")
    ttk.Label(window,textvariable=timeline).pack()
    seek = ttk.Scale(window,from_=0,to=1,orient="horizontal"); seek.pack(fill="x",padx=12)
    controls = ttk.Frame(window); controls.pack(pady=6)
    play = ttk.Button(controls,text="再生 / 一時停止 [Space]",state="disabled"); play.pack(side="left",padx=4)
    ttk.Button(controls,text="一覧を更新 [U]",command=lambda:refresh(force=True)).pack(side="left",padx=4)
    delete = ttk.Button(controls,text="この試行を削除… [D]",state="disabled"); delete.pack(side="left",padx=4)
    ttk.Button(controls,text="閉じる [Esc]",command=lambda:close()).pack(side="left",padx=4)
    confirmation = ttk.Frame(window)
    confirmation_text = tk.StringVar()
    ttk.Label(confirmation,textvariable=confirmation_text,wraplength=1080).pack()
    confirmation_buttons = ttk.Frame(confirmation); confirmation_buttons.pack()
    confirm = ttk.Button(confirmation_buttons,text="削除を予約する [C]"); confirm.pack(side="left",padx=4)
    ttk.Button(confirmation_buttons,text="キャンセル [X]",command=lambda:confirmation.pack_forget()).pack(side="left",padx=4)
    ttk.Label(window,text="保存中・保存待ちは削除予約し、全保存完了後に反映します。現在の録画を破棄しても追従とカメラ表示は続きます。",wraplength=1080).pack(pady=6)

    def current_discard_callback(item):
        live_path = getattr(active_controller,"video_path",None)
        callback = getattr(active_controller,"discard_recording",None)
        if callback and live_path and Path(live_path).resolve()==Path(item["collection_directory"]).resolve():
            return callback
        return None

    def is_active(item):
        return item.get("status") in ("active","recording","paused","pausing")

    def deletion_allowed(item):
        return (item is not None and not state["busy"] and not item.get("delete_requested")
                and item.get("status")!="deleted"
                and (not is_active(item) or current_discard_callback(item) is not None))

    def stop_preview():
        state["playing"] = False
        state["generation"] += 1
        previous = state["worker"]
        if previous is not None:
            previous.close()
        state["worker"] = None
        play.configure(state="disabled")
        return previous

    def choose(_event=None):
        selected = tree.selection()
        if not selected:
            return
        item = state["rows"].get(selected[0])
        if item is None:
            return
        stop_preview(); confirmation.pack_forget()
        state["selected"] = item
        source_text.set(str(item["directory"]))
        blocked = is_active(item) or item.get("status")=="deleted"
        delete.configure(state="normal" if deletion_allowed(item) else "disabled")
        for label in image_labels:
            label.configure(image="",text="現在の録画は再生対象外です。" if blocked else "読み込み中…")
        if blocked:
            status.set("この試行は削除済みです。" if item.get("status")=="deleted" else
                "Dで現在の録画を停止して破棄できます。学習用の保存は行いません。" if deletion_allowed(item) else
                "現在の録画は、この独立画面では破棄できません。収集画面の破棄操作を使ってください。")
            return
        state["worker"] = _PreviewWorker(item,messages,state["generation"])
        state["worker"].request(0)

    def seek_changed(value):
        if state["updating_seek"] or state["worker"] is None or not state["frames"]:
            return
        state["playing"] = False
        state["worker"].request(max(0,min(state["frames"]-1,int(float(value)))))

    def toggle_play():
        if state["worker"] is None:
            return
        state["playing"] = not state["playing"]
        state["next_tick"] = time.perf_counter()
        if state["playing"] and state["index"]>=state["frames"]-1:
            state["index"] = -1

    def deletion_apply(collection,previous_reader=None):
        try:
            if previous_reader is not None:
                previous_reader.thread.join(5)
                if previous_reader.thread.is_alive():
                    raise RuntimeError("動画を閉じる処理が終わっていません。削除予約は残します。")
            result = store.apply_requested_deletions_subprocess(Path(collection),root_path)
            messages.put(("apply_done",None,(collection,result)))
        except Exception as exc:
            messages.put(("operation_error",None,f"削除反映エラー: {exc}"))
        finally:
            busy_collections.discard(collection)

    def refresh(force=False):
        if state["closing"]:
            return
        try:
            rows = store.list_attempts(root_path,active_controller=active_controller)
            if force:
                applied_signatures.clear()
            request_directories = {}
            for row in rows:
                if row.get("delete_requested") and row.get("status")!="deleted":
                    request_directories.setdefault(str(row["collection_directory"]),[]).append(str(row["directory"]))
            selected = str(state["selected"]["directory"]) if state["selected"] else None
            state["rows"] = {str(row["directory"]):row for row in rows}
            for iid in tree.get_children():
                if iid not in state["rows"]:
                    tree.delete(iid)
            for iid,row in state["rows"].items():
                status_name = "delete_requested" if row.get("delete_requested") and row.get("status")!="deleted" else row.get("status","")
                status_label = STATUS_LABELS.get(status_name,status_name)
                if row.get("training_allowed") is False and status_name not in ("delete_requested","deleted"):
                    status_label += "／学習対象外" if "学習対象外" not in status_label else ""
                values = (Path(row["collection_directory"]).name,row["attempt"],row.get("episode_index") if row.get("episode_index") is not None else "—",row.get("frames",0),status_label)
                if tree.exists(iid): tree.item(iid,values=values)
                else: tree.insert("","end",iid=iid,values=values)
                collection = str(row["collection_directory"])
                finalized = bool(row.get("finalized"))
                signature = (collection,tuple(sorted(request_directories.get(collection,[]))))
                if finalized and row.get("delete_requested") and row.get("status")!="deleted" and signature not in applied_signatures and collection not in busy_collections:
                    applied_signatures.add(signature); busy_collections.add(collection)
                    previous = None
                    if state["selected"] and str(state["selected"]["collection_directory"])==collection:
                        previous = stop_preview()
                    threading.Thread(target=deletion_apply,args=(collection,previous),daemon=True).start()
            if selected in state["rows"]:
                state["selected"] = state["rows"][selected]
                if tree.selection() != (selected,):
                    tree.selection_set(selected)
                item = state["selected"]
                delete.configure(state="normal" if deletion_allowed(item) else "disabled")
                if item.get("status")=="deleted" and state["worker"]:
                    stop_preview(); status.set("削除が反映されました。")
        except Exception as exc:
            status.set(f"一覧を読み取れません: {exc}")

    def confirm_delete():
        item = state["selected"]
        if not deletion_allowed(item):
            return
        confirmation_text.set(f"収集 {Path(item['collection_directory']).name} の試行 {item['attempt']} を削除対象にします。先の録画は変更しません。よろしいですか？")
        if is_active(item):
            confirmation_text.set(f"現在の試行 {item['attempt']} を停止して破棄します。学習用の保存は行いません。診断用の原本は残します。よろしいですか？")
            confirm.configure(text="停止して破棄する [C]")
        else:
            confirm.configure(text="削除を予約する [C]")
        confirmation.pack(padx=8,pady=6)

    def request_delete():
        item = state["selected"]
        if not deletion_allowed(item) or not confirmation.winfo_manager():
            return
        state["busy"] = True; confirmation.pack_forget(); delete.configure(state="disabled")
        previous_reader = stop_preview()
        def operation():
            try:
                if previous_reader is not None:
                    previous_reader.thread.join(5)
                    if previous_reader.thread.is_alive():
                        raise RuntimeError("動画を閉じる処理が終わっていません。")
                live_path = getattr(active_controller,"video_path",None)
                callback = getattr(active_controller,"request_episode_deletion",None)
                if is_active(item):
                    discard = current_discard_callback(item)
                    accepted = bool(discard and discard(item["directory"]))
                    result = {"accepted":accepted,"discarded":accepted,"active_changed":not accepted}
                elif callback and live_path and Path(live_path).resolve()==Path(item["collection_directory"]).resolve():
                    result = callback(item["directory"])
                else:
                    result = store.request_deletion(Path(item["collection_directory"]),Path(item["directory"]),root_path,reason="User requested deletion from review UI")
                messages.put(("delete_done",None,result))
            except Exception as exc:
                messages.put(("operation_error",None,f"削除予約エラー: {exc}"))
        threading.Thread(target=operation,daemon=True).start()

    def close():
        state["closing"] = True
        stop_preview()
        for key in ("tick_after","refresh_after"):
            if state.get(key) is not None:
                window.after_cancel(state[key])
        window.destroy()

    def tick():
        if state["closing"]:
            return
        while True:
            try: kind,generation,payload = messages.get_nowait()
            except queue.Empty: break
            if generation is not None and generation!=state["generation"]:
                continue
            if kind=="reader_ready":
                state["frames"],state["fps"],source_kind = payload
                seek.configure(to=max(0,state["frames"]-1))
                play.configure(state="normal")
                status.set(f"{source_kind}：{state['frames']}フレーム / {state['fps']:g}fps。再生は録画ファイルのみです。")
            elif kind=="frame":
                index,pair = payload; state["index"] = index
                photos.clear()
                for label,rgb in zip(image_labels,pair):
                    photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((512,384)),master=window)
                    photos.append(photo); label.configure(image=photo,text="")
                state["updating_seek"] = True
                seek.set(index); state["updating_seek"] = False
                timeline.set(f"{index/state['fps']:.2f}秒 / {state['frames']/state['fps']:.2f}秒　フレーム {index+1}/{state['frames']}")
                state["next_tick"] = time.perf_counter()+1/state["fps"]
            elif kind=="preview_error":
                state["playing"] = False
                status.set(f"映像を再生できません: {payload}。停止済みの試行は、映像がなくても D で削除予約できます。")
            elif kind=="delete_done":
                state["busy"] = False
                if payload.get("discarded"):
                    status.set("現在の録画を停止して破棄しています。追従とカメラ表示は続きます。")
                elif payload.get("active_changed"):
                    status.set("対象の試行が変わったため破棄しませんでした。現在の試行を確認してください。")
                else:
                    status.set("削除を予約しました。保存完了後に反映します。" if payload.get("accepted",True) else f"削除できません: {payload}")
                refresh()
            elif kind=="apply_done":
                status.set(f"削除の反映が終了しました: {payload[1]}")
                refresh()
            elif kind=="operation_error":
                state["busy"] = False; status.set(payload); refresh()
        if state["playing"] and state["worker"] is not None and time.perf_counter()>=state["next_tick"]:
            if state["index"]+1>=state["frames"]:
                state["playing"] = False
            else:
                state["worker"].request(state["index"]+1)
                state["next_tick"] = float("inf")
        state["tick_after"] = window.after(20,tick)

    def periodic_refresh():
        if not state["closing"]:
            refresh(); state["refresh_after"] = window.after(2000,periodic_refresh)

    tree.bind("<<TreeviewSelect>>",choose)
    seek.configure(command=seek_changed)
    play.configure(command=toggle_play)
    delete.configure(command=confirm_delete)
    confirm.configure(command=request_delete)
    window.protocol("WM_DELETE_WINDOW",close)
    window.bind("<Escape>",lambda _event:close())
    held = set()
    def keyboard(event):
        key = event.keysym.lower()
        if key in held:
            return "break"
        held.add(key)
        if key=="space" and str(play.cget("state"))!="disabled": toggle_play()
        elif key=="u": refresh(force=True)
        elif key=="d" and str(delete.cget("state"))!="disabled": confirm_delete()
        elif key=="c" and confirmation.winfo_manager(): request_delete()
        elif key=="x" and confirmation.winfo_manager(): confirmation.pack_forget()
        else: return None
        return "break"
    window.bind("<KeyPress>",keyboard)
    window.bind("<KeyRelease>",lambda event:held.discard(event.keysym.lower()))
    window.bind("<FocusOut>",lambda _event:held.clear())
    window.review_state = state
    window.review_tree = tree
    window.review_refresh = refresh
    periodic_refresh(); tick()
    return window


if __name__=="__main__":
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    try:
        review = show_episode_review(root,ROOT)
        review.bind("<Destroy>",lambda event:root.destroy() if event.widget is review and root.winfo_exists() else None)
        root.mainloop()
    finally:
        try:
            if root.winfo_exists(): root.destroy()
        except tk.TclError:
            pass

