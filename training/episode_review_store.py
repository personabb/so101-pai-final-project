"""Local episode review/deletion requests; heavy Dataset editing runs in a child.

PNG image datasets only. Official v0.6.1 metadata/statistics rebuilding is reused;
data filtering uses bounded Arrow batches rather than whole-image pandas frames.
Unchanged Parquet files are hardlinked and never modified. Backups are retained.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def _inside(path, workspace):
    root = Path(workspace).resolve(strict=True)
    target = Path(path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Path is outside the workspace or is its root: {target}")
    return target


def _read(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _review_selection(previous, report, *, ready, reason, pending, removed_attempts=(), episode_mapping=None, attempts=()):
    """Preserve unresolved exclusions across requests, cutover and completion."""
    result=dict(previous)
    removed=set(removed_attempts)
    was_required=bool(previous.get("requires_physical_review"))
    previous_excluded=set(previous.get("excluded_attempts",[])) if was_required else set()
    failed={row["attempt"] for row in attempts if row.get("episode_index") is not None
            and (row.get("status")=="failed" or row.get("training_allowed") is False)}
    unresolved=(previous_excluded | failed)-removed
    unknown_exclusion=was_required and not previous_excluded
    requires=bool(unresolved or unknown_exclusion)
    error=str(report.get("error") or "")
    active_error=error.startswith(("Recording:","Storage:","Control:"))
    result.update(ready=bool(ready and not requires and not active_error),reason=(previous.get("reason") or reason)
                  if requires else error if active_error else reason,pending_deletions=list(pending),
                  requires_physical_review=requires,unresolved_recording_error=error if active_error else None)
    if was_required or unresolved:
        result["excluded_attempts"]=sorted(unresolved)
    if episode_mapping is not None and "excluded_episode_indices" in previous:
        result["excluded_episode_indices"]=[episode_mapping[index] for index in previous["excluded_episode_indices"] if index in episode_mapping]
    result["removed_attempts"]=sorted(set(previous.get("removed_attempts",[])) | removed)
    return result


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return False
        try:
            code=ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


class _EditLock:
    def __init__(self, collection, name=".episode-review-edit.lock"):
        self.path=collection / name
    def __enter__(self):
        try:
            fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            try:
                owner=_read(self.path,{"pid":None})
            except (OSError,ValueError):
                raise RuntimeError("Episode deletion is already running for this collection") from None
            if owner.get("pid") and not _pid_alive(owner["pid"]):
                self.path.unlink()
                fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            else:
                raise RuntimeError("Episode deletion is already running for this collection")
        with os.fdopen(fd,"w",encoding="utf-8") as file:
            json.dump({"pid":os.getpid(),"started_at":datetime.now().astimezone().isoformat()},file)
        return self
    def __exit__(self,*_):
        self.path.unlink(missing_ok=True)


def _state(collection, workspace):
    state = _read(collection / "review-state.json")
    if state is not None:
        return state
    # Historical collections have no review-state; find their completed Dataset
    # through its own provenance link, without loading any image data.
    for dataset in (Path(workspace) / "datasets").glob("demonstrations-*"):
        report = _read(dataset / "collection-report.json")
        if not report or not report.get("video_path"):
            continue
        if Path(report["video_path"]).resolve() != collection:
            continue
        info = _read(dataset / "meta/info.json", {})
        episodes = report.get("episodes_recorded", [])
        finalized = (len(episodes) == info.get("total_episodes") and
                     sum(row["frames"] for row in episodes) == info.get("total_frames"))
        result={"dataset_path": str(dataset), "repo_id": "local/so101-demonstrations",
                "fps": report.get("fps", 30), "finalized": finalized,
                "historical_finalization_check": "Committed episode count/frame count equals completed collection report; caller must not edit a live collection",
                "attempts": [{"attempt": int(Path(row["video_directory"]).name.split("_")[-1]),
                              "directory": row["video_directory"], "episode_index": row["episode"],
                              "frames": row["frames"], "status": "saved"} for row in episodes]}
        existing={Path(row["directory"]).resolve() for row in result["attempts"]}
        for directory in collection.glob("episode_*"):
            status=_read(directory / "episode_status.json")
            if status and directory.resolve() not in existing:
                result["attempts"].append({"attempt":int(directory.name.split("_")[-1]),"directory":str(directory),
                    "episode_index":None,"frames":status.get("frames",0),"status":"discarded"})
        return result
    return {"dataset_path": None, "repo_id": None, "fps": 30, "finalized": False, "attempts": []}


def list_attempts(workspace_root, active_controller=None):
    workspace = Path(workspace_root).resolve(strict=True)
    rows = []
    active_directory=None
    if active_controller is not None:
        with getattr(active_controller,"lock",nullcontext()):
            clip=getattr(active_controller,"clip",None)
            if clip is not None and getattr(active_controller,"state",None) in ("recording","paused","pausing"):
                active_directory=Path(clip.directory).resolve()
    for collection in sorted((workspace / "output/record-so101").glob("*")):
        if not collection.is_dir() or collection.name.startswith("."):
            continue
        try:
            state = _state(collection.resolve(), workspace)
            known = {Path(row["directory"]).resolve(): dict(row) for row in state.get("attempts", [])}
            requests = _read(collection / "deletion-requests.json", {"requests": []})["requests"]
            requested = {Path(row["attempt_dir"]).resolve(): row for row in requests}
            for directory in sorted(collection.glob("episode_*")):
                if not directory.is_dir():
                    continue
                directory = _inside(directory, workspace)
                status = _read(directory / "episode_status.json", {})
                row = known.get(directory, {"attempt": int(directory.name.split("_")[-1]), "directory": str(directory),
                    "episode_index": None, "frames": status.get("frames", 0), "status": "discarded" if status else "pending"})
                request = requested.get(directory)
                row.update(collection_directory=str(collection), dataset_path=state.get("dataset_path"),
                    finalized=state.get("finalized", False), camera_paths={name: str(directory / f"camera_{i}.avi")
                    for i,name in enumerate(("camera0", "camera1"))}, raw_manifest=str(directory / "pending_dataset/frames.jsonl"),
                    delete_requested=bool(request and not request.get("applied")))
                if row["delete_requested"]:
                    row["status"] = "delete_requested"
                if directory == active_directory:
                    row["status"]="active"
                row["training_allowed"] = (False if row.get("training_allowed") is False or row["status"] in ("discarded","deleted","delete_requested","failed")
                                           else True if row["status"]=="saved" else None)
                if state.get("dataset_path"):
                    selection=_read(Path(state["dataset_path"]) / "training-selection.json",{})
                    if selection.get("requires_physical_review") and row["attempt"] in selection.get("excluded_attempts",[]):
                        row["training_allowed"]=False
                        row["exclusion_reason"]=selection.get("reason","Unresolved recording error")
                if state.get("dataset_path") and Path(state["dataset_path"]).name=="demonstrations-20261012-142949-943345":
                    row["training_allowed"]=False
                    row["exclusion_reason"]="User excluded initial733-frame camera-quality test from learning"
                rows.append(row)
            for directory,row in known.items():
                if row.get("status") == "deleted" and not directory.exists():
                    row.update(collection_directory=str(collection),dataset_path=state.get("dataset_path"),
                               finalized=state.get("finalized",False),delete_requested=False,camera_paths={},raw_manifest=None)
                    rows.append(row)
        except (OSError, ValueError, KeyError):
            continue  # A changing live atomic state may be retried on the next GUI refresh.
    return rows


def _request_deletion(collection_dir, attempt_dir, workspace_root, reason=""):
    collection = _inside(collection_dir, workspace_root)
    directory = _inside(attempt_dir, workspace_root)
    if directory.parent != collection or not directory.name.startswith("episode_"):
        raise ValueError("Select one recording attempt inside this collection")
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    state = _state(collection, workspace_root)
    attempt = next((row for row in state.get("attempts", []) if Path(row["directory"]).resolve() == directory), None)
    if attempt and attempt.get("status") in ("active", "recording"):
        raise ValueError("Stop recording this attempt before requesting deletion")
    path = collection / "deletion-requests.json"
    requests = _read(path, {"requests": []})
    existing = next((row for row in requests["requests"] if Path(row["attempt_dir"]).resolve() == directory), None)
    if existing is None:
        existing = {"request_id": uuid.uuid4().hex, "attempt_dir": str(directory), "reason": str(reason),
                    "requested_at": datetime.now().astimezone().isoformat(), "applied": False}
        requests["requests"].append(existing)
        _write(path, requests)
    if state.get("dataset_path"):
        dataset=_inside(state["dataset_path"],workspace_root)
        if dataset.is_dir():
            marker_path=dataset / "training-selection.json"
            marker=_review_selection(_read(marker_path,{}),_read(dataset / "collection-report.json",{}),
                ready=False,reason="User-requested deletion pending; not for training",
                pending=[row["attempt_dir"] for row in requests["requests"] if not row.get("applied")],attempts=state.get("attempts",[]))
            _write(marker_path,marker)
    return {"accepted": True, "queued": not existing["applied"], **existing}


def request_deletion(collection_dir,attempt_dir,workspace_root,reason=""):
    collection=_inside(collection_dir,workspace_root)
    with _EditLock(collection,".episode-review-requests.lock"):
        return _request_deletion(collection,attempt_dir,workspace_root,reason)


def _rewrite_dataset(source_path, destination, keep, repo_id):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _copy_and_reindex_episodes_metadata
    from lerobot.datasets.io_utils import load_episodes

    source_meta = LeRobotDatasetMetadata(repo_id, root=source_path)
    if source_meta.video_keys:
        raise ValueError("This bounded editor supports the project's PNG image datasets only")
    source_meta.episodes = load_episodes(source_path)
    source = type("MetadataSource", (), {"meta": source_meta, "root": source_path})()
    target = LeRobotDatasetMetadata.create(repo_id, fps=source_meta.fps, features=source_meta.features,
        robot_type=source_meta.robot_type, root=destination, use_videos=False)
    mapping = {old:new for new,old in enumerate(keep)}
    tasks = list(source_meta.tasks.index)
    target.save_episode_tasks(tasks)
    task_mapping = {i: target.get_task_index(task) for i,task in enumerate(tasks)}
    files = {}
    for old in keep:
        files.setdefault(source_meta.get_data_file_path(old), []).append(old)
    data_metadata, global_index = {}, 0
    for relative, old_episodes in sorted(files.items()):
        source_file, target_file = source_path / relative, destination / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        parquet = pq.ParquetFile(source_file)
        scalar = parquet.read(columns=["episode_index", "index", "task_index"])
        old_values = scalar["episode_index"].to_pylist()
        preserve = (set(old_values) == set(old_episodes) and all(mapping[e] == e for e in old_values)
                    and scalar["index"].to_pylist() == list(range(global_index, global_index + len(old_values)))
                    and all(task_mapping[t] == t for t in scalar["task_index"].to_pylist()))
        if preserve:
            os.link(source_file, target_file)
            counts = {old: old_values.count(old) for old in old_episodes}
        else:
            counts = {old: 0 for old in old_episodes}
            writer = None
            try:
                for batch in parquet.iter_batches(batch_size=16):
                    table = pa.Table.from_batches([batch])
                    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(keep, type=table["episode_index"].type)))
                    if not table.num_rows:
                        continue
                    ep = table["episode_index"].to_pylist()
                    for old in ep:
                        counts[old] += 1
                    for key,values in (("episode_index", [mapping[e] for e in ep]),
                                       ("index", list(range(global_index + sum(counts.values()) - len(ep), global_index + sum(counts.values())))),
                                       ("task_index", [task_mapping[t] for t in table["task_index"].to_pylist()])):
                        i = table.schema.get_field_index(key)
                        table = table.set_column(i, table.schema.field(i), pa.array(values,type=table[key].type))
                    if writer is None:
                        writer = pq.ParquetWriter(target_file, table.schema, compression="snappy")
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
        for old in sorted(old_episodes):
            row = source_meta.episodes[old]
            count = counts[old]
            if count != row["length"]:
                raise ValueError("Source episode length differs from its Parquet rows")
            data_metadata[mapping[old]] = {"data/chunk_index": row["data/chunk_index"],
                "data/file_index": row["data/file_index"], "dataset_from_index": global_index,
                "dataset_to_index": global_index+count}
            global_index += count
    _copy_and_reindex_episodes_metadata(source, target, mapping, data_metadata)
    # Official metadata reload plus bounded scalar/state/action validation.
    checked = LeRobotDatasetMetadata(repo_id, root=destination)
    if checked.total_frames != global_index or checked.total_episodes != len(keep):
        raise ValueError("Rebuilt official Dataset metadata disagrees with row counts")
    index, expected_episode, frames_in_episode = 0, 0, 0
    episodes = load_episodes(destination)
    for relative in sorted(files):
        for batch in pq.ParquetFile(destination / relative).iter_batches(batch_size=2048,
                columns=["index", "episode_index", "frame_index", "observation.state", "action"]):
            rows = batch.to_pydict()
            for i in range(batch.num_rows):
                ep = rows["episode_index"][i]
                if ep != expected_episode:
                    if ep != expected_episode+1 or frames_in_episode != episodes[expected_episode]["length"]:
                        raise ValueError("Invalid rebuilt episode boundary")
                    expected_episode, frames_in_episode = ep, 0
                if rows["index"][i] != index or rows["frame_index"][i] != frames_in_episode:
                    raise ValueError("Invalid rebuilt frame/global index")
                if any(np.asarray(rows[k][i]).shape != (6,) or not np.isfinite(rows[k][i]).all()
                       for k in ("observation.state", "action")):
                    raise ValueError("Invalid rebuilt state/action values")
                index += 1
                frames_in_episode += 1
    if index != global_index or expected_episode != len(keep)-1 or frames_in_episode != episodes[expected_episode]["length"]:
        raise ValueError("Final rebuilt boundaries disagree with metadata")
    return {"episode_mapping": mapping, "total_frames": global_index, "total_episodes": len(keep),
            "official_stats_rebuilt": True, "bounded_image_batch_size": 16,
            "unchanged_parquet_hardlinks": True, "scalar_state_action_rows_verified": index}


def _apply_requested_deletions(collection_dir, workspace_root):
    workspace = Path(workspace_root).resolve(strict=True)
    collection = _inside(collection_dir, workspace)
    state = _state(collection, workspace)
    if state.get("finalized") is not True:
        raise ValueError("Collection must finish/finalize before Dataset deletion")
    requests_path = collection / "deletion-requests.json"
    document = _read(requests_path, {"requests": []})
    requests = [row for row in document["requests"] if not row.get("applied")]
    if not requests:
        return {"applied": False, "reason": "No pending requests"}
    dataset = _inside(state["dataset_path"], workspace)
    attempt_map = {Path(row["directory"]).resolve(): row for row in state["attempts"]}
    selected = []
    for request in requests:
        path = _inside(request["attempt_dir"], workspace)
        if path.parent != collection or path not in attempt_map:
            raise ValueError("Deletion request is not a known recording attempt")
        row = attempt_map[path]
        if row.get("status") in ("active", "recording", "pending", "exporting"):
            raise ValueError("Deletion request still refers to an unfinished attempt")
        selected.append(row)
    info = _read(dataset / "meta/info.json")
    source_selection=_read(dataset / "training-selection.json",{})
    deleted_indices = sorted({row["episode_index"] for row in selected if row.get("episode_index") is not None})
    if any(type(i) is not int or not 0 <= i < info["total_episodes"] for i in deleted_indices):
        raise ValueError("Deletion indices are outside the completed Dataset")
    keep = [i for i in range(info["total_episodes"]) if i not in deleted_indices]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive = _inside(workspace / "tools/archive/episode-review" / stamp, workspace)
    archive.mkdir(parents=True, exist_ok=False)
    stage = _inside(dataset.with_name(dataset.name + ".review-" + stamp), workspace)
    if deleted_indices and keep:
        summary = _rewrite_dataset(dataset, stage, keep, state.get("repo_id") or "local/so101-demonstrations")
    else:
        summary={"episode_mapping":{old:new for new,old in enumerate(keep)},
                 "total_episodes":len(keep),"total_frames":info["total_frames"] if not deleted_indices else 0,
                 "dataset_deleted":bool(deleted_indices and not keep)}
    mapping = summary["episode_mapping"]
    report = _read(dataset / "collection-report.json", {})
    source_report = dict(report)
    report["episodes_recorded"] = [{**row, "original_episode_index": row.get("original_episode_index",row["episode"]),
        "episode": mapping[row["episode"]]} for row in report.get("episodes_recorded",[]) if row["episode"] in mapping]
    report["dataset_path"] = str(dataset)
    report["episode_review"] = {"deleted_indices": deleted_indices, "archive": str(archive),
                              "source_collection_error": source_report.get("error"),
                              "source_collection_exit_code": source_report.get("exit_code"),
                              "task_success_verified": False,
                              "acquisition_error_resolved":bool(deleted_indices and info["total_episodes"]-1 in deleted_indices
                                  and str(source_report.get("error") or "").startswith(("Recording:","Storage:","Control:")))}
    # Preserve acquisition error provenance: removal does not prove all surviving demonstrations good.
    if deleted_indices and keep:
        if report["episode_review"]["acquisition_error_resolved"]:
            report["source_collection_error"]=report.pop("error",None)
            report["exit_code"]=0
        _write(stage / "collection-report.json", report)
        marker=_review_selection(source_selection,report,ready=False,reason="Episode deletion transaction in progress",
            pending=[row["attempt_dir"] for row in requests],removed_attempts=[row["attempt"] for row in selected],
            episode_mapping=mapping,attempts=state.get("attempts",[]))
        marker["excluded_original_indices"]=deleted_indices
        _write(stage / "training-selection.json",marker)
    selected_attempts={row["attempt"] for row in selected}
    if deleted_indices and keep and (dataset / "acquisition.jsonl").exists():
        with (dataset / "acquisition.jsonl").open(encoding="utf-8-sig") as src, \
             (stage / "acquisition.jsonl").open("w",encoding="utf-8") as out:
            for line in src:
                row = json.loads(line)
                if row.get("recording_attempt") in selected_attempts:
                    continue
                if row["episode"] in mapping:
                    row["original_episode_index"] = row.get("original_episode_index",row["episode"])
                    row["episode"] = mapping[row["episode"]]
                    out.write(json.dumps(row,ensure_ascii=False)+"\n")
    backup=archive / "dataset"
    original_files={path:path.read_bytes() if path.exists() else None for path in
                    (collection / "review-state.json",collection / "collection-report.json")}
    moved=[]
    try:
        # An open AVI must fail before Dataset cutover. Restore any earlier moves.
        for row in selected:
            directory=_inside(row["directory"],workspace)
            if directory.exists():
                target=archive / directory.name
                os.rename(directory,target)
                moved.append((directory,target))
        if deleted_indices:
            os.rename(dataset,backup)
            if keep:
                os.rename(stage,dataset)
            else:
                state["dataset_path"]=None
                state["dataset_deleted"]=True
        for row in selected:
            row.update(status="deleted",episode_index=None,archive_directory=str(archive / Path(row["directory"]).name))
        for row in state["attempts"]:
            if row.get("episode_index") in mapping:
                old=row["episode_index"]
                row["original_episode_index"]=row.get("original_episode_index",old)
                row["episode_index"]=mapping[old]
        state["last_deletion"]={"archive":str(archive),**summary}
        result={"applied":True,"dataset_path":state["dataset_path"],"archive":str(archive),
            "deleted_indices":deleted_indices,"deleted_attempts":[row["attempt"] for row in selected],**summary}
        _write(collection / "review-state.json",state)
        _write(collection / "collection-report.json",report)
        _write(archive / "deletion-result.json",result)
        with _EditLock(collection,".episode-review-requests.lock"):
            current=_read(requests_path,{"requests":[]})
            applied_ids={row["request_id"] for row in requests}
            for request in current["requests"]:
                if request["request_id"] in applied_ids:
                    request.update(applied=True,applied_at=datetime.now().astimezone().isoformat(),archive=str(archive))
            _write(requests_path,current)
            if dataset.exists():
                pending=[row["attempt_dir"] for row in current["requests"] if not row.get("applied")]
                marker=_review_selection(_read(dataset / "training-selection.json",{}),report,ready=not pending,
                    reason="Additional recording deletions pending" if pending else "Requested recording exclusions applied; task performance not verified",
                    pending=pending,removed_attempts=[row["attempt"] for row in selected],attempts=state.get("attempts",[]))
                _write(dataset / "training-selection.json",marker)
        return result
    except BaseException:
        if backup.exists():
            if dataset.exists():
                os.rename(dataset,archive / "failed-new-dataset")
            os.rename(backup,dataset)
        for original,target in reversed(moved):
            if target.exists():
                os.rename(target,original)
        for path,value in original_files.items():
            if value is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(value)
        # A marker write can fail after request acknowledgement: undo only this
        # transaction's acknowledgements, retaining requests added concurrently.
        with _EditLock(collection,".episode-review-requests.lock"):
            original_requests={row["request_id"]:row for row in requests}
            current=_read(requests_path,{"requests":[]})
            current["requests"]=[original_requests.get(row["request_id"],row) for row in current["requests"]]
            _write(requests_path,current)
        raise


def apply_requested_deletions(collection_dir,workspace_root):
    collection=_inside(collection_dir,workspace_root)
    with _EditLock(collection):
        return _apply_requested_deletions(collection,workspace_root)


def apply_requested_deletions_subprocess(collection_dir,workspace_root):
    collection=_inside(collection_dir,workspace_root)
    requests=_read(collection / "deletion-requests.json",{"requests":[]})
    if not any(not row.get("applied") for row in requests["requests"]):
        return {"applied":False,"reason":"No pending requests"}
    if not _state(collection,workspace_root).get("finalized"):
        return {"applied":False,"queued":True,"reason":"Waiting for collection finalization"}
    log=collection / ("episode-deletion-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".log")
    result_path=log.with_suffix(".json")
    with log.open("w",encoding="utf-8") as output:
        process=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),
            "--workspace-root",str(Path(workspace_root).resolve()),"--collection-dir",str(collection),
            "--apply","--result-path",str(result_path)],stdout=output,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if process.returncode:
        if "Episode deletion is already running" in log.read_text(encoding="utf-8",errors="replace"):
            return {"applied":False,"queued":True,"reason":"Another episode deletion is already running"}
        raise RuntimeError(f"Episode deletion failed (exit={process.returncode}); originals retained where possible; log: {log}")
    return _read(result_path)


def assert_training_eligible(dataset_root):
    """Compatibility forwarding to the shared lightweight training guard."""
    from training_selection import assert_training_eligible as guard
    return guard(dataset_root)


def complete_collection_review(collection_dir,workspace_root):
    """Publish eligibility under the same short lock as new requests."""
    collection=_inside(collection_dir,workspace_root)
    with _EditLock(collection,".episode-review-requests.lock"):
        state=_state(collection,workspace_root)
        requests=_read(collection / "deletion-requests.json",{"requests":[]})["requests"]
        queued=any(not row.get("applied") for row in requests)
        dataset_path=state.get("dataset_path")
        ready=bool(state.get("finalized") and not queued and dataset_path)
        if dataset_path:
            dataset=_inside(dataset_path,workspace_root)
            ready=ready and dataset.is_dir()
            if dataset.is_dir():
                previous=_read(dataset / "training-selection.json",{})
                marker=_review_selection(previous,_read(dataset / "collection-report.json",{}),ready=ready,
                    reason="Collection finalized; requested exclusions applied" if ready else "Pending recording deletions" if queued else "Collection not finalized",
                    pending=[row["attempt_dir"] for row in requests if not row.get("applied")],attempts=state.get("attempts",[]))
                _write(dataset / "training-selection.json",marker)
                ready=marker["ready"]
        return {"ready":ready,"queued":queued}


def watch_requested_deletions(collection_dir,workspace_root):
    """Wait without devices/torch until closed collection is safe to edit."""
    collection=_inside(collection_dir,workspace_root)
    print(json.dumps({"phase":"waiting_for_finalization","pid":os.getpid(),"collection":str(collection)},ensure_ascii=False),flush=True)
    last_result=None
    while True:
        document=_read(collection / "deletion-requests.json",{"requests":[]})
        if not any(not row.get("applied") for row in document["requests"]):
            return last_result or {"applied":False,"completed":True,"reason":"Requests already processed or none pending"}
        if _state(collection,workspace_root).get("finalized"):
            try:
                last_result=apply_requested_deletions(collection,workspace_root)
                continue  # Recheck requests added during this apply before exiting.
            except RuntimeError as error:
                if "already running" not in str(error):
                    raise
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root",type=Path,required=True)
    parser.add_argument("--collection-dir",type=Path,required=True)
    parser.add_argument("--apply",action="store_true")
    parser.add_argument("--watch",action="store_true",help="Wait for finalization then apply pending requests; no devices")
    parser.add_argument("--request-attempt",type=Path)
    parser.add_argument("--reason",default="")
    parser.add_argument("--result-path",type=Path)
    args = parser.parse_args()
    if args.request_attempt:
        result=request_deletion(args.collection_dir,args.request_attempt,args.workspace_root,args.reason)
    elif args.watch:
        result=watch_requested_deletions(args.collection_dir,args.workspace_root)
    elif args.apply:
        result=apply_requested_deletions(args.collection_dir,args.workspace_root)
    else:
        result=list_attempts(args.workspace_root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if args.result_path:
        _write(_inside(args.result_path,args.workspace_root),result)


if __name__ == "__main__":
    main()

