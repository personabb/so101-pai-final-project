"""Run one locked finite ACT camera augmentation suite with bounded parallelism."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from camera_job import atomic_json
from camera_progress import log_progress


def now() -> str:
    return datetime.now().astimezone().isoformat()


def run_suite(args) -> int:
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan or root / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    profiles = plan['profiles']
    steps = int(plan.get('steps', 100000))
    concurrency = int(plan.get('concurrency', 1))
    seed = int(plan.get('seed', 1000))
    if (not isinstance(profiles, list) or not profiles or len(set(profiles)) != len(profiles)
            or any(not isinstance(name, str) or not name.replace('_', '').isalnum() or name == 'baseline' for name in profiles)
            or steps < 1 or concurrency < 1):
        raise ValueError('Invalid profiles, steps or concurrency in plan.json')
    status_path = root / 'status.json'
    if status_path.exists():
        raise FileExistsError('Suite already has status.json; refusing another launch')
    # This start marker remains on disk even after completion or interruption.
    # No runner silently recovers/relaunches previously started jobs.
    fd = os.open(root / 'suite.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as lock:
        json.dump({'pid': os.getpid(), 'started_at': now(), 'plan': str(plan_path)}, lock)
        lock.flush()
        os.fsync(lock.fileno())
    names = list(profiles) if args.skip_baseline else ['baseline', *profiles]
    if any((root / name).exists() for name in names):
        raise FileExistsError('A job directory already exists; refusing to overwrite or relaunch')
    status = {'pid': os.getpid(), 'state': 'running', 'started_at': now(), 'root': str(root),
              'plan': plan, 'steps_per_profile': steps, 'concurrency': concurrency,
              'source_checkpoint': str(args.source), 'baseline': 'original 200k checkpoint; evaluation only',
              'robot_access': False, 'jobs': {name: {'state': 'queued'} for name in names}}
    atomic_json(status_path, status)
    scripts = Path(__file__).resolve().parent
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    pending = names.copy()
    active: dict[str, tuple[subprocess.Popen, object]] = {}
    baseline_done = args.skip_baseline
    try:
        while pending or active:
            # Complete baseline first, then schedule training+evaluation as one
            # slot per profile so evaluation cannot overcommit the selected limit.
            slots = concurrency if baseline_done else 1
            while pending and len(active) < slots:
                if not baseline_done and active:
                    break
                name = pending.pop(0)
                command = [sys.executable, str(scripts / 'camera_job.py'), '--profile', name,
                           '--root', str(root), '--base', str(args.base), '--source', str(args.source),
                           '--dataset', str(args.dataset), '--steps', str(steps), '--seed', str(seed)]
                log = (root / f'{name}-job.log').open('x', encoding='utf-8')
                process = subprocess.Popen(command, cwd=args.base, env=environment, stdout=log, stderr=subprocess.STDOUT)
                active[name] = (process, log)
                status['jobs'][name] = {'state': 'running', 'pid': process.pid, 'started_at': now(), 'command': command}
                atomic_json(status_path, status)
                if not baseline_done:
                    break
            for name, (process, log) in list(active.items()):
                report_path = root / name / 'job-result.json'
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    status['jobs'][name]['phase'] = report.get('state')
                    status['jobs'][name]['checkpoint_saved'] = report.get('checkpoint_saved')
                status['jobs'][name]['progress'] = log_progress(root / name / 'training.log', steps)
                code = process.poll()
                if code is not None:
                    log.close()
                    status['jobs'][name].update(state='completed' if code == 0 else 'failed',
                                                exit_code=code, completed_at=now())
                    del active[name]
                    if name == 'baseline':
                        baseline_done = True
            status['updated_at'] = now()
            atomic_json(status_path, status)
            print(json.dumps({'updated_at': status['updated_at'], 'running': list(active), 'queued': pending,
                              'finished': [name for name, item in status['jobs'].items() if 'exit_code' in item]}), flush=True)
            if pending or active:
                time.sleep(10)
        code = 0 if all(item.get('exit_code') == 0 for item in status['jobs'].values()) else 1
        status.update(state='completed' if code == 0 else 'completed_with_failures', exit_code=code, completed_at=now())
        atomic_json(status_path, status)
        return code
    except BaseException as error:
        # Leave existing jobs untouched; record their PIDs for explicit recovery.
        status.update(state='interrupted', error=repr(error), interrupted_at=now(),
                      active_job_pids={name: process.pid for name, (process, _) in active.items()})
        atomic_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--base', type=Path, default=Path.cwd())
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--skip-baseline', action='store_true', help='Reuse the completed original baseline evaluation')
    args = parser.parse_args()
    for name in ("root", "base", "source", "dataset"):
        setattr(args, name, getattr(args, name).resolve())
    if args.plan:
        args.plan = args.plan.resolve()
    raise SystemExit(run_suite(args))


if __name__ == '__main__':
    main()
