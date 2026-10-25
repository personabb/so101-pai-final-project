"""Read camera-suite status and recent training progress without changing any run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re



def tail_text(path: Path, limit: int = 262144) -> str:
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode('utf-8', errors='replace')


def log_progress(path: Path, target: int = 100000) -> dict:
    text = tail_text(path)
    # Configuration lines such as cfg.steps=100000 describe the target, not progress.
    lines = [line for line in text.splitlines() if re.search(r'(?<![.\w])step\s*:\s*[\d.]', line)]
    result = {'log': str(path), 'target_steps': target, 'step': None, 'eta_seconds': None}
    if not lines:
        bars = re.findall(r'Training:\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)', text)
        if bars and int(bars[-1][1]) == target:
            result['step'] = int(bars[-1][0])
            result['progress_source'] = 'official tqdm progress bar; update timing not logged yet'
        return result
    line = lines[-1]
    match = re.search(r'(?<![.\w])step\s*:\s*([\d.]+)\s*([KMG]?)', line, re.IGNORECASE)
    if match:
        result['step'] = round(float(match.group(1)) * {'': 1, 'K': 1000, 'M': 1000000, 'G': 1000000000}[match.group(2).upper()])
    result['last_progress_line'] = line
    timestamp = re.search(r'(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[.,]\d+)?)', line)
    result['last_progress_timestamp'] = timestamp.group(1) if timestamp else None
    # The official update_s/dataloading_s metrics are times per update. Do not
    # interpret throughput or the display's batch count as a measured duration.
    update = re.search(r'\b(?:updt_s|update_s)\s*[:=]\s*([\d.]+)', line)
    loading = re.search(r'\b(?:data_s|dataloading_s)\s*[:=]\s*([\d.]+)', line)
    if update and result['step'] is not None:
        duration = float(update.group(1)) + (float(loading.group(1)) if loading else 0)
        result['seconds_per_update'] = duration
        result['eta_seconds'] = max(0, target - result['step']) * duration
        result['eta_note'] = 'Estimate from the most recent logged timing; parallel load can change it.'
    return result


def snapshot(root: Path) -> dict:
    status_path = root / 'status.json'
    if not status_path.is_file():
        return {'sampled_at': datetime.now().astimezone().isoformat(), 'root': str(root), 'state': 'not_started'}
    result = json.loads(status_path.read_text(encoding='utf-8'))
    result['sampled_at'] = datetime.now().astimezone().isoformat()
    result['read_only_snapshot'] = True
    for name, job in result.get('jobs', {}).items():
        path = root / name / 'job-result.json'
        if path.is_file():
            job['job_result'] = json.loads(path.read_text(encoding='utf-8'))
        job['progress'] = log_progress(root / name / 'training.log', result.get('steps_per_profile', 100000))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), indent=2), flush=True)


if __name__ == '__main__':
    main()
