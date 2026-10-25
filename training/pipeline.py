"""Portable launchers for the official LeRobot commands used in this project.

These commands operate on real local datasets and checkpoints. They do not
open a robot connection or upload datasets/models to a hosted service.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

from training_selection import assert_training_eligible

ROOT = Path(__file__).resolve().parents[1]


def executable(name):
    suffix = '.exe' if sys.platform == 'win32' else ''
    adjacent = Path(sys.executable).parent / (name + suffix)
    resolved = adjacent if adjacent.is_file() else shutil.which(name)
    if not resolved:
        raise FileNotFoundError(f'{name} is not installed in the active environment')
    return str(resolved)


def check_source(path):
    path = path.resolve(strict=True)
    assert_training_eligible(path)
    if (path / 'mock_manifest.json').exists():
        raise ValueError('Only real demonstration datasets are accepted')
    info = json.loads((path / 'meta/info.json').read_text(encoding='utf-8'))
    if info['total_episodes'] <= 0 or info['total_frames'] <= 0:
        raise ValueError('Dataset is empty')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    merge = sub.add_parser('merge', help='Merge selected real datasets through the official editor')
    merge.add_argument('--roots', nargs='+', type=Path, required=True)
    merge.add_argument('--repo-ids', nargs='+', required=True)
    merge.add_argument('--output', type=Path, required=True)
    merge.add_argument('--repo-id', default='local/so101-merged')
    train = sub.add_parser('train', help='Train ACT with the configured hyperparameters')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--repo-id', default='local/so101-merged')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    resume = sub.add_parser('resume', help='Resume the official optimizer state into a separate output directory')
    resume.add_argument('--train-config', type=Path, required=True)
    resume.add_argument('--output', type=Path, required=True)
    resume.add_argument('--steps', type=int, default=200000)
    a = p.parse_args()
    if version('lerobot') != '0.6.1':
        raise RuntimeError('This project uses LeRobot 0.6.1 with the included dataset reader patch')
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.operation == 'merge':
        if len(a.roots) != len(a.repo_ids):
            p.error('Each dataset root needs one repo ID')
        roots = [str(check_source(path)) for path in a.roots]
        command = [executable('lerobot-edit-dataset'), f'--new_repo_id={a.repo_id}',
                   f'--new_root={output}', '--operation.type=merge',
                   '--operation.repo_ids=' + json.dumps(a.repo_ids),
                   '--operation.roots=' + json.dumps(roots)]
    elif a.operation == 'train':
        dataset = check_source(a.dataset)
        command = [executable('lerobot-train'),
                   f'--config_path={ROOT / "configs/training_initial.json"}',
                   f'--dataset.root={dataset}', f'--dataset.repo_id={a.repo_id}',
                   f'--output_dir={output}', f'--policy.device={a.device}',
                   '--policy.push_to_hub=false', '--wandb.enable=false',
                   '--save_checkpoint_to_hub=false']
    else:
        config = a.train_config.resolve(strict=True)
        saved = json.loads(config.read_text(encoding='utf-8'))
        if a.steps <= int(saved['steps']):
            p.error('The cumulative step count must exceed the saved training step count')
        if not (config.parent / 'model.safetensors').is_file():
            raise FileNotFoundError('The checkpoint weights next to train_config.json are missing')
        command = [executable('lerobot-train'), f'--config_path={config}', '--resume=true',
                   f'--steps={a.steps}', f'--output_dir={output}', '--job_name=act_so101_resume',
                   '--policy.push_to_hub=false', '--wandb.enable=false',
                   '--save_checkpoint_to_hub=false']
    print(json.dumps({'argv': command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
