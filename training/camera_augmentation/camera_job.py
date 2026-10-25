"""One finite camera augmentation training/evaluation job, without hardware access."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys



def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def make_train_config(source: Path, directory: Path, dataset: Path, steps: int, seed: int, profile: str) -> dict:
    cfg = copy.deepcopy(json.loads((source / 'train_config.json').read_text(encoding='utf-8')))
    cfg['policy']['pretrained_path'] = str(source)
    cfg['policy']['push_to_hub'] = False
    cfg['policy']['device'] = 'cuda'
    cfg['dataset']['root'] = str(dataset)
    cfg['dataset']['image_transforms']['enable'] = False
    cfg['dataset']['eval_split'] = 0.0
    cfg.setdefault('wandb', {})['enable'] = False
    cfg.update(resume=False, checkpoint_path=None, steps=steps, batch_size=8, num_workers=4,
               seed=seed, output_dir=str(directory / 'output'), job_name=f'act_camera_{profile}',
               save_freq=20000, log_freq=200, save_checkpoint=True, save_checkpoint_to_hub=False,
               eval_steps=0)
    return cfg


def run_command(command: list[str], log_path: Path, env: dict, cwd: Path) -> int:
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return process.wait()


def run_job(args) -> int:
    directory = args.root / args.profile
    directory.mkdir(parents=True, exist_ok=False)
    source = args.source
    baseline = args.profile == 'baseline'
    model = source if baseline else directory / 'output/checkpoints' / f'{args.steps:06d}' / 'pretrained_model'
    # Resolving a venv's symlink selects the bare interpreter and loses its packages.
    python = sys.executable
    scripts = Path(__file__).resolve().parent
    train_command = [python, str(scripts / 'train_camera_specific.py'), '--config_path=' + str(directory / 'train.json')]
    eval_command = [python, str(scripts / 'evaluate_camera_model.py'), '--model', str(model),
                    '--dataset', str(args.dataset), '--output', str(directory / 'evaluation.json'),
                    '--device', 'cuda', '--samples', '48', '--seed', str(args.seed)]
    environment = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                       SO101_AUG_PROFILE=args.profile, PYTHONUNBUFFERED='1')
    report = {'profile': args.profile, 'pid': os.getpid(), 'started_at': now(), 'state': 'preparing',
              'source_checkpoint': str(source), 'model': str(model), 'train_command': None if baseline else train_command,
              'evaluation_command': eval_command, 'cwd': str(args.base), 'baseline_evaluation_only': baseline,
              'train_exit_code': None, 'evaluation_exit_code': None, 'checkpoint_saved': False,
              'environment_overrides': {key: environment[key] for key in (
                  'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'WANDB_MODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                  'TOKENIZERS_PARALLELISM', 'SO101_AUG_PROFILE', 'PYTHONUNBUFFERED')}, 'robot_access': False}
    result_path = directory / 'job-result.json'
    atomic_json(result_path, report)
    # Each manual evaluation uses the identical index-selection seed and cases.
    script = '#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(args.base)) + '\n'
    script += 'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
    # A caller supplies a new report path so an earlier report is never overwritten.
    manual_command = eval_command.copy()
    manual_command[manual_command.index('--output') + 1] = '__OUTPUT_ARGUMENT__'
    script += ': "${1:?Usage: evaluate.sh NEW_REPORT.json}"\n'
    script += ' '.join('"$1"' if item == '__OUTPUT_ARGUMENT__' else shlex.quote(item) for item in manual_command) + '\n'
    (directory / 'evaluate.sh').write_text(script, encoding='utf-8')
    (directory / 'evaluate.sh').chmod(0o755)
    try:
        if not baseline:
            config = make_train_config(source, directory, args.dataset, args.steps, args.seed, args.profile)
            atomic_json(directory / 'train.json', config)
            report.update(state='training', training_started_at=now())
            atomic_json(result_path, report)
            report['train_exit_code'] = run_command(train_command, directory / 'training.log', environment, args.base)
            report['checkpoint_saved'] = (model / 'model.safetensors').is_file()
            report['training_completed_at'] = now()
            atomic_json(result_path, report)
            if report['train_exit_code'] != 0:
                report.update(state='training_failed', exit_code=report['train_exit_code'], completed_at=now())
                atomic_json(result_path, report)
                return report['train_exit_code']
            if not report['checkpoint_saved']:
                raise FileNotFoundError(f'Training exited successfully but final model is absent: {model}')
        else:
            report['checkpoint_saved'] = (model / 'model.safetensors').is_file()
        report.update(state='evaluating', evaluation_started_at=now())
        atomic_json(result_path, report)
        report['evaluation_exit_code'] = run_command(eval_command, directory / 'evaluation.log', environment, args.base)
        code = report['evaluation_exit_code']
        report.update(state='completed' if code == 0 else 'evaluation_failed', exit_code=code, completed_at=now())
        atomic_json(result_path, report)
        return code
    except Exception as error:
        report.update(state='failed', error=repr(error), exit_code=1, completed_at=now(),
                      checkpoint_saved=(model / 'model.safetensors').is_file())
        atomic_json(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path.cwd())
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    for name in ("root", "base", "source", "dataset"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.profile or not args.profile.replace('_', '').isalnum() or args.steps < 1:
        parser.error('profile must be a simple name and steps must be positive')
    raise SystemExit(run_job(args))


if __name__ == '__main__':
    main()
