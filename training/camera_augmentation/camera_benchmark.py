"""Measure complete camera-specific ACT training at one, two, and three processes."""
import argparse
import sys
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--base",type=Path,default=Path.cwd())
    args=parser.parse_args()
    ROOT=args.root.resolve()
    SOURCE=args.source.resolve(strict=True)
    BASE=args.base.resolve(strict=True)
    dataset=args.dataset.resolve(strict=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    result_path=ROOT/'benchmark.json'
    if result_path.exists():
        raise RuntimeError('Benchmark already exists')
    template=json.loads((SOURCE/'train_config.json').read_text())
    rows=[]
    for concurrency in [1,2,3]:
        active=[]
        started=time.monotonic()
        for number in range(concurrency):
            run=ROOT/f'benchmark-{concurrency}-{number}'
            run.mkdir(exist_ok=False)
            cfg=json.loads(json.dumps(template))
            cfg['dataset']['image_transforms']['enable']=False
            cfg['dataset']['root']=str(dataset)
            cfg['policy']['pretrained_path']=str(SOURCE)
            cfg['policy']['push_to_hub']=False
            cfg['wandb']['enable']=False
            cfg.update(resume=False,checkpoint_path=None,steps=120,output_dir=str(run/'output'),
                       job_name=f'camera_benchmark_{concurrency}_{number}',log_freq=20,
                       save_checkpoint=False,save_checkpoint_to_hub=False)
            path=run/'train.json'
            path.write_text(json.dumps(cfg,indent=2))
            env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled',
                     SO101_AUG_PROFILE='combined',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,str(Path(__file__).with_name('train_camera_specific.py')),'--config_path='+str(path)]
            log=(run/'training.log').open('w')
            proc=subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((proc,log,run,command))
        codes=[]
        rates=[]
        for proc,log,run,command in active:
            code=proc.wait(timeout=900)
            log.close()
            codes.append(code)
            if code:
                for other,_,_,_ in active:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f'Benchmark failed: {run}; code {code}')
            text=(run/'training.log').read_text()
            times=re.findall(r'updt_s:([0-9.]+) data_s:([0-9.]+)',text)
            final=[float(a)+float(b) for a,b in times[-3:]]
            rates.append(1/(sum(final)/len(final)))
        elapsed=time.monotonic()-started
        row={'concurrency':concurrency,'steps_per_process':120,'exit_codes':codes,
             'elapsed_s':elapsed,'complete_steps_per_s':concurrency*120/elapsed,
             'estimated_steady_steps_per_s':sum(rates),'per_process_steps_per_s':rates}
        rows.append(row)
        result_path.write_text(json.dumps({'started_at':datetime.now().astimezone().isoformat(),'rows':rows},indent=2))
        print(json.dumps(row),flush=True)
    # Prefer two processes unless the third provides at least 10% more steady throughput.
    two=rows[1]['estimated_steady_steps_per_s']
    three=rows[2]['estimated_steady_steps_per_s']
    chosen=3 if three>=two*1.1 else 2
    result={'rows':rows,'chosen_concurrency':chosen,
            'selection':'Use three if >=10% steady throughput improvement over two; otherwise use two.',
            'cpu_threads_per_process':2,'dataloader_workers_per_process':4,'exit_code':0}
    result_path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
