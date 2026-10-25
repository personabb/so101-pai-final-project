"""Compare completed offline reports only after verifying identical evaluation inputs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports={path.parent.name:json.loads(path.read_text()) for path in args.root.glob('*/evaluation.json')}
    if 'baseline' not in reports:
        raise ValueError('Original baseline evaluation.json is required')
    baseline=reports['baseline']
    rows=[]
    for name,report in reports.items():
        if report['indices']!=baseline['indices'] or report['seed']!=baseline['seed']:
            raise ValueError('Evaluation sample selection differs for '+name)
        if set(report['aggregates'])!=set(baseline['aggregates']):
            raise ValueError('Evaluation cases differ for '+name)
        for case,item in report['aggregates'].items():
            ref=baseline['aggregates'][case]
            if item['input_sha256']!=ref['input_sha256']:
                raise ValueError('Stress input differs: '+name+'/'+case)
            rows.append({'profile':name,'case':case,'normalized_mae':item['normalized_mae'],
                         'mae_change_vs_original_200k':item['normalized_mae']-ref['normalized_mae'],
                         'delta_from_clean':item['normalized_delta_from_clean'],
                         'first_action_mae':item['normalized_first_action_mae'],
                         'samples':item['samples']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'reports':list(reports),'rows':len(rows),'output':str(args.output),
                      'warning':'In-domain offline errors; not task success. Extra-training effects are not isolated.'}))


if __name__=='__main__':
    main()
