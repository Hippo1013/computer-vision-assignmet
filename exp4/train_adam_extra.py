"""Supplement fixed and scheduled Adam learning rates, reusing the frozen stage-1 training implementation."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision
import train_stage1 as t

BASE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,default=BASE/'outputs/adam_extra_lr_20261029')
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a+b')
    if platform.system()=='Windows':
        import msvcrt
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base_file=BASE/'outputs/stage1_20261029_train_test/protocol.json'
    base=json.loads(base_file.read_text(encoding='utf-8'))
    assert t.sha(BASE/'train_stage1.py')==base['source_sha256']
    assert t.sha(BASE/'cv_homework2/resnet-cifar100.py')==base['model_sha256']
    config=json.loads((BASE/'adam_extra.json').read_text(encoding='utf-8'))
    common=base['common']
    assert common['batch_size']==1280 and common['epochs']==50
    assert [(r['optimizer'],r['lr']) for r in config['runs']]==[('adam',.01),('adam',.1),('adam',.001)]
    assert config['runs'][-1]['milestones']==[30,40]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    device=torch.device('cuda:0')
    data=BASE/'cv_homework2/data'
    for name,expected in base['data_sha256'].items():
        assert t.sha(data/'cifar-100-python'/name)==expected
    raw_train=torchvision.datasets.CIFAR100(str(data),train=True)
    raw_test=torchvision.datasets.CIFAR100(str(data),train=False)
    assert len(raw_train)==50000 and len(raw_test)==10000
    train=t.DeviceCIFAR(raw_train,device,True)
    test=t.DeviceCIFAR(raw_test,device,False)
    del raw_train,raw_test
    protocol={'config':{'common':common,'runs':config['runs'],'data_protocol':'full_train_test_each_epoch'},
              'common':common,'base_protocol_sha256':t.sha(base_file),
              'source_sha256':base['source_sha256'],'model_sha256':base['model_sha256'],
              'supplement_source_sha256':t.sha(__file__),
              'data_protocol':base['data_protocol'],'data_sha256':base['data_sha256'],
              'environment':base['environment']}
    pp=out/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text(encoding='utf-8'))==protocol,'Protocol changed; use a new output directory'
    else:
        t.write_json(pp,protocol)
        t.write_json(out/'launch.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()})
    completed,failed=[],[]
    for run in config['runs']:
        try:
            t.run_one(run,common,train,test,out,device)
            completed.append(run['id'])
        except t.PauseRequested:
            t.write_json(out/'status.json',{'state':'paused','run':run['id'],'completed_runs':completed})
            return 75
        except FloatingPointError:
            failed.append(run['id'])
    t.write_json(out/'status.json',{'state':'incomplete' if failed else 'complete','completed_runs':completed,'failed_runs':failed})
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
