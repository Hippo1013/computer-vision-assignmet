"""Audit completed stage-1 records; optionally replay all saved checkpoints on CUDA."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,default=BASE/'outputs/stage1_20261029_train_test')
    p.add_argument('--replay',action='store_true')
    args=p.parse_args()
    root=args.output
    protocol=read(root/'protocol.json')
    runs=protocol['config']['runs']
    assert len(runs)==5 and read(root/'status.json')['state']=='complete'
    assert read(root/'runner_status.json')['state']=='complete'
    assert protocol['common']['batch_size']==1280
    assert protocol['common']['seed']==20261029
    assert protocol['data_protocol']['training_count']==50000
    assert protocol['data_protocol']['test_count']==10000
    assert protocol['data_protocol']['validation_count']==0
    assert digest(BASE/'train_stage1.py')==protocol['source_sha256']
    assert digest(BASE/'cv_homework2/resnet-cifar100.py')==protocol['model_sha256']
    data=BASE/'cv_homework2/data/cifar-100-python'
    with (data/'test').open('rb') as f:
        targets=pickle.load(f,encoding='latin1')['fine_labels']
    assert len(targets)==10000
    for name,expected in protocol['data_sha256'].items():
        assert digest(data/name)==expected
    result={'passed':True,'runs':{},'data_hashes_match':True,'checkpoint_replay':args.replay}
    histories={}
    for config in runs:
        name=config['id'];folder=root/name
        h=read(folder/'history.json');histories[name]=h
        assert [r['epoch'] for r in h]==list(range(1,51))
        effective=read(folder/'config.json')
        assert effective['training_count']==50000 and effective['test_count']==10000
        assert effective['batch_size']==1280 and effective['seed']==20261029
        assert effective['optimizer']==config['optimizer'] and effective['lr']==config['lr']
        for r in h:
            lr=config['lr']*.1**sum(r['epoch']>m for m in config.get('milestones',[]))
            assert math.isclose(r['lr'],lr,rel_tol=1e-12)
            assert math.isfinite(r['train_loss']) and math.isfinite(r['test']['loss'])
            assert 0<=r['test']['accuracy']<=1 and r['test']['count']==10000
            assert r['epoch_seconds']>=r['train_seconds']>0
        best=max(h,key=lambda r:r['test']['accuracy'])
        audit={}
        for tag,reference in [('last',h[-1]),('best',best)]:
            saved=read(folder/(tag+'_test.json'))
            assert saved['epoch']==reference['epoch'] and saved['metrics']==reference['test']
            assert saved['targets']==targets and len(saved['predictions'])==10000
            assert all(0<=i<100 for i in saved['predictions'])
            accuracy=sum(a==b for a,b in zip(saved['predictions'],targets))/10000
            assert accuracy==saved['metrics']['accuracy']
            cp=folder/(tag+'.pt')
            audit[tag]={'epoch':saved['epoch'],'accuracy':accuracy,'checkpoint_sha256':digest(cp)}
        result['runs'][name]=audit
    for a,b in zip(histories['sgd_lr01'][:30],histories['sgd_step'][:30]):
        assert a['train_loss']==b['train_loss'] and a['train_accuracy']==b['train_accuracy'] and a['test']==b['test']
    result['fixed_vs_step_first_30_epochs_identical']=True
    if args.replay:
        import torch
        import torchvision
        import train_stage1 as t
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda')
        dataset=t.DeviceCIFAR(torchvision.datasets.CIFAR100(str(data.parent),train=False),device,False)
        for config in runs:
            name=config['id'];folder=root/name
            model=t.model_new(device)
            for tag in ['last','best']:
                state=t.load_checkpoint(folder/(tag+'.pt'),model)
                saved=read(folder/(tag+'_test.json'))
                assert state['epoch']==saved['epoch'] and state['history']==histories[name][:state['epoch']]
                metrics,pred,true=t.evaluate(model,dataset.batches(1280,20261029,False),device)
                assert pred==saved['predictions'] and true==saved['targets']
                assert metrics['accuracy']==saved['metrics']['accuracy']
                assert math.isclose(metrics['loss'],saved['metrics']['loss'],rel_tol=1e-7,abs_tol=1e-7)
                result['runs'][name][tag]['replay_exact_predictions']=True
                print(name,tag,'replay passed',flush=True)
            del model,state
    (root/('verification_replay.json' if args.replay else 'verification_local.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
