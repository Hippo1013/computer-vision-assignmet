"""Audit the three Adam supplements and optionally replay all six new checkpoints."""
import argparse
import json
import math
import pickle
from pathlib import Path
from verify_stage1 import read,digest,text_digest_matches

BASE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--replay',action='store_true')
    args=p.parse_args()
    root=BASE/'outputs/adam_extra_lr_20261029'
    basefile=BASE/'outputs/stage1_20261029_train_test/protocol.json'
    base=read(basefile);protocol=read(root/'protocol.json')
    assert read(root/'status.json')['state']=='complete'
    assert read(root/'runner_status.json')['state']=='complete'
    assert protocol['base_protocol_sha256']==digest(basefile)
    for key in ['common','environment','data_protocol','data_sha256','source_sha256','model_sha256']:
        assert protocol[key]==base[key]
    assert text_digest_matches(BASE/'train_stage1.py',protocol['source_sha256'])
    assert text_digest_matches(BASE/'train_adam_extra.py',protocol['supplement_source_sha256'])
    assert text_digest_matches(BASE/'cv_homework2/resnet-cifar100.py',protocol['model_sha256'])
    runs=protocol['config']['runs']
    assert [(r['id'],r['lr'],r.get('milestones',[])) for r in runs]==[('adam_lr001',.01,[]),('adam_lr01',.1,[]),('adam_step',.001,[30,40])]
    common=protocol['common']
    assert common['epochs']==50 and common['batch_size']==1280 and common['seed']==20261029
    data=BASE/'cv_homework2/data/cifar-100-python'
    for name,value in protocol['data_sha256'].items():assert digest(data/name)==value
    with (data/'test').open('rb') as f:targets=pickle.load(f,encoding='latin1')['fine_labels']
    result={'passed':True,'data_and_base_protocol_hashes_match':True,'checkpoint_replay':args.replay,'runs':{}}
    histories={}
    for config in runs:
        name=config['id'];folder=root/name
        history=read(folder/'history.json');histories[name]=history
        assert [r['epoch'] for r in history]==list(range(1,51))
        effective=read(folder/'config.json')
        assert all(effective[k]==v for k,v in common.items())
        assert effective['training_count']==50000 and effective['test_count']==10000
        assert all(effective[k]==v for k,v in config.items())
        for row in history:
            expected=config['lr']*.1**sum(row['epoch']>m for m in config.get('milestones',[]))
            assert math.isclose(row['lr'],expected,rel_tol=1e-12)
            assert math.isfinite(row['train_loss']) and math.isfinite(row['test']['loss'])
            assert row['test']['count']==10000 and 0<=row['test']['accuracy']<=1
            assert row['epoch_seconds']>=row['train_seconds']>0
        peak=max(history,key=lambda r:r['test']['accuracy'])
        result['runs'][name]={}
        for tag,row in [('last',history[-1]),('best',peak)]:
            record=read(folder/(tag+'_test.json'))
            assert record['epoch']==row['epoch'] and record['metrics']==row['test']
            assert record['targets']==targets and len(record['predictions'])==10000
            assert all(0<=x<100 for x in record['predictions'])
            accuracy=sum(x==y for x,y in zip(record['predictions'],targets))/10000
            assert accuracy==record['metrics']['accuracy']
            result['runs'][name][tag]={'epoch':row['epoch'],'accuracy':accuracy,'checkpoint_sha256':digest(folder/(tag+'.pt'))}
    control=read(BASE/'outputs/stage1_20261029_train_test/adam_lr0001/history.json')
    for x,y in zip(control[:30],histories['adam_step'][:30]):
        assert x['train_loss']==y['train_loss'] and x['train_accuracy']==y['train_accuracy'] and x['test']==y['test']
    result['dynamic_and_fixed_first_30_epochs_identical']=True
    if args.replay:
        import torch,torchvision
        import train_stage1 as t
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda:0')
        dataset=t.DeviceCIFAR(torchvision.datasets.CIFAR100(str(data.parent),train=False),device,False)
        for config in runs:
            name=config['id'];folder=root/name;model=t.model_new(device)
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
