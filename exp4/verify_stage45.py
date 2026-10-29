"""Verify stage-four records and optionally replay the eight new checkpoints."""
import argparse
import json
import math
import pickle
from pathlib import Path
from verify_stage1 import read,digest,text_digest_matches

BASE=Path(__file__).resolve().parent
PARAMETERS={'resnet10':4949412,'resnet18':11220132,'resnet50':23705252}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--normal-batch',action='store_true')
    args=parser.parse_args()
    root=BASE/'outputs'/('stage4_batch_20261029' if args.normal_batch else 'stage45_20261029')
    protocol=read(root/'protocol.json');config=protocol['config'];common=config['common'];capacity=protocol['capacity']
    assert read(root/'status.json')['state']=='stage4_complete_awaiting_user'
    assert read(root/'runner_status.json')['state']=='stage4_complete_awaiting_user'
    assert config['stage5']['enabled'] is False
    assert not read(root/'status.json')['stage5_started']
    assert common['epochs']==50 and common['seed']==20261029
    if args.normal_batch:
        assert common['update_rule']=='one_optimizer_step_per_batch' and common['batch_sizes']=={'resnet18':640,'resnet50':160}
        assert all(protocol['preflight_checks'].values())
        assert text_digest_matches(BASE/'train_stage45.py',protocol['model_factory_source_sha256'])
    else:assert common['effective_batch_size']==1280
    assert text_digest_matches(BASE/('train_stage4_batch.py' if args.normal_batch else 'train_stage45.py'),protocol['source_sha256'])
    assert text_digest_matches(BASE/'train_stage1.py',protocol['shared_source_sha256'])
    assert text_digest_matches(BASE/'cv_homework2/resnet-cifar100.py',protocol['model_source_sha256'])
    assert all(capacity['checks'].values())
    data=BASE/'cv_homework2/data/cifar-100-python'
    for name,expected in protocol['data_sha256'].items():assert digest(data/name)==expected
    with (data/'test').open('rb') as f:targets=pickle.load(f,encoding='latin1')['fine_labels']
    assert len(targets)==10000
    results={'passed':True,'checkpoint_replay':args.replay,'stage5_started':False,'runs':{},'parameter_counts':PARAMETERS}
    histories={}
    assert len(config['stage4'])==4
    for run in config['stage4']:
        name=run['id'];folder=root/name
        h=read(folder/'history.json');histories[name]=h
        assert [r['epoch'] for r in h]==list(range(1,51))
        effective=read(folder/'config.json');micro=capacity['micro_batch_sizes'][run['architecture']]
        assert effective['batch_size' if args.normal_batch else 'micro_batch_size']==micro
        if args.normal_batch:assert micro==common['batch_sizes'][run['architecture']]
        else:assert 1280%micro==0
        assert all(effective[k]==v for k,v in common.items())
        assert all(effective[k]==v for k,v in run.items())
        assert effective['initial_state_sha256']==capacity['initial_state_sha256'][run['architecture']]
        assert effective['training_count']==50000 and effective['test_count']==10000
        for row in h:
            assert row['training_count']==50000 and row['optimizer_updates']==(math.ceil(50000/micro) if args.normal_batch else 40)
            assert math.isfinite(row['train_loss']) and math.isfinite(row['test']['loss'])
            assert row['train_ce_loss']==row['train_loss'] and row['train_distill_loss']==0.
            expected=run['lr']*.1**sum(row['epoch']>m for m in [30,40])
            assert math.isclose(row['lr'],expected,rel_tol=1e-12)
            assert row['test']['count']==10000 and 0<=row['test']['accuracy']<=1
            assert row['epoch_seconds']>=row['train_seconds']>0
        peak=max(h,key=lambda r:r['test']['accuracy'])
        results['runs'][name]={}
        for tag,row in [('last',h[-1]),('best',peak)]:
            record=read(folder/(tag+'_test.json'))
            assert record['epoch']==row['epoch'] and record['metrics']==row['test']
            assert record['targets']==targets and len(record['predictions'])==10000
            assert all(0<=x<100 for x in record['predictions'])
            accuracy=sum(x==y for x,y in zip(record['predictions'],targets))/10000
            assert accuracy==record['metrics']['accuracy']
            results['runs'][name][tag]={'epoch':row['epoch'],'accuracy':accuracy,'checkpoint_sha256':digest(folder/(tag+'.pt'))}
    def select(architecture):
        return max([r for r in config['stage4'] if r['architecture']==architecture],key=lambda r:histories[r['id']][-1]['test']['accuracy'])
    teacher,student=select('resnet50'),select('resnet18')
    saved=read(root/'stage4_selection.json')
    assert saved['teacher_run']==teacher['id'] and saved['student_baseline_run']==student['id']
    assert saved['teacher_checkpoint']=='last.pt' and saved['stage5_started'] is False
    assert saved['teacher_last_accuracy']==histories[teacher['id']][-1]['test']['accuracy']
    assert saved['student_last_accuracy']==histories[student['id']][-1]['test']['accuracy']
    assert saved['teacher_better']==(saved['teacher_last_accuracy']>saved['student_last_accuracy'])
    results['selection']=saved
    if args.replay:
        import torch,torchvision
        import train_stage45 as stage
        import train_stage1 as t
        device=stage.configure_cuda()
        dataset=t.DeviceCIFAR(torchvision.datasets.CIFAR100(str(data.parent),train=False),device,False)
        for run in config['stage4']:
            name=run['id'];folder=root/name
            t.seed_all(common['seed']);model=stage.model_new(run['architecture'],device)
            assert stage.state_digest(model)==capacity['initial_state_sha256'][run['architecture']]
            assert sum(p.numel() for p in model.parameters())==PARAMETERS[run['architecture']]
            for tag in ['last','best']:
                state=t.load_checkpoint(folder/(tag+'.pt'),model)
                record=read(folder/(tag+'_test.json'))
                assert state['epoch']==record['epoch'] and state['history']==histories[name][:state['epoch']]
                metrics,pred,true=t.evaluate(model,dataset.batches(capacity['micro_batch_sizes'][run['architecture']],common['seed'],False),device)
                assert pred==record['predictions'] and true==record['targets']
                assert metrics['accuracy']==record['metrics']['accuracy']
                assert math.isclose(metrics['loss'],record['metrics']['loss'],rel_tol=1e-7,abs_tol=1e-7)
                results['runs'][name][tag]['replay_exact_predictions']=True
                print(name,tag,'replay passed',flush=True)
            del model,state
    (root/('verification_replay.json' if args.replay else 'verification_local.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
