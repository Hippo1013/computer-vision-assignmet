"""Stage four with one optimizer update per actual batch; no gradient accumulation."""
import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
import train_stage1 as t
from train_stage45 import model_new,state_digest,configure_cuda,select_teacher_student,COURSE,BASE

OUT=BASE/'outputs/stage4_batch_20261029'


def batch_update(model,optimizer,x,y):
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    loss=nn.functional.cross_entropy(logits,y)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite_loss')
    loss.backward()
    flags=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not flags or not torch.stack(flags).all():raise FloatingPointError('nonfinite_gradient')
    optimizer.step()
    return {'loss_sum':loss.item()*len(y),'ce_sum':loss.item()*len(y),'distill_sum':0.,
            'correct':(logits.argmax(1)==y).sum().item(),'count':len(y)}


def update_checks(train,device,common):
    checks={}
    for batch in [640,160]:
        t.seed_all(common['seed'])
        toy=nn.Linear(3,5).to(device);optimizer=torch.optim.SGD(toy.parameters(),lr=.001)
        calls=[];handle=optimizer.register_step_post_hook(lambda *args:calls.append(1))
        total=0;last_count=0
        for start in range(0,50000,batch):
            count=min(batch,50000-start)
            x=torch.randn(count,3,device=device);y=torch.arange(count,device=device)%5
            row=batch_update(toy,optimizer,x,y);total+=row['count'];last_count=count
        handle.remove()
        assert total==50000 and last_count==80 and len(calls)==math.ceil(50000/batch)
    checks['one_step_per_batch_79_or_313_updates_and_80_tail']=True
    del toy,optimizer,x,y
    t.seed_all(common['seed'])
    model=model_new('resnet18',device);reference=model_new('resnet18',device)
    reference.load_state_dict(model.state_dict())
    a=t.optimizer_new(model,{'optimizer':'adam','lr':.001})
    b=t.optimizer_new(reference,{'optimizer':'adam','lr':.001})
    x,y=next(train.batches(4,common['seed']+2,True))
    model.train();batch_update(model,a,x,y)
    reference.train();b.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(reference(x),y).backward();b.step()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
    checks['matches_direct_pytorch_step_exactly']=True
    with tempfile.TemporaryDirectory(prefix='exp4-normal-batch-check-') as temp:
        path=Path(temp)/'resume.pt'
        t.save_checkpoint(path,model,a,1,[],-1.)
        batch_update(model,a,x,y)
        expected={k:v.detach().clone() for k,v in model.state_dict().items()}
        t.load_checkpoint(path,model,a);model.train();batch_update(model,a,x,y)
        assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
    checks['model_optimizer_resume_exact']=True
    del model,reference,a,b,x,y,expected
    gc.collect();torch.cuda.empty_cache()
    for architecture in ['resnet18','resnet50']:
        t.seed_all(common['seed']);model=model_new(architecture,device)
        optimizer=t.optimizer_new(model,{'optimizer':'adam','lr':.001})
        x,y=next(train.batches(common['batch_sizes'][architecture],common['seed']+3,True))
        model.train();batch_update(model,optimizer,x,y)
        assert state_digest(model)
        del model,optimizer,x,y
        gc.collect();torch.cuda.empty_cache()
    checks['both_selected_batches_fit_with_adam']=True
    for lr in [.1,.001]:
        expected=[lr,lr,lr*.1,lr*.1,lr*.01,lr*.01]
        actual=[t.lr_at({'lr':lr,'milestones':[30,40]},epoch) for epoch in [1,30,31,40,41,50]]
        assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(expected,actual))
    checks['learning_rate_boundaries']=True
    return checks


def run_one(config, common, capacity, train, test, out, device):
    folder=out/config['id'];folder.mkdir(exist_ok=True)
    batch_size=common['batch_sizes'][config['architecture']]
    effective={**common,**config,'batch_size':batch_size,'training_count':len(train),
               'test_count':len(test),'initial_state_sha256':capacity['initial_state_sha256'][config['architecture']]}
    cp=folder/'config.json'
    if cp.exists():assert json.loads(cp.read_text(encoding='utf-8'))==effective,'Configuration changed'
    else:t.write_json(cp,effective)
    t.seed_all(common['seed']);model=model_new(config['architecture'],device)
    assert state_digest(model)==effective['initial_state_sha256']
    optimizer=t.optimizer_new(model,config)
    history,best,start=[],-1.,1
    last=folder/'last.pt'
    if last.exists():
        state=t.load_checkpoint(last,model,optimizer)
        history,best,start=state['history'],state['best'],state['epoch']+1
        del state
    for epoch in range(start,common['epochs']+1):
        if (out/'pause.request').exists():raise t.PauseRequested(config['id'])
        t.seed_all(common['seed']+epoch)
        for group in optimizer.param_groups:group['lr']=t.lr_at(config,epoch)
        model.train(); torch.cuda.synchronize();begin=time.perf_counter()
        sums={'loss_sum':0.,'ce_sum':0.,'distill_sum':0.,'correct':0,'count':0}
        updates=0
        try:
            for x,y in train.batches(batch_size,common['seed']+epoch,True):
                row=batch_update(model,optimizer,x,y)
                for key in sums:sums[key]+=row[key]
                updates+=1
        except FloatingPointError as error:
            t.write_json(folder/'failure.json',{'epoch':epoch,'reason':str(error)})
            raise
        torch.cuda.synchronize();train_seconds=time.perf_counter()-begin
        assert sums['count']==50000 and updates==math.ceil(len(train)/batch_size)
        metrics,pred,targets=t.evaluate(model,test.batches(batch_size,common['seed'],False),device)
        torch.cuda.synchronize()
        row={'epoch':epoch,'lr':t.lr_at(config,epoch),'train_loss':sums['loss_sum']/sums['count'],
             'train_ce_loss':sums['ce_sum']/sums['count'],'train_distill_loss':0.,
             'train_accuracy':sums['correct']/sums['count'],'training_count':sums['count'],
             'optimizer_updates':updates,'test':metrics,'train_seconds':train_seconds,
             'epoch_seconds':time.perf_counter()-begin}
        history.append(row)
        improved=metrics['accuracy']>best
        if improved:
            best=metrics['accuracy']
            t.save_checkpoint(folder/'best.pt',model,optimizer,epoch,history,best)
            t.write_json(folder/'best_test.json',{'epoch':epoch,'metrics':metrics,'checkpoint':'best.pt','predictions':pred,'targets':targets})
        t.save_checkpoint(last,model,optimizer,epoch,history,best)
        t.write_json(folder/'last_test.json',{'epoch':epoch,'metrics':metrics,'checkpoint':'last.pt','predictions':pred,'targets':targets})
        t.write_json(folder/'history.json',history)
        t.write_json(out/'status.json',{'state':'training','stage':4,'run':config['id'],'epoch':epoch,'epochs':common['epochs']})
        print(json.dumps({'run':config['id'],**row}),flush=True)
    # Recover metadata if a previous interruption occurred after atomic checkpoint commit.
    t.write_json(folder/'history.json',history)
    for tag,reference in [('last',history[-1]),('best',max(history,key=lambda r:r['test']['accuracy']))]:
        record_path=folder/(tag+'_test.json')
        needs_repair=not record_path.exists()
        if not needs_repair:
            record=json.loads(record_path.read_text(encoding='utf-8'))
            needs_repair=record['epoch']!=reference['epoch'] or record['metrics']!=reference['test']
        if needs_repair:
            t.load_checkpoint(folder/(tag+'.pt'),model)
            metrics,pred,targets=t.evaluate(model,test.batches(batch_size,common['seed'],False),device)
            assert metrics==reference['test']
            t.write_json(record_path,{'epoch':reference['epoch'],'metrics':metrics,'checkpoint':tag+'.pt','predictions':pred,'targets':targets})
    return history


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a+b')
    if platform.system()=='Windows':
        import msvcrt
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((BASE/'stage4_batch.json').read_text(encoding='utf-8'));common=config['common']
    assert common['update_rule']=='one_optimizer_step_per_batch' and common['epochs']==50
    assert common['batch_sizes']=={'resnet18':640,'resnet50':160} and config['stage5']['enabled'] is False
    device=configure_cuda()
    capacity_file=BASE/'outputs/stage45_20261029/protocol.json'
    source=json.loads(capacity_file.read_text(encoding='utf-8'));capacity=source['capacity']
    assert t.sha(BASE/'train_stage45.py')==source['source_sha256']
    assert t.sha(BASE/'train_stage1.py')==source['shared_source_sha256']
    assert t.sha(COURSE)==source['model_source_sha256']
    assert common['batch_sizes']==capacity['micro_batch_sizes']
    assert platform.python_version()==source['environment']['python']
    assert torch.__version__==source['environment']['torch'] and torchvision.__version__==source['environment']['torchvision']
    assert np.__version__==source['environment']['numpy']
    data=BASE/'cv_homework2/data'
    for name,expected in source['data_sha256'].items():assert t.sha(data/'cifar-100-python'/name)==expected
    raw_train=torchvision.datasets.CIFAR100(str(data),train=True)
    raw_test=torchvision.datasets.CIFAR100(str(data),train=False)
    assert len(raw_train)==50000 and len(raw_test)==10000
    train=t.DeviceCIFAR(raw_train,device,True);test=t.DeviceCIFAR(raw_test,device,False)
    del raw_train,raw_test
    free,total=torch.cuda.mem_get_info()
    budget=min(capacity['memory_budget_bytes'],int(torch.cuda.memory_allocated()+free-384*1024**2))
    torch.cuda.set_per_process_memory_fraction(budget/total,device)
    pp=out/'protocol.json'
    if not pp.exists():
        checks=update_checks(train,device,common)
        protocol={'config':config,'capacity':capacity,'preflight_checks':checks,'memory_budget_bytes':budget,
                  'capacity_source_protocol_sha256':t.sha(capacity_file),'environment':source['environment'],
                  'data_protocol':source['data_protocol'],'data_sha256':source['data_sha256'],
                  'source_sha256':t.sha(__file__),'model_factory_source_sha256':t.sha(BASE/'train_stage45.py'),
                  'shared_source_sha256':t.sha(BASE/'train_stage1.py'),'model_source_sha256':t.sha(COURSE),
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()}
        t.write_json(pp,protocol);t.write_json(out/'preflight.json',checks)
    else:
        protocol=json.loads(pp.read_text(encoding='utf-8'))
        assert protocol['source_sha256']==t.sha(__file__) and protocol['config']==config,'Source/config changed'
    if args.preflight_only:
        print(json.dumps({'preflight':'passed','batch_sizes':common['batch_sizes']}),flush=True)
        return 0
    histories,failed={},[]
    for run in config['stage4']:
        try:histories[run['id']]=run_one(run,common,capacity,train,test,out,device)
        except t.PauseRequested:
            t.write_json(out/'status.json',{'state':'paused','stage':4,'run':run['id'],'completed_runs':list(histories)})
            return 75
        except FloatingPointError:failed.append(run['id'])
        gc.collect();torch.cuda.empty_cache()
    if failed:
        t.write_json(out/'status.json',{'state':'incomplete','stage':4,'completed_runs':list(histories),'failed_runs':failed,'stage5_started':False})
        return 1
    t.write_json(out/'stage4_selection.json',select_teacher_student(config,histories))
    t.write_json(out/'status.json',{'state':'stage4_complete_awaiting_user','stage':4,'completed_runs':list(histories),'failed_runs':[],'stage5_started':False})
    return 0


if __name__=='__main__':sys.exit(main())
