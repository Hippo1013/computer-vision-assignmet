"""CIFAR-adapted ResNet18/50 comparisons with reproducible gradient accumulation.

Stage four stops after four runs. Stage five is intentionally disabled pending report.
The distillation step is exercised only in capacity/correctness checks at this stage.
"""
import argparse
import gc
import hashlib
import importlib.util
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
from torchvision.models.resnet import BasicBlock, Bottleneck
import train_stage1 as t

BASE = Path(__file__).resolve().parent
OUT = BASE/'outputs/stage45_20261029'
COURSE = BASE/'cv_homework2/resnet-cifar100.py'


def model_new(architecture, device):
    spec = importlib.util.spec_from_file_location('course_resnet_stage45', COURSE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    block, layers = {'resnet18': (BasicBlock, [2,2,2,2]),
                     'resnet50': (Bottleneck, [3,4,6,3])}[architecture]
    return module.ResNetCustom(block, layers, num_classes=100).to(device)


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def configure_cuda():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda:0')


def forward_teacher(teacher, x, chunk):
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        return torch.cat([teacher(part) for part in x.split(chunk)])


def loss_terms(logits, labels, teacher_logits=None, method='ce'):
    ce = nn.functional.cross_entropy(logits, labels)
    distill = torch.zeros((), device=logits.device)
    if method == 'soft':
        assert teacher_logits is not None and not teacher_logits.requires_grad
        temperature = 2.
        distill = nn.functional.kl_div(nn.functional.log_softmax(logits/temperature, dim=1),
                                     nn.functional.softmax(teacher_logits/temperature, dim=1),
                                     reduction='batchmean') * temperature**2
        total = ce + .5 * distill
    elif method == 'logits':
        assert teacher_logits is not None and not teacher_logits.requires_grad
        distill = nn.functional.mse_loss(logits, teacher_logits, reduction='mean')
        total = ce + .1 * distill
    else:
        assert method == 'ce'
        total = ce
    return total, ce, distill


def accumulated_update(model, optimizer, x, y, micro, teacher=None, teacher_chunk=None, method='ce'):
    """One optimizer update per effective batch, including unequal/tail microbatches."""
    optimizer.zero_grad(set_to_none=True)
    teacher_logits = None if teacher is None else forward_teacher(teacher, x, teacher_chunk)
    count, correct = len(y), 0
    total_sum, ce_sum, distill_sum = 0., 0., 0.
    for start in range(0, count, micro):
        stop = min(start+micro, count)
        logits = model(x[start:stop])
        target = None if teacher_logits is None else teacher_logits[start:stop]
        loss, ce, distill = loss_terms(logits, y[start:stop], target, method)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite_loss')
        # Microbatch means must be weighted by actual sample counts, not fixed accumulation steps.
        (loss * ((stop-start)/count)).backward()
        total_sum += loss.item() * (stop-start)
        ce_sum += ce.item() * (stop-start)
        distill_sum += distill.item() * (stop-start)
        correct += (logits.argmax(1) == y[start:stop]).sum().item()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite_gradient')
    optimizer.step()
    return {'loss_sum': total_sum, 'ce_sum': ce_sum, 'distill_sum': distill_sum,
            'correct': correct, 'count': count}


def consistency_checks(device, train):
    checks = {}
    # No BN: unequal microbatches and a short effective tail must match direct full-batch updates.
    for count, micro in [(19,6), (80,64)]:
        t.seed_all(20261029)
        direct = nn.Sequential(nn.Linear(7,13),nn.Tanh(),nn.Linear(13,5)).to(device)
        split = nn.Sequential(nn.Linear(7,13),nn.Tanh(),nn.Linear(13,5)).to(device)
        split.load_state_dict(direct.state_dict())
        x, y = torch.randn(count,7,device=device), torch.arange(count,device=device)%5
        a = torch.optim.SGD(direct.parameters(),lr=.1,momentum=.9)
        b = torch.optim.SGD(split.parameters(),lr=.1,momentum=.9)
        a.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(direct(x),y).backward(); a.step()
        calls = []
        handle = b.register_step_post_hook(lambda *args: calls.append(1))
        accumulated_update(split,b,x,y,micro)
        handle.remove()
        assert len(calls)==1
        assert all(torch.allclose(p,q,atol=2e-7,rtol=2e-6) for p,q in zip(direct.parameters(),split.parameters()))
    checks['weighted_accumulation_and_single_step'] = True
    assert len(range(0,50000,1280))==40 and 50000%1280==80
    checks['40_updates_and_80_sample_tail'] = True
    # Verify the declared scalar loss formulas and gradient connectivity.
    s = torch.randn(4,100,device=device,requires_grad=True)
    teacher_z = torch.randn(4,100,device=device)
    y = torch.arange(4,device=device)
    ce = nn.functional.cross_entropy(s,y)
    kl = (nn.functional.softmax(teacher_z/2,dim=1) *
          (nn.functional.log_softmax(teacher_z/2,dim=1)-nn.functional.log_softmax(s/2,dim=1))).sum()/4 * 4
    assert torch.allclose(loss_terms(s,y,teacher_z,'soft')[0],ce+.5*kl,atol=1e-6)
    assert torch.allclose(loss_terms(s,y,teacher_z,'logits')[0],ce+.1*((s-teacher_z)**2).mean(),atol=1e-6)
    checks['distillation_formulas'] = True
    for architecture, expected_params in [('resnet18',11220132),('resnet50',23705252)]:
        t.seed_all(20261029)
        model = model_new(architecture,device)
        assert sum(p.numel() for p in model.parameters())==expected_params
        assert model.conv1.kernel_size==(3,3) and model.conv1.stride==(1,1)
        assert model(torch.randn(2,3,32,32,device=device)).shape==(2,100)
        del model
    checks['architecture_shapes_and_parameter_counts'] = True
    # Actual BN model: restart after an effective update with optimizer and model buffers restored.
    t.seed_all(20261029)
    student = model_new('resnet18',device)
    optimizer = t.optimizer_new(student,{'optimizer':'adam','lr':.001})
    x,y = next(train.batches(8,20261029+2,True))
    student.train(); accumulated_update(student,optimizer,x,y,4)
    with tempfile.TemporaryDirectory(prefix='exp4-stage45-check-') as temp:
        path = Path(temp)/'resume.pt'
        t.save_checkpoint(path,student,optimizer,1,[],-1.)
        student.train(); accumulated_update(student,optimizer,x,y,4)
        expected = {k:v.detach().clone() for k,v in student.state_dict().items()}
        t.load_checkpoint(path,student,optimizer)
        student.train(); accumulated_update(student,optimizer,x,y,4)
        assert all(torch.equal(expected[k],v) for k,v in student.state_dict().items())
    checks['actual_model_optimizer_resume_exact'] = True
    del student,optimizer,expected,x,y
    # Freeze parameters AND BN buffers, while proving the student changes for both KD methods.
    teacher = model_new('resnet50',device).eval().requires_grad_(False)
    before_teacher = state_digest(teacher)
    for method in ['soft','logits']:
        t.seed_all(20261029)
        student = model_new('resnet18',device)
        before_student = state_digest(student)
        optimizer = t.optimizer_new(student,{'optimizer':'adam','lr':.001})
        x,y = next(train.batches(8,20261029+3,True))
        student.train(); accumulated_update(student,optimizer,x,y,4,teacher,2,method)
        assert state_digest(student)!=before_student and state_digest(teacher)==before_teacher
        assert all(p.grad is None for p in teacher.parameters())
        del student,optimizer,x,y
    checks['teacher_frozen_and_student_updated'] = True
    del teacher
    a = next(train.batches(8,20261029+5,True)); b=next(train.batches(8,20261029+5,True))
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    checks['augmented_data_repeatable'] = True
    assert [round(t.lr_at({'lr':.1,'milestones':[30,40]},e),8) for e in [1,30,31,40,41,50]]==[.1,.1,.01,.01,.001,.001]
    checks['learning_rate_boundaries'] = True
    return checks


def capacity_probe(architecture, micro, device, teacher_chunk=None):
    t.seed_all(20261029)
    model = model_new(architecture,device)
    optimizer = t.optimizer_new(model,{'optimizer':'adam','lr':.001})
    teacher = None
    method = 'ce'
    if teacher_chunk is not None:
        teacher=model_new('resnet50',device).eval().requires_grad_(False)
        method='soft'
    model.train()
    x=torch.randn(1280,3,32,32,device=device)
    y=torch.arange(1280,device=device)%100
    staging=torch.empty(32*1024**2,dtype=torch.uint8,device=device)  # Reserve augmentation staging headroom.
    torch.cuda.reset_peak_memory_stats()
    accumulated_update(model,optimizer,x,y,micro,teacher,teacher_chunk,method)
    torch.cuda.synchronize(); start=time.perf_counter()
    accumulated_update(model,optimizer,x,y,micro,teacher,teacher_chunk,method)
    torch.cuda.synchronize()
    return {'micro_batch_size':micro,'images_per_second':1280/(time.perf_counter()-start),
            'reserved_staging_MiB':32,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'passed':True,
            'mode':'teacher_plus_student' if teacher is not None else 'standalone'}


def preflight(config, train, out, device):
    checks = consistency_checks(device,train)
    gc.collect(); torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info()
    budget=int(torch.cuda.memory_allocated()+free-config['common']['memory_margin_MiB']*1024**2)
    # A hard caching-allocator cap prevents WDDM spilling oversized probes to system RAM.
    torch.cuda.set_per_process_memory_fraction(budget/total,device)
    choices,records={},{}
    for architecture in ['resnet50','resnet18']:
        rows=[]
        for micro in [40,64,80,128,160,256,320,640,1280]:
            try:
                row=capacity_probe(architecture,micro,device,choices.get('resnet50') if architecture=='resnet18' else None)
                row['within_budget']=row['peak_allocated_bytes']<=budget
                rows.append(row)
                print(json.dumps({'architecture':architecture,'capacity_probe':row}),flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append({'micro_batch_size':micro,'passed':False,'reason':'CUDA OOM under dedicated-memory cap'})
            gc.collect();torch.cuda.empty_cache()
            if not rows[-1]['passed'] or not rows[-1].get('within_budget',False):break
        valid=[r for r in rows if r['passed'] and r['within_budget']]
        assert valid,'No approved microbatch fits '+architecture
        fastest=max(r['images_per_second'] for r in valid)
        choices[architecture]=max(r['micro_batch_size'] for r in valid if r['images_per_second']>=.95*fastest)
        records[architecture]=rows
    # Hash the exact pristine initialization for each architecture; these hashes gate every run.
    initial={}
    for architecture in ['resnet18','resnet50']:
        t.seed_all(config['common']['seed']); model=model_new(architecture,device)
        initial[architecture]=state_digest(model)
        del model
    gc.collect();torch.cuda.empty_cache()
    result={'checks':checks,'micro_batch_sizes':choices,'teacher_forward_chunk':choices['resnet50'],
            'initial_state_sha256':initial,'memory_budget_bytes':budget,'device_total_bytes':total,
            'capacity_probes':records}
    t.write_json(out/'preflight.json',result)
    return result


def run_one(config, common, capacity, train, test, out, device):
    folder=out/config['id'];folder.mkdir(exist_ok=True)
    micro=capacity['micro_batch_sizes'][config['architecture']]
    effective={**common,**config,'micro_batch_size':micro,'training_count':len(train),
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
            for x,y in train.batches(common['effective_batch_size'],common['seed']+epoch,True):
                row=accumulated_update(model,optimizer,x,y,micro)
                for key in sums:sums[key]+=row[key]
                updates+=1
        except FloatingPointError as error:
            t.write_json(folder/'failure.json',{'epoch':epoch,'reason':str(error)})
            raise
        torch.cuda.synchronize();train_seconds=time.perf_counter()-begin
        assert sums['count']==50000 and updates==40
        metrics,pred,targets=t.evaluate(model,test.batches(micro,common['seed'],False),device)
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
            metrics,pred,targets=t.evaluate(model,test.batches(micro,common['seed'],False),device)
            assert metrics==reference['test']
            t.write_json(record_path,{'epoch':reference['epoch'],'metrics':metrics,'checkpoint':tag+'.pt','predictions':pred,'targets':targets})
    return history


def select_teacher_student(config,histories):
    def best(architecture):
        return max([r for r in config['stage4'] if r['architecture']==architecture],
                   key=lambda r:histories[r['id']][-1]['test']['accuracy'])
    teacher,student=best('resnet50'),best('resnet18')
    ta=histories[teacher['id']][-1]['test']['accuracy'];sa=histories[student['id']][-1]['test']['accuracy']
    return {'teacher_run':teacher['id'],'student_baseline_run':student['id'],'teacher_checkpoint':'last.pt',
            'teacher_last_accuracy':ta,'student_last_accuracy':sa,'teacher_better':ta>sa,
            'selection_rule':config['stage5']['selection'],'stage5_started':False,
            'stop_reason':'User requests stage-four report before stage five'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=BASE/'stage45.json')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a+b')
    if platform.system()=='Windows':
        import msvcrt
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(args.config.read_text(encoding='utf-8'));common=config['common']
    assert config['stage5']['enabled'] is False,'Stage five needs a new explicit instruction'
    assert common['effective_batch_size']==1280 and common['epochs']==50
    device=configure_cuda()
    data=BASE/'cv_homework2/data'
    baseline=json.loads((BASE/'outputs/stage1_20261029_train_test/protocol.json').read_text(encoding='utf-8'))
    assert t.sha(BASE/'train_stage1.py')==baseline['source_sha256']
    assert t.sha(COURSE)==baseline['model_sha256']
    for name,expected in baseline['data_sha256'].items():assert t.sha(data/'cifar-100-python'/name)==expected
    assert platform.python_version()==baseline['environment']['python']
    assert torch.__version__==baseline['environment']['torch'] and torchvision.__version__==baseline['environment']['torchvision']
    assert np.__version__==baseline['environment']['numpy']
    raw_train=torchvision.datasets.CIFAR100(str(data),train=True)
    raw_test=torchvision.datasets.CIFAR100(str(data),train=False)
    assert len(raw_train)==50000 and len(raw_test)==10000
    train=t.DeviceCIFAR(raw_train,device,True);test=t.DeviceCIFAR(raw_test,device,False)
    del raw_train,raw_test
    if not (out/'protocol.json').exists():
        capacity=preflight(config,train,out,device)
        protocol={'config':config,'capacity':capacity,'data_protocol':baseline['data_protocol'],
                  'data_sha256':baseline['data_sha256'],'environment':baseline['environment'],
                  'source_sha256':t.sha(__file__),'shared_source_sha256':t.sha(BASE/'train_stage1.py'),
                  'model_source_sha256':t.sha(COURSE),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()}
        t.write_json(out/'protocol.json',protocol)
    else:
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['source_sha256']==t.sha(__file__) and protocol['config']==config,'Source/config changed; use a new output directory'
        capacity=protocol['capacity']
        _,total=torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(capacity['memory_budget_bytes']/total,device)
    if args.preflight_only:
        print(json.dumps({'preflight':'passed','micro_batch_sizes':capacity['micro_batch_sizes']}),flush=True)
        return 0
    histories,failed={},[]
    for run in config['stage4']:
        try:
            histories[run['id']]=run_one(run,common,capacity,train,test,out,device)
        except t.PauseRequested:
            t.write_json(out/'status.json',{'state':'paused','stage':4,'run':run['id'],'completed_runs':list(histories)})
            return 75
        except FloatingPointError:
            failed.append(run['id'])
        gc.collect();torch.cuda.empty_cache()
    if failed:
        t.write_json(out/'status.json',{'state':'incomplete','stage':4,'completed_runs':list(histories),'failed_runs':failed,'stage5_started':False})
        return 1
    selection=select_teacher_student(config,histories)
    t.write_json(out/'stage4_selection.json',selection)
    t.write_json(out/'status.json',{'state':'stage4_complete_awaiting_user','stage':4,'completed_runs':list(histories),'failed_runs':[],'stage5_started':False})
    print('STAGE4_COMPLETE_AWAITING_USER',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
