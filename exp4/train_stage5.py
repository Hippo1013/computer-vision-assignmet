"""ResNet50-to-18 distillation: fresh and fine-tuned students, one step per batch.

All old training implementations and outputs are read-only dependencies.
"""
import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

import train_stage1 as t
import train_stage45 as stage

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/stage5_distillation_20261030'
STAGE4 = BASE / 'outputs/stage4_batch_20261029'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def distillation_update(student, optimizer, x, y, teacher, teacher_chunk, method):
    """Teacher sees the same augmented tensor; the entire student batch updates once."""
    optimizer.zero_grad(set_to_none=True)
    teacher_logits = stage.forward_teacher(teacher, x, teacher_chunk)
    logits = student(x)
    loss, ce, distill = stage.loss_terms(logits, y, teacher_logits, method)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite_loss')
    loss.backward()
    flags = [torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None]
    if not flags or not torch.stack(flags).all():
        raise FloatingPointError('nonfinite_gradient')
    assert all(p.grad is None for p in teacher.parameters())
    optimizer.step()
    return {'loss_sum': loss.item() * len(y), 'ce_sum': ce.item() * len(y),
            'distill_sum': distill.item() * len(y),
            'correct': (logits.argmax(1) == y).sum().item(), 'count': len(y)}


def initialization(student, mode, common, source):
    assert stage.state_digest(student) == source['capacity']['initial_state_sha256']['resnet18']
    if mode == 'finetune':
        state = t.load_checkpoint(STAGE4 / 'resnet18_adam/last.pt', student)
        assert state['epoch'] == 50
        del state
    else:
        assert mode == 'random'
    return stage.state_digest(student)


def run_one(config, common, source, train, test, teacher, teacher_digest, out, device):
    folder = out / config['id']
    folder.mkdir(exist_ok=True)
    if (folder / 'failure.json').exists():
        raise FloatingPointError('Previously recorded candidate failure; not retraining')
    t.seed_all(common['seed'])
    student = stage.model_new('resnet18', device)
    initial_digest = initialization(student, config['initialization'], common, source)
    optimizer = t.optimizer_new(student, config)
    assert len(optimizer.state) == 0
    assert all(p.requires_grad for p in student.parameters())
    effective = {**common, **config, 'architecture': 'resnet18', 'training_count': len(train),
                 'test_count': len(test), 'initial_state_sha256': initial_digest,
                 'initial_optimizer_state': 'fresh Adam; no inherited moments or steps',
                 'prior_training_epochs': 50 if config['initialization'] == 'finetune' else 0}
    cp = folder / 'config.json'
    if cp.exists():
        assert read(cp) == effective, 'Configuration changed'
    else:
        t.write_json(cp, effective)
    history, best, start = [], -1., 1
    last = folder / 'last.pt'
    if last.exists():
        state = t.load_checkpoint(last, student, optimizer)
        history, best, start = state['history'], state['best'], state['epoch'] + 1
        del state
    for epoch in range(start, common['epochs'] + 1):
        if (out / 'pause.request').exists():
            raise t.PauseRequested(config['id'])
        t.seed_all(common['seed'] + epoch)
        for group in optimizer.param_groups:
            group['lr'] = t.lr_at(config, epoch)
        assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
        student.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        sums = {'loss_sum': 0., 'ce_sum': 0., 'distill_sum': 0., 'correct': 0, 'count': 0}
        updates = 0
        try:
            for x, y in train.batches(common['batch_size'], common['seed'] + epoch, True):
                row = distillation_update(student, optimizer, x, y, teacher,
                                          common['teacher_forward_chunk'], config['method'])
                for key in sums:
                    sums[key] += row[key]
                updates += 1
        except FloatingPointError as error:
            t.write_json(folder / 'failure.json', {'epoch': epoch, 'reason': str(error)})
            raise
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - begin
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        assert sums['count'] == 50000 and updates == math.ceil(50000 / common['batch_size']) == 79
        metrics, pred, targets = t.evaluate(student, test.batches(common['batch_size'], common['seed'], False), device)
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - begin
        # Includes teacher parameters and BN buffers; any accidental teacher update is fatal.
        assert stage.state_digest(teacher) == teacher_digest
        assert all(p.grad is None for p in teacher.parameters())
        weight = .5 if config['method'] == 'soft' else .1
        row = {'epoch': epoch, 'lr': t.lr_at(config, epoch),
               'train_loss': sums['loss_sum'] / sums['count'],
               'train_ce_loss': sums['ce_sum'] / sums['count'],
               'train_distill_loss': sums['distill_sum'] / sums['count'],
               'train_weighted_distill_loss': weight * sums['distill_sum'] / sums['count'],
               'train_accuracy': sums['correct'] / sums['count'],
               'training_count': sums['count'], 'optimizer_updates': updates, 'test': metrics,
               'teacher_frozen_verified': True, 'train_seconds': train_seconds,
               'epoch_seconds': epoch_seconds, 'peak_allocated_bytes': peak_allocated,
               'peak_reserved_bytes': peak_reserved}
        history.append(row)
        improved = metrics['accuracy'] > best
        if improved:
            best = metrics['accuracy']
            t.save_checkpoint(folder / 'best.pt', student, optimizer, epoch, history, best)
            t.write_json(folder / 'best_test.json', {'epoch': epoch, 'metrics': metrics,
                         'checkpoint': 'best.pt', 'predictions': pred, 'targets': targets})
        t.save_checkpoint(last, student, optimizer, epoch, history, best)
        t.write_json(folder / 'last_test.json', {'epoch': epoch, 'metrics': metrics,
                     'checkpoint': 'last.pt', 'predictions': pred, 'targets': targets})
        t.write_json(folder / 'history.json', history)
        t.write_json(out / 'status.json', {'state': 'training', 'stage': 5, 'run': config['id'],
                     'epoch': epoch, 'epochs': common['epochs']})
        print(json.dumps({'run': config['id'], **row}), flush=True)
    # Repair metadata after an interruption between atomic checkpoint and JSON writes.
    t.write_json(folder / 'history.json', history)
    for tag, reference in [('last', history[-1]), ('best', max(history, key=lambda r: r['test']['accuracy']))]:
        record_path = folder / (tag + '_test.json')
        record = read(record_path) if record_path.exists() else None
        if record is None or record['epoch'] != reference['epoch'] or record['metrics'] != reference['test']:
            t.load_checkpoint(folder / (tag + '.pt'), student)
            metrics, pred, targets = t.evaluate(student, test.batches(common['batch_size'], common['seed'], False), device)
            assert metrics == reference['test']
            t.write_json(record_path, {'epoch': reference['epoch'], 'metrics': metrics,
                         'checkpoint': tag + '.pt', 'predictions': pred, 'targets': targets})
    return history


def validate_sources(config):
    source = read(STAGE4 / 'protocol.json')
    assert read(STAGE4 / 'status.json')['state'] == 'stage4_complete_awaiting_user'
    assert read(STAGE4 / 'verification_replay.json')['passed']
    assert t.sha(BASE / 'train_stage4_batch.py') == source['source_sha256']
    assert t.sha(BASE / 'train_stage45.py') == source['model_factory_source_sha256']
    assert t.sha(BASE / 'train_stage1.py') == source['shared_source_sha256']
    assert t.sha(stage.COURSE) == source['model_source_sha256']
    assert platform.python_version() == source['environment']['python']
    assert torch.__version__ == source['environment']['torch']
    assert torchvision.__version__ == source['environment']['torchvision']
    assert np.__version__ == source['environment']['numpy']
    selected = read(STAGE4 / 'stage4_selection.json')
    assert selected['teacher_run'] == 'resnet50_adam' and selected['student_baseline_run'] == 'resnet18_adam'
    assert selected['teacher_last_accuracy'] > selected['student_last_accuracy']
    for role, name in [('teacher', 'resnet50_adam'), ('student_baseline', 'resnet18_adam')]:
        ref = config[role]
        assert ref['run'] == name and ref['checkpoint'] == 'last.pt'
        assert t.sha(STAGE4 / name / 'last.pt') == ref['checkpoint_sha256']
        h = read(STAGE4 / name / 'history.json')
        assert len(h) == 50 and h[-1]['epoch'] == 50
    baseline = read(STAGE4 / 'resnet18_adam/config.json')
    common = config['common']
    for key in ['seed', 'epochs', 'precision', 'update_rule']:
        assert common[key] == baseline[key]
    assert common['batch_size'] == baseline['batch_size'] == 640
    assert common['teacher_forward_chunk'] == source['capacity']['teacher_forward_chunk'] == 160
    assert common['temperature'] == 2. and common['soft_weight'] == .5 and common['logits_weight'] == .1
    assert common['epochs'] == 50 and common['seed'] == 20261029
    assert common['weight_decay'] == .0001 and common['adam_betas'] == [.9, .999] and common['adam_eps'] == 1e-8
    assert common['student_trainable_layers'] == 'all'
    assert common['update_rule'] == 'one_optimizer_step_per_batch'
    assert len(config['runs']) == 4
    assert {(r['initialization'], r['method']) for r in config['runs']} == {
        ('random', 'soft'), ('random', 'logits'), ('finetune', 'soft'), ('finetune', 'logits')}
    assert len({r['id'] for r in config['runs']}) == 4
    for run in config['runs']:
        assert run['optimizer'] == 'adam' and run['milestones'] == [30, 40]
        assert run['lr'] == (baseline['lr'] if run['initialization'] == 'random' else .0001)
    for name, expected in source['data_sha256'].items():
        assert t.sha(BASE / 'cv_homework2/data/cifar-100-python' / name) == expected
    return source


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, default=BASE / 'stage5.json')
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'run.lock', 'a+b')
    if platform.system() == 'Windows':
        import msvcrt
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(args.config)
    source = validate_sources(config)
    common = config['common']
    device = stage.configure_cuda()
    raw_train = torchvision.datasets.CIFAR100(str(BASE / 'cv_homework2/data'), train=True)
    raw_test = torchvision.datasets.CIFAR100(str(BASE / 'cv_homework2/data'), train=False)
    assert len(raw_train) == 50000 and len(raw_test) == 10000
    train = t.DeviceCIFAR(raw_train, device, True)
    test = t.DeviceCIFAR(raw_test, device, False)
    del raw_train, raw_test
    free, total = torch.cuda.mem_get_info()
    budget = min(source['capacity']['memory_budget_bytes'], int(torch.cuda.memory_allocated() + free - 384 * 1024**2))
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    teacher = stage.model_new('resnet50', device)
    state = t.load_checkpoint(STAGE4 / config['teacher']['run'] / 'last.pt', teacher)
    assert state['epoch'] == 50
    del state
    teacher.eval().requires_grad_(False)
    teacher_digest = stage.state_digest(teacher)
    protocol = {'config': config, 'source_sha256': t.sha(__file__),
                'model_factory_source_sha256': t.sha(BASE / 'train_stage45.py'),
                'shared_source_sha256': t.sha(BASE / 'train_stage1.py'),
                'model_source_sha256': t.sha(stage.COURSE),
                'stage4_protocol_sha256': t.sha(STAGE4 / 'protocol.json'),
                'environment': source['environment'], 'data_protocol': source['data_protocol'],
                'data_sha256': source['data_sha256'], 'teacher_state_sha256': teacher_digest,
                'random_initial_state_sha256': source['capacity']['initial_state_sha256']['resnet18'],
                'memory_budget_bytes': budget, 'git_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=BASE.parent, text=True).strip(),
                'validation_mode': 'prior_capacity_and_formula_checks_plus_per_epoch_runtime_assertions',
                'teacher_input': 'same augmented FP32 tensor as student, sequential teacher chunks only',
                'distill_loss_units': {'soft': 'T^2 KL(teacher || student), class sum / image count',
                                       'logits': 'raw-logit MSE, mean over images and classes'}}
    pp = out / 'protocol.json'
    if pp.exists():
        previous = read(pp)
        for key in ['config', 'source_sha256', 'model_factory_source_sha256', 'shared_source_sha256',
                    'model_source_sha256', 'stage4_protocol_sha256', 'teacher_state_sha256', 'data_sha256', 'environment']:
            assert protocol[key] == previous[key], 'Source, data, teacher or protocol changed: ' + key
    else:
        t.write_json(pp, protocol)
    histories, failed = {}, []
    for run in config['runs']:
        try:
            histories[run['id']] = run_one(run, common, source, train, test, teacher, teacher_digest, out, device)
        except t.PauseRequested:
            t.write_json(out / 'status.json', {'state': 'paused', 'stage': 5, 'run': run['id'],
                         'completed_runs': list(histories), 'failed_runs': failed})
            return 75
        except FloatingPointError:
            failed.append(run['id'])
        gc.collect()
        torch.cuda.empty_cache()
    assert stage.state_digest(teacher) == teacher_digest
    t.write_json(out / 'status.json', {'state': 'complete' if not failed else 'incomplete', 'stage': 5,
                 'completed_runs': list(histories), 'failed_runs': failed, 'teacher_frozen_verified': True})
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
