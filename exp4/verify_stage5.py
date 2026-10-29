"""Audit the four completed students and optionally replay their eight checkpoints."""
import argparse
import json
import math
import pickle
from pathlib import Path
from verify_stage1 import read, digest, text_digest_matches

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/stage5_distillation_20261030'
STAGE4 = BASE / 'outputs/stage4_batch_20261029'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    protocol = read(OUT / 'protocol.json')
    config = protocol['config']
    common = config['common']
    assert read(OUT / 'status.json')['state'] == 'complete'
    assert read(OUT / 'runner_status.json')['state'] == 'complete'
    assert common['batch_size'] == 640 and common['epochs'] == 50 and common['seed'] == 20261029
    assert common['update_rule'] == 'one_optimizer_step_per_batch'
    assert common['precision'] == 'FP32'
    for path, key in [('train_stage5.py', 'source_sha256'), ('train_stage45.py', 'model_factory_source_sha256'),
                      ('train_stage1.py', 'shared_source_sha256'), ('cv_homework2/resnet-cifar100.py', 'model_source_sha256')]:
        assert text_digest_matches(BASE / path, protocol[key])
    assert digest(STAGE4 / 'protocol.json') == protocol['stage4_protocol_sha256']
    data = BASE / 'cv_homework2/data/cifar-100-python'
    for name, expected in protocol['data_sha256'].items():
        assert digest(data / name) == expected
    with (data / 'test').open('rb') as f:
        targets = pickle.load(f, encoding='latin1')['fine_labels']
    assert len(targets) == 10000
    for role in ['teacher', 'student_baseline']:
        ref = config[role]
        assert digest(STAGE4 / ref['run'] / 'last.pt') == ref['checkpoint_sha256']
    assert config['teacher']['run'] == 'resnet50_adam' and config['teacher']['checkpoint'] == 'last.pt'
    assert config['student_baseline']['run'] == 'resnet18_adam' and config['student_baseline']['checkpoint'] == 'last.pt'
    assert common['temperature'] == 2 and common['soft_weight'] == .5 and common['logits_weight'] == .1
    assert common['weight_decay'] == .0001 and common['adam_betas'] == [.9, .999] and common['adam_eps'] == 1e-8
    assert len(config['runs']) == 4
    initial, histories = {}, {}
    results = {'passed': True, 'checkpoint_replay': args.replay, 'runs': {}}
    for run in config['runs']:
        name = run['id']
        folder = OUT / name
        effective = read(folder / 'config.json')
        assert all(effective[k] == v for k, v in common.items())
        assert all(effective[k] == v for k, v in run.items())
        assert effective['training_count'] == 50000 and effective['test_count'] == 10000
        assert effective['prior_training_epochs'] == (50 if run['initialization'] == 'finetune' else 0)
        assert effective['initial_optimizer_state'] == 'fresh Adam; no inherited moments or steps'
        init = run['initialization']
        if init in initial:
            assert initial[init] == effective['initial_state_sha256']
        initial[init] = effective['initial_state_sha256']
        if init == 'random':
            assert initial[init] == protocol['random_initial_state_sha256']
        assert run['lr'] == (.001 if init == 'random' else .0001)
        assert run['optimizer'] == 'adam' and run['milestones'] == [30, 40]
        h = read(folder / 'history.json')
        histories[name] = h
        assert [r['epoch'] for r in h] == list(range(1, 51))
        weight = .5 if run['method'] == 'soft' else .1
        for row in h:
            assert row['training_count'] == 50000 and row['optimizer_updates'] == 79
            assert row['teacher_frozen_verified']
            assert math.isclose(row['lr'], run['lr'] * .1 ** sum(row['epoch'] > m for m in [30, 40]), rel_tol=1e-12)
            assert row['test']['count'] == 10000 and 0 <= row['test']['accuracy'] <= 1
            assert 0 <= row['train_accuracy'] <= 1
            assert all(math.isfinite(row[k]) for k in ['train_loss', 'train_ce_loss', 'train_distill_loss', 'train_weighted_distill_loss'])
            assert math.isclose(row['train_weighted_distill_loss'], weight * row['train_distill_loss'], abs_tol=1e-12)
            assert math.isclose(row['train_loss'], row['train_ce_loss'] + row['train_weighted_distill_loss'], rel_tol=2e-6, abs_tol=1e-6)
            assert row['epoch_seconds'] >= row['train_seconds'] > 0
        results['runs'][name] = {}
        for tag, row in [('last', h[-1]), ('best', max(h, key=lambda r: r['test']['accuracy']))]:
            record = read(folder / (tag + '_test.json'))
            assert record['epoch'] == row['epoch'] and record['metrics'] == row['test']
            assert record['targets'] == targets and len(record['predictions']) == 10000
            assert all(0 <= x < 100 for x in record['predictions'])
            accuracy = sum(x == y for x, y in zip(record['predictions'], targets)) / 10000
            assert accuracy == record['metrics']['accuracy']
            results['runs'][name][tag] = {'epoch': row['epoch'], 'accuracy': accuracy,
                                        'checkpoint_sha256': digest(folder / (tag + '.pt'))}
    assert initial['random'] != initial['finetune']
    results['paired_initializations_match'] = True
    if args.replay:
        import torch
        import torchvision
        import train_stage1 as t
        import train_stage45 as stage
        device = stage.configure_cuda()
        dataset = t.DeviceCIFAR(torchvision.datasets.CIFAR100(str(data.parent), train=False), device, False)
        teacher = stage.model_new('resnet50', device)
        state = t.load_checkpoint(STAGE4 / config['teacher']['run'] / 'last.pt', teacher)
        assert state['epoch'] == 50 and stage.state_digest(teacher) == protocol['teacher_state_sha256']
        del teacher, state
        for run in config['runs']:
            folder = OUT / run['id']
            t.seed_all(common['seed'])
            model = stage.model_new('resnet18', device)
            if run['initialization'] == 'finetune':
                state = t.load_checkpoint(STAGE4 / 'resnet18_adam/last.pt', model)
                del state
            assert stage.state_digest(model) == initial[run['initialization']]
            for tag in ['last', 'best']:
                state = t.load_checkpoint(folder / (tag + '.pt'), model)
                record = read(folder / (tag + '_test.json'))
                assert state['epoch'] == record['epoch']
                assert state['history'] == histories[run['id']][:state['epoch']]
                assert all(int(value['step']) == 79 * state['epoch'] for value in state['optimizer']['state'].values())
                metrics, pred, true = t.evaluate(model, dataset.batches(640, common['seed'], False), device)
                assert pred == record['predictions'] and true == record['targets']
                assert metrics['accuracy'] == record['metrics']['accuracy']
                assert math.isclose(metrics['loss'], record['metrics']['loss'], rel_tol=1e-7, abs_tol=1e-7)
                results['runs'][run['id']][tag]['replay_exact_predictions'] = True
                print(run['id'], tag, 'replay passed', flush=True)
            del model, state
    (OUT / ('verification_replay.json' if args.replay else 'verification_local.json')).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
