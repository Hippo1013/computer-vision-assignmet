"""Generate stage-1 tables and figures using test metrics recorded each epoch."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    config = json.loads((root/'protocol.json').read_text(encoding='utf-8'))['config']
    rows = []
    for run in config['runs']:
        path = root/run['id']/'history.json'
        if not path.exists():
            continue
        history = json.loads(path.read_text(encoding='utf-8'))
        if not history:
            continue
        peak = max(history, key=lambda x: x['test']['accuracy'])
        rows.append({'id': run['id'], 'epochs': len(history), 'peak_test_epoch': peak['epoch'],
                     'peak_test_accuracy': peak['test']['accuracy'],
                     'last_test_accuracy': history[-1]['test']['accuracy'],
                     'train_minutes': sum(x['train_seconds'] for x in history)/60,
                     'train_test_minutes': sum(x['epoch_seconds'] for x in history)/60})
    (root/'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# 前三项阶段结果', '',
             '| 配置 | 已完成轮数 | 末轮测试准确率 | 最高测试准确率 | 最高轮次 | 训练分钟 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['epochs']} | {100*r['last_test_accuracy']:.2f}% | {100*r['peak_test_accuracy']:.2f}% | {r['peak_test_epoch']} | {r['train_minutes']:.2f} |")
    lines.extend(['', '各组均使用官方全部50000张训练图，每轮以官方10000张测试图评估；不划验证集，不额外重训。末轮与最高测试准确率分开报告；最高值对应测试曲线选出的轮次，并非额外独立评估。训练时间不含测试和检查点写盘；另记录训练加测试时间。结果限于本次单随机种子和参数范围。'])
    (root/'阶段结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    groups = {'optimizer_comparison': ['sgd_lr0001', 'adam_lr0001'],
              'sgd_learning_rates': ['sgd_lr0001', 'sgd_lr001', 'sgd_lr01'],
              'fixed_vs_step': ['sgd_lr01', 'sgd_step']}
    labels = {'sgd_lr0001':'SGD, lr=0.001', 'adam_lr0001':'Adam, lr=0.001',
              'sgd_lr001':'SGD, lr=0.01', 'sgd_lr01':'SGD, lr=0.1',
              'sgd_step':'SGD, step 0.1 / 0.01 / 0.001'}
    for name, ids in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
        for run in ids:
            path = root/run/'history.json'
            if not path.exists():
                continue
            h = json.loads(path.read_text(encoding='utf-8'))
            for ax, key, label in [(axes[0], 'train_loss', 'Training loss'),
                                   (axes[1], 'test_accuracy', 'Test accuracy (%)'),
                                   (axes[2], 'lr', 'Learning rate')]:
                values = [100*x['test']['accuracy'] if key == 'test_accuracy' else x[key] for x in h]
                ax.plot([x['epoch'] for x in h], values, label=labels[run],
                        linestyle='--' if run=='adam_lr0001' else '-')
                ax.set(xlabel='Epoch', ylabel=label)
                ax.grid(alpha=.2)
        axes[2].set_yscale('log')
        if axes[0].lines:
            axes[0].legend(fontsize=7)
        fig.savefig(root/(name+'.png'), dpi=180)
        plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
