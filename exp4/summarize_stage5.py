"""Summarize the four distillation runs beside the unchanged ResNet18 baseline."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/stage5_distillation_20261030'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = read(BASE / 'stage5.json')
    baseline = BASE / 'outputs/stage4_batch_20261029/resnet18_adam'
    entries = [('baseline', 'CE baseline, random start', baseline, .001, 0, 'ce')]
    for run in config['runs']:
        label = ('Fresh' if run['initialization'] == 'random' else 'Fine-tune') + (' / soft' if run['method'] == 'soft' else ' / logits MSE')
        entries.append((run['id'], label, OUT / run['id'], run['lr'],
                        50 if run['initialization'] == 'finetune' else 0, run['method']))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    rows, histories = [], {}
    for name, label, folder, lr, prior, method in entries:
        if not (folder / 'history.json').exists():
            continue
        h = read(folder / 'history.json')
        if not h:
            continue
        histories[name] = h
        peak = max(h, key=lambda r: r['test']['accuracy'])
        seconds = sum(r['train_seconds'] for r in h)
        rows.append({'id': name, 'label': label, 'method': method, 'prior_epochs': prior,
                     'initial_lr': lr, 'epochs': len(h), 'last_test_accuracy': h[-1]['test']['accuracy'],
                     'peak_test_accuracy': peak['test']['accuracy'], 'peak_epoch': peak['epoch'],
                     'last_test_loss': h[-1]['test']['loss'], 'train_minutes': seconds / 60,
                     'train_test_minutes': sum(r['epoch_seconds'] for r in h) / 60,
                     'train_images_per_second': 50000 * len(h) / seconds,
                     'last_train_ce_loss': h[-1]['train_ce_loss'],
                     'last_train_distill_loss': h[-1]['train_distill_loss']})
        for ax, key, ylabel in [(axes[0], 'accuracy', 'Test accuracy (%)'),
                                (axes[1], 'loss', 'Test cross-entropy loss'),
                                (axes[2], 'train_ce_loss', 'Training label cross-entropy')]:
            values = [100 * r['test'][key] if key == 'accuracy' else r['test'][key] if key == 'loss' else r[key] for r in h]
            ax.plot([r['epoch'] for r in h], values, label=label, linestyle='--' if prior else '-')
            ax.set(xlabel='Epoch in this 50-epoch run', ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0].axhline(.7578 * 100, color='gray', linestyle=':', linewidth=1, label='Teacher: 75.78%')
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / 'student_comparison.png', dpi=180)
    plt.close(fig)
    for init in ['random', 'finetune']:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
        for method, ax in [('soft', axes[0]), ('logits', axes[1])]:
            h = histories.get(init + '_' + method, [])
            for key, label in [('train_ce_loss', 'Label CE'), ('train_weighted_distill_loss', 'Weighted distillation'), ('train_loss', 'Total')]:
                ax.plot([r['epoch'] for r in h], [r[key] for r in h], label=label)
            ax.set(title='Soft target' if method == 'soft' else 'Logits MSE', xlabel='Epoch', ylabel='Training loss')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            axes[2].plot([r['epoch'] for r in h], [r['lr'] for r in h], label=method)
        axes[2].set(xlabel='Epoch', ylabel='Learning rate')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=.2)
        axes[2].legend(fontsize=8)
        fig.savefig(OUT / (init + '_loss_terms.png'), dpi=180)
        plt.close(fig)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# 第五步知识蒸馏结果', '', '| 学生训练方式 | 已有轮数 | 本组轮数 | 初始学习率 | 末轮准确率 | 峰值准确率（轮次） | 训练分钟 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['prior_epochs']} | {r['epochs']} | {r['initial_lr']:g} | {100*r['last_test_accuracy']:.2f}% | {100*r['peak_test_accuracy']:.2f}%（{r['peak_epoch']}） | {r['train_minutes']:.2f} |")
    lines += ['', '四组学生均为ResNet18、Adam、实际batch640，每批更新一次，每轮79次更新。老师固定为ResNet50 Adam的第50轮检查点，末轮准确率75.78%。无蒸馏基线复用第四步50轮结果。',
              '', '随机起点两组使用与无蒸馏基线相同的随机初始权重和0.001→0.0001→0.00001学习率；微调两组使用基线末轮权重和0.0001→0.00001→0.000001，再训练50轮，累计100轮。均在本次第30/40轮后衰减，Adam状态重新建立。',
              '', 'Soft target为CE+0.5×T²KL，T=2；logits为CE+0.1×原始logits MSE。图中蒸馏分量已乘各自权重，不把不同量纲的权重解释为相同指导强度。测试损失均为真实标签交叉熵。',
              '', '微调与从头训练同时改变训练起点、学习率及累计训练量；没有普通交叉熵续训50轮对照，不能把微调相对原基线的提升全部归因于蒸馏。全部结论限于本次单种子与参数设置，峰值仅是测试曲线峰值。',
              '', '![学生比较](student_comparison.png)', '', '![从头训练损失分量](random_loss_terms.png)', '', '![微调训练损失分量](finetune_loss_terms.png)']
    (OUT / '第五步结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
