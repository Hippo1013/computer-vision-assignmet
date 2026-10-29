"""Generate local stage-1 tables and figures from actual experiment records."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    root=args.output
    config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))['config']
    rows=[]
    for run in config['runs']:
        folder=root/run['id']
        if not (folder/'history.json').exists():
            continue
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
        best=max(history,key=lambda x:x['validation']['accuracy'])
        test=json.loads((folder/'test.json').read_text(encoding='utf-8'))['metrics']['accuracy'] if (folder/'test.json').exists() else None
        rows.append({'id':run['id'],'epochs':len(history),'best_epoch':best['epoch'],
                     'validation_accuracy':best['validation']['accuracy'],'test_accuracy':test,
                     'train_minutes':sum(x['train_seconds'] for x in history)/60,
                     'train_validation_minutes':sum(x['epoch_seconds'] for x in history)/60})
    (root/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# 前三项阶段结果','', '| 配置 | 轮数 | 最佳验证轮次 | 验证准确率 | 测试准确率 | 训练分钟 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        test='待评估' if row['test_accuracy'] is None else f"{100*row['test_accuracy']:.2f}%"
        lines.append(f"| {row['id']} | {row['epochs']} | {row['best_epoch']} | {100*row['validation_accuracy']:.2f}% | {test} | {row['train_minutes']:.2f} |")
    final=root/'final_full_train/test.json'
    if final.exists():
        obj=json.loads(final.read_text(encoding='utf-8'))
        lines.extend(['',f"最终全训练集重训：{obj['selected_config']}，末轮测试准确率 {100*obj['metrics']['accuracy']:.2f}%。"])
    lines.extend(['','本表按验证集最佳轮次报告对照组测试结果；全训练集重训采用末轮结果。训练耗时包含训练数据加载、前后向与参数更新，不含验证、检查点写盘及首次环境准备。仅一个随机种子，结果限于本次参数范围。'])
    (root/'阶段结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    groups={'sgd_learning_rates':['sgd_lr0001','sgd_lr001','sgd_lr01'],
            'adam_learning_rates':['adam_lr00001','adam_lr0001','adam_lr001'],
            'fixed_vs_step':['sgd_lr01','sgd_step']}
    for name,ids in groups.items():
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
        for run in ids:
            path=root/run/'history.json'
            if not path.exists():continue
            h=json.loads(path.read_text(encoding='utf-8'))
            for ax,key,label in [(axes[0],'train_loss','Training loss'),(axes[1],'val_accuracy','Validation accuracy (%)'),(axes[2],'lr','Learning rate')]:
                values=[100*x['validation']['accuracy'] if key=='val_accuracy' else x[key] for x in h]
                ax.plot([x['epoch'] for x in h],values,label=run)
                ax.set(xlabel='Epoch',ylabel=label)
                ax.grid(alpha=.2)
        axes[2].set_yscale('log')
        axes[0].legend(fontsize=7)
        fig.savefig(root/(name+'.png'),dpi=180)
        plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
