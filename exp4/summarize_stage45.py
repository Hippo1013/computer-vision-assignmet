"""Stage-four network and optimizer comparisons, without starting distillation."""
import json
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--normal-batch',action='store_true')
    args=parser.parse_args()
    root=BASE/'outputs'/('stage4_batch_20261029' if args.normal_batch else 'stage45_20261029')
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    rows=[];histories={}
    paths=[(BASE/'outputs/stage1_20261029_train_test','sgd_step','resnet10','sgd',1280),
           (BASE/'outputs/adam_extra_lr_20261029','adam_step','resnet10','adam',1280)]
    paths += [(root,r['id'],r['architecture'],r['optimizer'],protocol['capacity']['micro_batch_sizes'][r['architecture']]) for r in protocol['config']['stage4']]
    for folder,name,architecture,optimizer,micro in paths:
        f=folder/name/'history.json'
        if not f.exists():continue
        h=json.loads(f.read_text(encoding='utf-8'))
        if not h:continue
        peak=max(h,key=lambda r:r['test']['accuracy'])
        label=architecture+' '+optimizer
        rows.append({'id':name,'architecture':architecture,'optimizer':optimizer,'epochs':len(h),
                     'parameters':{'resnet10':4949412,'resnet18':11220132,'resnet50':23705252}[architecture],
                     'effective_batch_size':micro if args.normal_batch and architecture!='resnet10' else 1280,'micro_batch_size':micro,
                     'updates_per_epoch':math.ceil(50000/(micro if args.normal_batch and architecture!='resnet10' else 1280)),
                     'last_test_accuracy':h[-1]['test']['accuracy'],'peak_test_accuracy':peak['test']['accuracy'],
                     'peak_epoch':peak['epoch'],'train_minutes':sum(r['train_seconds'] for r in h)/60,
                     'train_test_minutes':sum(r['epoch_seconds'] for r in h)/60})
        histories[label]=h
    (root/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# 第四步网络比较结果','','| 网络 | 优化器 | 已完成轮数 | 参数量（百万） | 实际小批次 | 末轮测试准确率 | 最高测试准确率 | 训练分钟 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['architecture']} | {r['optimizer']} | {r['epochs']} | {r['parameters']/1e6:.3f} | {r['micro_batch_size']} | {100*r['last_test_accuracy']:.2f}% | {100*r['peak_test_accuracy']:.2f}% | {r['train_minutes']:.2f} |")
    note='ResNet18 batch=640每轮79次更新；ResNet50 batch=160每轮313次更新，均每batch更新一次，无梯度累积。ResNet10复用已有batch1280、每轮40次更新的结果。批大小及更新次数不同，不能将差异完全归因于深度。' if args.normal_batch else '各组有效批大小1280，18/50采用分块梯度累积，BN统计与直接大批量不同，不能将差异完全归因于深度。'
    lines.extend(['',note+'各组50轮，完整50000训练/10000测试，FP32。训练分钟不含测试与检查点写盘。第五步尚未启动。'])
    (root/'第四步结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for optimizer in ['sgd','adam']:
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
        for architecture in ['resnet10','resnet18','resnet50']:
            label=architecture+' '+optimizer
            if label not in histories:continue
            h=histories[label]
            for ax,key,ylabel in [(axes[0],'train_loss','Training loss'),(axes[1],'accuracy','Test accuracy (%)'),(axes[2],'lr','Learning rate')]:
                values=[100*r['test']['accuracy'] if key=='accuracy' else r[key] for r in h]
                ax.plot([r['epoch'] for r in h],values,label=label)
                ax.set(xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2)
        axes[2].set_yscale('log');axes[0].legend(fontsize=8)
        fig.savefig(root/('networks_'+optimizer+'.png'),dpi=180);plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
