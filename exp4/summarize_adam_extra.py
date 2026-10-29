"""Compare the original Adam 0.001 run with the two fixed-rate supplements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path(__file__).resolve().parent/'outputs'
    out=base/'adam_extra_lr_20261029';out.mkdir(parents=True,exist_ok=True)
    configs=[(base/'stage1_20261029_train_test','adam_lr0001',.001),
             (out,'adam_lr001',.01),(out,'adam_lr01',.1)]
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    rows=[]
    for root,name,lr in configs:
        f=root/name/'history.json'
        if not f.exists():continue
        h=json.loads(f.read_text(encoding='utf-8'))
        if not h:continue
        peak=max(h,key=lambda r:r['test']['accuracy'])
        rows.append({'id':name,'lr':lr,'epochs':len(h),'last_test_accuracy':h[-1]['test']['accuracy'],
                     'peak_test_accuracy':peak['test']['accuracy'],'peak_epoch':peak['epoch'],
                     'train_minutes':sum(r['train_seconds'] for r in h)/60})
        for ax,key,label in [(axes[0],'train_loss','Training loss'),(axes[1],'test_accuracy','Test accuracy (%)'),(axes[2],'lr','Learning rate')]:
            values=[100*r['test']['accuracy'] if key=='test_accuracy' else r[key] for r in h]
            ax.plot([r['epoch'] for r in h],values,label=f'Adam, lr={lr:g}')
            ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].set_yscale('log')
    fig.savefig(out/'adam_learning_rates.png',dpi=180);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# Adam学习率补测结果','','| 固定学习率 | 已完成轮数 | 末轮测试准确率 | 最高测试准确率 | 最高轮次 |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['lr']:g} | {r['epochs']} | {100*r['last_test_accuracy']:.2f}% | {100*r['peak_test_accuracy']:.2f}% | {r['peak_epoch']} |")
    lines.extend(['','0.001复用原始50轮结果。补测与原始实验使用相同ResNet10、初始化与增强种子、batch=1280、FP32、50000训练/10000测试及50轮设置。仅固定学习率变化；不包含动态Adam实验。','', '![Adam固定学习率比较](adam_learning_rates.png)'])
    (out/'补测结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
