"""Measure Windows loader startup and steady-state throughput, using training images only."""
import argparse
import json
import time
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import train_stage1 as training


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda')
    tf=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    ef=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    data=training.BASE/'cv_homework2/data'
    train_all=torchvision.datasets.CIFAR100(str(data),train=True,transform=tf)
    val_all=torchvision.datasets.CIFAR100(str(data),train=True,transform=ef)
    ids,vids=list(range(16384)),list(range(5000))
    train=Subset(train_all,ids[:16384]);val=Subset(val_all,vids)
    results=[]
    for workers in [0,2,4]:
        training.seed_all(20261029)
        model=training.model_new(device)
        optimizer=training.optimizer_new(model,{'optimizer':'adam','lr':.001})
        loader=DataLoader(train,batch_size=256,shuffle=True,num_workers=workers,pin_memory=True,persistent_workers=workers>0,worker_init_fn=training.seed_worker,generator=torch.Generator().manual_seed(20261029))
        vloader=DataLoader(val,batch_size=256,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
        # Warm GPU kernels before comparing process startup and steady state.
        x=torch.randn(256,3,32,32,device=device);y=torch.zeros(256,dtype=torch.long,device=device)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(x),y).backward();optimizer.step()
        torch.cuda.synchronize()
        for epoch in [1,2]:
            start=time.perf_counter();total=0
            model.train()
            for x,y in loader:
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                z=model(x);loss=torch.nn.functional.cross_entropy(z,y)
                assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                # Match existing per-batch metric synchronization.
                loss.item();(z.argmax(1)==y).sum().item();total+=len(y)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            vstart=time.perf_counter();metrics=training.evaluate(model,vloader,device)[0];torch.cuda.synchronize()
            row={'workers':workers,'persistent':workers>0,'pass':epoch,'training_samples':total,'train_seconds':seconds,'train_images_per_second':total/seconds,'evaluation_seconds':time.perf_counter()-vstart,'evaluation_samples':metrics['count']}
            results.append(row);training.write_json(args.output,results);print(json.dumps(row),flush=True)
        del loader,vloader,model,optimizer
        torch.cuda.empty_cache()


if __name__=='__main__':main()
