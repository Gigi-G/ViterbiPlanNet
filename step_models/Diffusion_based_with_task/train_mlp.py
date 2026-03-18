import glob
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloader.dataloader import PlanningDataset
from model.helpers import get_lr_schedule_with_warmup, Logger, AverageMeter
from utils.args import get_args


def cycle(dl):
    while True:
        for data in dl:
            yield data


class head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(head, self).__init__()
        middle_dim1 = input_dim // 3
        middle_dim2 = input_dim * 4
        self.fc1 = nn.Linear(input_dim, middle_dim1)
        self.fc2 = nn.Linear(middle_dim1, middle_dim2)
        self.fc3 = nn.Linear(middle_dim2, middle_dim1)
        self.fc4 = nn.Linear(middle_dim1, output_dim)

        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc2.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = torch.mean(x, dim=1)
        x = self.fc4(x)
        return x


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3, horizon=3, seed=None):
    torch.save(state, os.path.join(checkpoint_dir, f"epoch{epoch:04d}_T{horizon}_{seed}.pth.tar"))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, f"epoch{epoch - n_ckpt:04d}_T{horizon}_{seed}.pth.tar")
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def save_best_checkpoint(state, checkpoint_dir, old_epoch, epoch, horizon, seed):
    torch.save(state, os.path.join(checkpoint_dir, f"epoch{epoch:04d}_T{horizon}_{seed}.pth.tar"))
    if old_epoch > 0:
        oldest_ckpt = os.path.join(checkpoint_dir, f"epoch{old_epoch:04d}_T{horizon}_{seed}.pth.tar")
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def get_last_checkpoint(checkpoint_dir, args):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'epoch*_T{args.horizon}_{args.seed}.pth.tar'))
    return sorted(all_ckpt)[-1] if all_ckpt else ''


def train_epoch(train_loader, n_train_steps, model, scheduler, args, optimizer, if_calculate_acc):
    model.train()
    losses = AverageMeter()
    train_loader_ = cycle(train_loader)
    optimizer.zero_grad()

    for _ in range(n_train_steps):
        for _ in range(args.gradient_accumulate_every):
            batch_states, _, batch_tasks = next(train_loader_)

            if len(batch_states.shape) == 5:
                batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], batch_states.shape[2], -1)
            else:
                batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], -1)

            if batch_states.size(1) > 2:
                if len(batch_states.shape) == 4:
                    batch_states = torch.cat((batch_states[:, 0, 0:1, :], batch_states[:, -1, -1:, :]), dim=1)
                elif len(batch_states.shape) == 3:
                    batch_states = torch.cat((batch_states[:, 0:1, :], batch_states[:, -1:, :]), dim=1)

            bs, _, dim = batch_states.size()
            task_class = batch_tasks.view(-1).cuda()
            global_img_tensors = batch_states.cuda().contiguous().float()
            observations = torch.zeros(bs, 2, dim, device=global_img_tensors.device)
            observations[:, 0, :] = global_img_tensors[:, 0, :]
            observations[:, 1, :] = global_img_tensors[:, -1, :]

            task_logits = model(observations)
            loss = F.cross_entropy(task_logits, task_class)
            loss = loss / args.gradient_accumulate_every
            loss.backward()
            losses.update(loss.item(), bs)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if if_calculate_acc:
        with torch.no_grad():
            pred = task_logits.argmax(dim=-1)
            acc = torch.sum(pred.eq(task_class)) / bs * 100
        return torch.tensor(losses.avg), acc.detach()

    return torch.tensor(losses.avg)


def eval_epoch(val_loader, model):
    model.eval()
    losses = AverageMeter()
    acc_top1 = AverageMeter()

    for batch_states, _, batch_tasks in val_loader:
        if len(batch_states.shape) == 5:
            batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], batch_states.shape[2], -1)
        else:
            batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], -1)

        if batch_states.size(1) > 2:
            if len(batch_states.shape) == 4:
                batch_states = torch.cat((batch_states[:, 0, 0:1, :], batch_states[:, -1, -1:, :]), dim=1)
            elif len(batch_states.shape) == 3:
                batch_states = torch.cat((batch_states[:, 0:1, :], batch_states[:, -1:, :]), dim=1)

        global_img_tensors = batch_states.cuda().contiguous().float()
        bs, _, dim = global_img_tensors.size()
        task_class = batch_tasks.view(-1).cuda()

        with torch.no_grad():
            observations = torch.zeros(bs, 2, dim, device=global_img_tensors.device)
            observations[:, 0, :] = global_img_tensors[:, 0, :]
            observations[:, 1, :] = global_img_tensors[:, -1, :]
            task_logits = model(observations)
            loss = F.cross_entropy(task_logits, task_class)
            task_pred = task_logits.argmax(dim=-1)
            acc = torch.sum(task_pred.eq(task_class)) / bs * 100

        losses.update(loss.item(), bs)
        acc_top1.update(acc.item(), bs)

    return torch.tensor(losses.avg), torch.tensor(acc_top1.avg)


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    mlp_root = args.mlp_checkpoint_root if args.mlp_checkpoint_root else f"{args.checkpoint_root}_mlp"
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    args.log_root += '_mlp'

    if args.verbose:
        print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Selected dataset: {args.dataset}")
    print(f"Selected split: {args.split}")
    print(f"Selected horizon: {args.horizon}")

    train_dataset = PlanningDataset(
        video_list=args.train_json,
        horizon=args.horizon,
        num_action=args.action_dim,
        aug_range=args.aug_range,
        M=args.M,
        mode="train",
        PKG_labels=False
    )

    valid_dataset = PlanningDataset(
        video_list=args.valid_json,
        horizon=args.horizon,
        num_action=args.action_dim,
        aug_range=args.aug_range,
        M=args.M,
        mode="valid",
        PKG_labels=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                              pin_memory=args.pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                              drop_last=False)

    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)

    model = head(args.observation_dim, args.class_dim).cuda(args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = get_lr_schedule_with_warmup(optimizer, int(args.n_train_steps * args.epochs))

    checkpoint_dir = os.path.join(mlp_root, f'{args.dataset}_{args.split}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir, args)
    else:
        checkpoint_path = ''

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.rank}', weights_only=False)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        tb_logdir = checkpoint['tb_logdir']
        tb_logger = Logger(tb_logdir)
    else:
        time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logname = f"{args.log_root}_{time_pre}_{args.dataset}_{args.split}_T{args.horizon}_{args.seed}"
        tb_logdir = os.path.join(args.log_root, logname)
        os.makedirs(tb_logdir, exist_ok=True)
        tb_logger = Logger(tb_logdir)
        tb_logger.log_info(args)

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    max_eva = 0
    old_max_epoch = 0
    save_max = os.path.join(mlp_root, f'{args.dataset}_{args.split}_max_mlp')
    os.makedirs(save_max, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        if (epoch + 1) % 2 == 0 and args.evaluate:
            val_loss, val_acc = eval_epoch(valid_loader, model)
            logs = OrderedDict([
                ('Val/EpochLoss', val_loss.item()),
                ('Val/EpochAcc@1', val_acc.item()),
            ])
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, epoch + 1)

            if val_acc.item() >= max_eva:
                save_best_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'tb_logdir': tb_logdir,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    save_max,
                    old_max_epoch,
                    epoch + 1,
                    args.horizon,
                    args.seed,
                )
                max_eva = val_acc.item()
                old_max_epoch = epoch + 1

        if (epoch + 1) % 2 == 0:
            train_loss, train_acc = train_epoch(train_loader, args.n_train_steps, model, scheduler, args, optimizer, True)
            logs = OrderedDict([
                ('Train/EpochLoss', train_loss.item()),
                ('Train/EpochAcc@1', train_acc.item()),
            ])
        else:
            train_loss = train_epoch(train_loader, args.n_train_steps, model, scheduler, args, optimizer, False)
            logs = OrderedDict([('Train/EpochLoss', train_loss.item())])

        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'tb_logdir': tb_logdir,
                    'scheduler': scheduler.state_dict(),
                },
                checkpoint_dir,
                epoch + 1,
                horizon=args.horizon,
                seed=args.seed,
            )


if __name__ == '__main__':
    main()
