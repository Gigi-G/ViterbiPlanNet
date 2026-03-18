import os
import glob
import time
import json
import random
from collections import OrderedDict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils
from model import diffusion, temporal
from model.helpers import get_lr_schedule_with_warmup, Logger
from utils.args import get_args
from utils import *
from dataloader.dataloader import PlanningDataset


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    if args.verbose:
        print(args)
        
    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Selected dataset: \t{args.dataset}")
    print(f"Selected split:   \t{args.split}")
    print(f"Selected horizon: \t{args.horizon}")
    print(f"Selected M:       \t{args.M}")
    
    # Setup directories
    os.makedirs(os.path.join(args.checkpoint_root, f'{args.dataset}_{args.split}'), exist_ok=True)
        
    print("Loading training data...")
    train_dataset = PlanningDataset(
        video_list=args.train_json, horizon=args.horizon, num_action=args.action_dim,
        aug_range=args.aug_range, M=args.M, mode="train", PKG_labels=False
    )
    
    print("Loading valid data...")
    valid_dataset = PlanningDataset(
        video_list=args.valid_json, horizon=args.horizon, num_action=args.action_dim,
        aug_range=args.aug_range, M=args.M, mode="valid", PKG_labels=False
    )
    
    # Dump original JSONs to target output paths
    with open(os.path.join(args.json_path_train, f'train_{args.dataset}_{args.split}_T{args.horizon}_{args.seed}.json'), "w") as f:
        json.dump(json.load(open(args.train_json, 'r')), f, indent=4)

    with open(os.path.join(args.json_path_val, f'test_{args.dataset}_{args.split}_T{args.horizon}_{args.seed}.json'), "w") as f:
        json.dump(json.load(open(args.valid_json, 'r')), f, indent=4)
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=args.pin_memory, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=1, drop_last=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    main_worker(args.gpu, args, train_loader, valid_loader)


def main_worker(gpu, args, train_loader, test_loader):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    # 1. Initialize Models
    temporal_model = temporal.TemporalUnet(
        args.action_dim + args.observation_dim, dim=256, dim_mults=(1, 2, 4)
    )

    diffusion_model = diffusion.GaussianDiffusion(
        temporal_model, args.horizon, args.observation_dim, args.action_dim, 
        args.n_diffusion_steps, loss_type='Weighted_MSE', clip_denoised=True
    )

    model = utils.Trainer(
        diffusion_model, train_loader, args.ema_decay, args.lr, 
        args.gradient_accumulate_every, args.step_start_ema, 
        args.update_ema_every, args.log_freq
    )

    # 2. Load pre-trained weights or map to GPU
    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        raise ValueError("No GPU is available, please set --gpu to a valid GPU id.")

    scheduler = get_lr_schedule_with_warmup(model.optimizer, int(args.n_train_steps * args.epochs))

    # 3. Handle Checkpoints and Logging
    checkpoint_dir = os.path.join(args.checkpoint_root, f'{args.dataset}_{args.split}')
    checkpoint_path = get_last_checkpoint(checkpoint_dir, args) if args.resume else None

    if checkpoint_path:
        # Resume from existing checkpoint
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.rank}')
        args.start_epoch = checkpoint["epoch"]
        model.model.load_state_dict(checkpoint["model"])
        model.ema_model.load_state_dict(checkpoint["ema"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        model.step = checkpoint["step"]
        scheduler.load_state_dict(checkpoint["scheduler"])
        
        tb_logdir = checkpoint["tb_logdir"]
        tb_logger = Logger(tb_logdir)
    else:
        # Initialize fresh logging directory
        time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logname = f"{args.log_root}_{time_pre}_{args.dataset}_{args.split}_T{args.horizon}_{args.seed}"
        tb_logdir = os.path.join(args.log_root, logname)
        os.makedirs(tb_logdir, exist_ok=True)
        
        tb_logger = Logger(tb_logdir)
        tb_logger.log_info(args)

    if args.cudnn_benchmark:
        cudnn.benchmark = True
    
    # 4. Training Loop setup
    max_eva = 0
    old_max_epoch = 0
    save_max = os.path.join(args.checkpoint_root, f'{args.dataset}_{args.split}_max')
    os.makedirs(save_max, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch : {epoch}')
        
        # Calculate full training metrics every 10 epochs
        calculate_metrics = (epoch + 1) % 10 == 0
        
        if calculate_metrics:
            losses, _, _, _, _, _, acc_a0, acc_aT = model.train(args.n_train_steps, True, args, scheduler)
            
            acc_average = (acc_a0.item() + acc_aT.item()) / 2
            
            if args.rank == 0:
                logs = OrderedDict([
                    ('Train/EpochLoss', losses.item()), ('Train/EpochAccAvg', acc_average),
                    ('Train/acc_a0', acc_a0.item()), ('Train/acc_aT', acc_aT.item())
                ])
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)
                tb_logger.flush()

                print('--- Train results ---')
                for k, v in logs.items(): print(f"{k.split('/')[-1]}: {v:.4f}")
                print('---------------------')
        else:
            losses = model.train(args.n_train_steps, False, args, scheduler).cuda()
            
            if args.rank == 0:
                print('LRs:', [p['lr'] for p in model.optimizer.param_groups])
                tb_logger.log_scalar(losses.item(), 'Train/EpochLoss', epoch + 1)
                tb_logger.flush()

        # 5. Validation Loop
        if ((epoch + 1) % 5 == 0) and args.evaluate:
            losses, _, _, _, _, _, acc_a0, acc_aT = validate(test_loader, model.ema_model, args)

            acc_average_val = (acc_a0.item() + acc_aT.item()) / 2

            if args.rank == 0:
                logs = OrderedDict([
                    ('Val/EpochLoss', losses.item()), ('Val/EpochAccAvg', acc_average_val),
                    ('Val/acc_a0', acc_a0.item()), ('Val/acc_aT', acc_aT.item())
                ])
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)
                tb_logger.flush()
                
                print('--- Validation results ---')
                for k, v in logs.items(): print(f"{k.split('/')[-1]}: {v:.4f}")
                print(f"Max Step Acc: {max_eva:.4f}")
                print('--------------------------')

            # Save best model
            if acc_average_val > max_eva:
                state_dict = {
                    "epoch": epoch + 1, "model": model.model.state_dict(), "ema": model.ema_model.state_dict(),
                    "optimizer": model.optimizer.state_dict(), "step": model.step, "tb_logdir": tb_logdir, 
                    "scheduler": scheduler.state_dict()
                }
                save_best_checkpoint(state_dict, save_max, old_max_epoch, epoch + 1, args.horizon, args.seed)
                max_eva = acc_average_val
                old_max_epoch = epoch + 1

        # 6. Periodic Saving
        if (epoch + 1) % args.save_freq == 0 and args.rank == 0:
            state_dict = {
                "epoch": epoch + 1, "model": model.model.state_dict(), "ema": model.ema_model.state_dict(),
                "optimizer": model.optimizer.state_dict(), "step": model.step, "tb_logdir": tb_logdir, 
                "scheduler": scheduler.state_dict()
            }
            save_rolling_checkpoint(state_dict, checkpoint_dir, epoch + 1, horizon=args.horizon, seed=args.seed)

def save_rolling_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3, horizon=3, seed=None):
    """Saves the current checkpoint and deletes the one 'n_ckpt' steps ago to save space."""
    torch.save(state, os.path.join(checkpoint_dir, f"epoch{epoch:04d}_T{horizon}_{seed}.pth.tar"))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, f"epoch{epoch-n_ckpt:04d}_T{horizon}_{seed}.pth.tar")
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)
            

def save_best_checkpoint(state, checkpoint_dir, old_epoch, epoch, horizon, seed):
    """Saves the best performing checkpoint and deletes the previous best."""
    torch.save(state, os.path.join(checkpoint_dir, f"epoch{epoch:04d}_T{horizon}_{seed}.pth.tar"))
    if old_epoch > 0:
        oldest_ckpt = os.path.join(checkpoint_dir, f"epoch{old_epoch:04d}_T{horizon}_{seed}.pth.tar")
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def get_last_checkpoint(checkpoint_dir, args):
    """Fetches the latest checkpoint path based on horizon and seed."""
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'epoch*_T{args.horizon}_{args.seed}.pth.tar'))
    return sorted(all_ckpt)[-1] if all_ckpt else ''


if __name__ == "__main__":
    main()