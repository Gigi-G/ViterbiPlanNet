import os
import random
import time
import json
import glob
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils
from model import diffusion, temporal
from utils.args import get_args
from utils import AverageMeter
from dataloader.dataloader import PlanningDataset

def calculate_accuracy(output, target, max_traj_len=0):
    """Calculates Top-1 accuracy, and the accuracy of the first (a0) and last (aT) steps."""
    with torch.no_grad():        
        # Get the predicted classes (argmax)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Calculate step-specific accuracy (first step and last step)
        correct_a = correct[:1].view(-1, max_traj_len)
        correct_a0 = correct_a[:, 0].float().mean().mul_(100.0)
        correct_aT = correct_a[:, -1].float().mean().mul_(100.0)

        return (correct_a0.item() + correct_aT.item()) / 2, correct_a0.item(), correct_aT.item()


def test(val_loader, model, args, mode):
    """Evaluates the model and saves predicted action sequences."""
    model.eval()
    
    # Trackers for our metrics
    acc_avg_meter = AverageMeter()
    A0_acc = AverageMeter()
    AT_acc = AverageMeter()
    final_pred_list = []
    
    # Determine the correct output path based on train/test mode
    horizon = args.horizon if mode == "train" else args.horizon_test
    file_final_list_test = os.path.join(
        args.steps_path, f'{mode}_list_{args.dataset}_{args.split}_T{horizon}_{args.seed}.json'
    )

    for batch_states, y, batch_tasks in val_loader:
        # Flatten spatial dimensions if they exist
        if len(batch_states.shape) == 5:
            batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], batch_states.shape[2], -1)
        else:
            batch_states = batch_states.reshape(batch_states.shape[0], batch_states.shape[1], -1)
            
        # Extract only the first and last states if sequence length is > 2
        if batch_states.size(1) > 2:
            if len(batch_states.shape) == 4:
                batch_states = torch.cat((batch_states[:, 0, 0:1, :], batch_states[:, -1, -1:, :]), dim=1)
            elif len(batch_states.shape) == 3:
                batch_states = torch.cat((batch_states[:, 0:1, :], batch_states[:, -1:, :]), dim=1)
                
        video_label = y.cuda()
        task_class = batch_tasks.view(-1).cuda()
        batch_size_current, T = video_label.size()
        global_img_tensors = batch_states.cuda().contiguous().float()

        # Build conditional inputs for the diffusion model
        cond = {}
        with torch.no_grad():
            cond[0] = global_img_tensors[:, 0, :]
            if T < args.horizon:
                for i in range(T, args.horizon):
                    cond[i] = global_img_tensors[:, -1, :]
            cond[T - 1] = global_img_tensors[:, -1, :] 

            task_onehot = torch.zeros((task_class.size(0), args.class_dim), device=global_img_tensors.device)
            task_onehot[torch.arange(0, len(task_class), device=global_img_tensors.device), task_class] = 1.
            task_class_ = task_onehot.unsqueeze(1).repeat(1, args.horizon, 1)
            cond['task'] = task_class_

            video_label_reshaped = video_label.view(-1)
            
            # Forward pass
            output = model(cond, if_jump=True)
            actions_pred = output[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
            argmax_index = torch.argmax(actions_pred, dim=-1) 
            
            # Store predictions for JSON output
            for i in range(len(video_label)):
                final_pred_list.append(argmax_index[i].tolist())

            # Reshape predictions for accuracy calculation
            actions_pred = actions_pred.view(-1, args.action_dim)
            if T < args.horizon:
                actions_pred = actions_pred.view(batch_size_current, args.horizon, -1)[:, :T, :].contiguous()
                actions_pred = actions_pred.view(-1, args.action_dim)  
                
            # Calculate metrics
            acc_avg, a0_acc, aT_acc = calculate_accuracy(actions_pred.cpu(), video_label_reshaped.cpu(), max_traj_len=T)

        # Update average meters
        acc_avg_meter.update(acc_avg, batch_size_current)
        A0_acc.update(a0_acc, batch_size_current)
        AT_acc.update(aT_acc, batch_size_current)
    
    # Save step predictions to JSON
    with open(file_final_list_test, 'w') as ou:
        json.dump(final_pred_list, ou)
        
    return acc_avg_meter.avg, A0_acc.avg, AT_acc.avg


def process_final_json(args, mode, dataset_obj):
    """Helper to merge step predictions back into the original JSON formatted data."""
    horizon = args.horizon if mode == "train" else args.horizon_test
    
    # Load original data
    json_path = args.json_path_train if mode == "train" else args.json_path_val
    orig_json_file = os.path.join(json_path, f'{mode}_{args.dataset}_{args.split}_T{horizon}_{args.seed}.json')
    with open(orig_json_file, 'r') as f:
        original_data = json.load(f)
    
    # Check if JSON contains 'vid'
    for item in original_data:
        if "vid" not in item["id"]:
            item["id"]["vid"] = item["id"]["feature"].split("/")[-1].split(".")[0]
    
    # Filter out invalid videos
    original_data = [item for item in original_data if item["id"]["vid"] not in dataset_obj.no_valid_videos]

    # Load predicted list
    steps_json = os.path.join(args.steps_path, f'{mode}_list_{args.dataset}_{args.split}_T{horizon}_{args.seed}.json')
    with open(steps_json, 'r') as f:
        large_list = json.load(f)

    # Inject predictions
    for i, item in enumerate(original_data):
        item["id"]["pred_list"] = large_list[i]

    # Save final output
    final_output = os.path.join(args.step_model_output, f'{mode}_steps_{args.dataset}_{args.split}_T{horizon}_{args.seed}.json')
    with open(final_output, 'w') as f:
        json.dump(original_data, f, indent=4)


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    diffusion_root = args.diffusion_checkpoint_root if args.diffusion_checkpoint_root else args.checkpoint_root
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    if args.verbose:
        print(args)
        
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
    
    # Setup directories
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(diffusion_root, f'{args.dataset}_{args.split}'), exist_ok=True)
        
    print("Loading training data...")
    train_dataset = PlanningDataset(
        video_list=args.train_json, horizon=args.horizon, num_action=args.action_dim,
        aug_range=args.aug_range, M=args.M, mode="train", PKG_labels=False
    )
    
    print("Loading valid data...")
    valid_dataset = PlanningDataset(
        video_list=args.valid_json, horizon=args.horizon_test, num_action=args.action_dim,
        aug_range=args.aug_range, M=args.M, mode="valid", PKG_labels=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    # Process Train Data
    print("Validating training data...")
    main_worker(args.gpu, args, train_loader, mode='train', diffusion_root=diffusion_root)
    process_final_json(args, mode='train', dataset_obj=train_dataset)
    
    # Process Validation Data
    print("Validating validation data...")
    main_worker(args.gpu, args, valid_loader, mode='test', diffusion_root=diffusion_root)
    process_final_json(args, mode='test', dataset_obj=valid_dataset)


def main_worker(gpu, args, test_loader, mode, diffusion_root):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    # Initialize Models
    temporal_model = temporal.TemporalUnet(
        args.class_dim + args.action_dim + args.observation_dim,
        dim=256,
        dim_mults=(1, 2, 4), 
    )

    diffusion_model = diffusion.GaussianDiffusion(
        temporal_model, args.horizon, args.observation_dim, args.action_dim, args.class_dim,
        args.n_diffusion_steps, loss_type='Weighted_MSE', clip_denoised=True
    )

    model = utils.Trainer(
        diffusion_model, None, args.ema_decay, args.lr, args.gradient_accumulate_every,
        args.step_start_ema, args.update_ema_every, args.log_freq
    )

    # Load Weights
    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path, weights_only=False)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.ema_model = torch.nn.DataParallel(model.ema_model).cuda()

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_dir = os.path.join(diffusion_root, f'{args.dataset}_{args.split}_max')
        checkpoint_path = get_last_checkpoint(checkpoint_dir, args)
        if checkpoint_path:
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.rank}', weights_only=False)
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"], strict=True)
            model.ema_model.load_state_dict(checkpoint["ema"], strict=True)
            model.step = checkpoint["step"]
        else:
            raise FileNotFoundError("Could not find checkpoint to resume from.")

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    time_start = time.time()

    # Run Evaluation
    acc_avg, acc_a0, acc_aT = test(test_loader, model.ema_model, args, mode)

    print(f'Time: {time.time() - time_start:.2f}s')
    print(f'Val/EpochAccAvg: {acc_avg:.4f}')
    print(f'Val/acc_a0: {acc_a0:.4f}')
    print(f'Val/acc_aT: {acc_aT:.4f}')
    
    # Save test results
    if mode == 'test':
        results = {
            "state_acc": acc_avg,
            "first_action_acc": acc_a0,
            "last_action_acc": acc_aT,
            "task_acc": 0
        }
        json_path = os.path.join(args.saved_path, f"{args.dataset}_{args.seed}", f"T{args.horizon}_eval_results.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {json_path}")


def get_last_checkpoint(checkpoint_dir, args):
    """Fetches the latest checkpoint path based on horizon and seed."""
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'epoch*_T{args.horizon}_{args.seed}.pth.tar'))
    return sorted(all_ckpt)[-1] if all_ckpt else ''


if __name__ == "__main__":
    main()