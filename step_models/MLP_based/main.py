import torch
import os
import time
import numpy as np
import json

from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.step_model import StepModel
from models.utils import AverageMeter
from tensorboardX import SummaryWriter 
from tools.parser import create_parser
from dataset.dataloader import PlanningDataset

def eval(
        args,
        data_loader,
        model,
        logger,
        state_prompt_features,
        e=0,
        device=torch.device("cuda"),
        writer=None,
        is_train=False
    ):
    # losses
    losses_state  = AverageMeter()
    losses_task   = AverageMeter()

    state_acc = AverageMeter()
    task_acc = AverageMeter()
    first_action_acc = AverageMeter()
    last_action_acc = AverageMeter()
    
    with torch.no_grad():
        for i, (batch_states, batch_actions, batch_tasks) in enumerate(data_loader):
            '''
            batch_states:  (batch_size, time_horizon, 2, embedding_dim)
            batch_actions: (batch_size, time_horizon)
            batch_prompts: (batch_size, 2*time_horizon, num_prompts, embedding_dim)
            '''

            batch_size, _ = batch_actions.shape
            
            if len(batch_states.shape) == 5:
                batch_states = batch_states.reshape(batch_size, batch_states.shape[1], batch_states.shape[2], -1)
            else:
                batch_states = batch_states.reshape(batch_size, batch_states.shape[1], -1)
                
            # Pad the tensor if its sequence length is less than the max trajectory length
            if batch_states.shape[1] < args.max_traj_len:
                pad_len = int(args.max_traj_len - batch_states.shape[1])

                if batch_states.dim() == 4:
                    last_element = batch_states[:, -1:, :, :]
                    pad_tensor = last_element.repeat(1, pad_len, 1, 1)
                else: 
                    last_element = batch_states[:, -1:, :]
                    pad_tensor = last_element.repeat(1, pad_len, 1)

                batch_states = torch.cat([batch_states, pad_tensor], dim=1)
                
                last_action = batch_actions[:, -1:]
                pad_tensor = last_action.repeat(1, pad_len)
                batch_actions = torch.cat([batch_actions, pad_tensor], dim=1)

            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)

            outputs, labels, losses = model(
                visual_features = batch_states,
                state_prompt_features = state_prompt_features,
                actions = batch_actions,
                tasks = batch_tasks
            )

            losses_state.update(losses["state_encode"].item(), batch_size)
            losses_task.update(losses["task"].item(), batch_size)

            ## metrics for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())
            outputs["state_encode"] = outputs["state_encode"].reshape(batch_size, -1, outputs["state_encode"].shape[-1])
            outputs["state_encode"] = outputs["state_encode"].softmax(dim=-1).argmax(dim=-1).contiguous()
            labels["state"] = labels["state"].reshape(batch_size, -1)
            first_action_acc.update(first_action_accuracy(outputs["state_encode"].cpu().tolist(), labels["state"].cpu().tolist()))
            last_action_acc.update(last_action_accuracy(outputs["state_encode"].cpu().tolist(), labels["state"].cpu().tolist()))

            # metrics for task prediction
            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item(), batch_size)

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tFirst Action Acc: {:.2f}% Last Action Acc: {:.2f}%"\
                    .format(first_action_acc.avg, last_action_acc.avg))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%"\
                    .format(losses_task.avg, task_acc.avg))

        if is_train:
            writer.add_scalar('valid_loss/state', losses_state.avg, e+1)
            writer.add_scalar('valid_loss/task', losses_task.avg, e+1)
            
            writer.add_scalar('valid_state/acc', state_acc.avg, e+1)            
            writer.add_scalar('valid_task/acc', task_acc.avg, e+1)
        
    return state_acc.avg, first_action_acc.avg, last_action_acc.avg, task_acc.avg


def evaluate(args):
    log_file_path = os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len}_log_eval.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/crosstask_state_prompt_features.npy')
    
    elif args.dataset == "crosstask_105":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/crosstask_105_state_prompt_features.npy')
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/coin_state_prompt_features.npy')

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/niv_state_prompt_features.npy')
        
    elif args.dataset == "egoper":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/egoper_state_prompt_features.npy')
        
    logger.info("Loading valid data...")
    valid_dataset = PlanningDataset(
        video_list=args.valid_json,
        horizon=args.max_traj_len_test,
        num_action=args.num_action,
        aug_range=args.aug_range,
        M=args.M,
        mode="valid",
        PKG_labels=False
    )
    
    logger.info("Testing set volumn: {}".format(len(valid_dataset)))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = StepModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        args=args
    ).to(device)
    
    model_path = os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len}_model_best.pth")
    model.load_state_dict(torch.load(model_path))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params}")
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
    model.eval()
    
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()

    valid_state_acc, first_action_acc, last_action_acc, valid_task_acc = eval(
        args,
        valid_loader, 
        model,
        logger, 
        state_prompt_features, 
        -1,
        device
    )
    
    results = {
        "state_acc": valid_state_acc,
        "first_action_acc": first_action_acc,
        "last_action_acc": last_action_acc,
        "task_acc": valid_task_acc
    }
    
    json_path = os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len}_eval_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {json_path}")

def train(args):
    logger_path = "logs/{}_{}_len{}_{}".format(
                    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                    args.model_name, 
                    args.max_traj_len,
                    args.seed)
    if args.last_epoch > -1:
        logger_path += "_last{}".format(args.last_epoch)
    os.makedirs(logger_path)
    log_file_path = os.path.join(logger_path, "log.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    validate_interval = 1
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/crosstask_state_prompt_features.npy')
    
    elif args.dataset == "crosstask_105":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/crosstask_105_state_prompt_features.npy')
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/coin_state_prompt_features.npy')

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/niv_state_prompt_features.npy')
        
    elif args.dataset == "egoper":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'../../data/state_description_features/egoper_state_prompt_features.npy')
        
    logger.info("Loading training data...")
    train_dataset = PlanningDataset(
        video_list=args.train_json,
        horizon=args.max_traj_len,
        num_action=args.num_action,
        aug_range=args.aug_range,
        M=args.M,
        mode="train",
        PKG_labels=False
    )
    
    logger.info("Loading valid data...")
    valid_dataset = PlanningDataset(
        video_list=args.valid_json,
        horizon=args.max_traj_len_test,
        num_action=args.num_action,
        aug_range=args.aug_range,
        M=args.M,
        mode="valid",
        PKG_labels=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))

    writer = SummaryWriter(logger_path)

    model = StepModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        args=args
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
        ],
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size, 
        gamma=args.lr_decay, 
        last_epoch=-1
    )

    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()

    max_state_acc = 0

    for e in range(0, args.epochs):
        model.train()
        # losses
        losses_state  = AverageMeter()
        losses_task = AverageMeter()

        state_acc = AverageMeter()
        task_acc = AverageMeter()

        for i, (batch_states, batch_actions, batch_tasks) in enumerate(train_loader):
            batch_size, _ = batch_actions.shape
            
            if len(batch_states.shape) == 5:
                batch_states = batch_states.reshape(batch_size, batch_states.shape[1], batch_states.shape[2], -1)
            else:
                batch_states = batch_states.reshape(batch_size, batch_states.shape[1], -1)
            
            optimizer.zero_grad()
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)
            
            outputs, labels, losses = model(
                visual_features=batch_states,
                state_prompt_features=state_prompt_features,
                actions=batch_actions,
                tasks=batch_tasks
            )
            
            total_loss = losses["state_encode"] + losses["task"]
            total_loss.backward()
            optimizer.step()
            
            losses_state.update(losses["state_encode"].item())
            losses_task.update(losses["task"].item())

            # Compute accuracy for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item())

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%".format(losses_task.avg, task_acc.avg))

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/lr', lr, e+1)

        writer.add_scalar('train_loss/state', losses_state.avg, e+1)
        writer.add_scalar('train_state/acc', state_acc.avg, e+1)
        writer.add_scalar('train_loss/task', losses_task.avg, e+1)
        writer.add_scalar('train_task/acc', task_acc.avg, e+1)

        if args.last_epoch < 0 or e < args.last_epoch:
            scheduler.step()

        ## validation
        if (e+1) % validate_interval == 0:
            model.eval()
            valid_state_acc, _, _, _ = eval(args, 
                      valid_loader, 
                      model, 
                      logger, 
                      state_prompt_features, 
                      e, 
                      device,
                      writer=writer, 
                      is_train=True)
            
            torch.save(
                model.state_dict(), 
                os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_last.pth"  
                )
            )
            
            if valid_state_acc > max_state_acc:
                max_state_acc = valid_state_acc
                log_save_path = os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_best.pth"  
                )
                checkpoint_save_path = os.path.join(
                        args.saved_path, 
                        args.dataset + "_" + str(args.seed),
                        f"T{args.max_traj_len}_model_best.pth"
                    )
                torch.save(model.state_dict(), checkpoint_save_path)        
                os.system(f"cp {checkpoint_save_path} {log_save_path}")
            

if __name__ == "__main__":
    args = create_parser()
    
    os.makedirs(os.path.join(args.saved_path, args.dataset + "_" + str(args.seed)), exist_ok=True)

    if args.eval:
        evaluate(args)
    else:
        train(args)
