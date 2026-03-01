import torch
import os
import time
import numpy as np

from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.procedure_model import ProcedureModel
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
        transition_matrix,
        transition_matrix_torch,
        e=0,
        device=torch.device("cuda"),
        writer=None,
        is_train=False
    ):
    # losses
    losses_state  = AverageMeter()
    losses_action = AverageMeter()
    losses_task   = AverageMeter()

    # metrics for action
    action_acc1 = AverageMeter()
    action_acc5 = AverageMeter()
    action_sr   = AverageMeter()
    action_miou = AverageMeter()

    # metrics for viterbi
    viterbi_sr = AverageMeter()
    viterbi_acc1 = AverageMeter()
    viterbi_miou = AverageMeter()

    state_acc = AverageMeter()
    task_acc = AverageMeter()
    first_action_acc = AverageMeter()
    last_action_acc = AverageMeter()
    
    y_true = []
    y_pred = []
    
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

                # We check the number of dimensions of the tensor before repeating.
                if batch_states.dim() == 4:
                    # For a 4D tensor of shape (batch_size, seq_len, dim1, dim2)
                    # We must use a 4D repeat pattern.
                    last_element = batch_states[:, -1:, :, :] # Shape: (batch_size, 1, dim1, dim2)
                    pad_tensor = last_element.repeat(1, pad_len, 1, 1)
                else: # This handles the 3D case
                    # For a 3D tensor of shape (batch_size, seq_len, features)
                    # We use the original 3D repeat pattern.
                    last_element = batch_states[:, -1:, :] # Shape: (batch_size, 1, features)
                    pad_tensor = last_element.repeat(1, pad_len, 1)

                # Concatenate the original tensor with the padding tensor
                batch_states = torch.cat([batch_states, pad_tensor], dim=1)
                
                # Pad also the batch_actions
                last_action = batch_actions[:, -1:]
                pad_tensor = last_action.repeat(1, pad_len)
                batch_actions = torch.cat([batch_actions, pad_tensor], dim=1)

            ## compute loss
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)

            outputs, labels, losses = model(
                visual_features = batch_states,
                state_prompt_features = state_prompt_features,
                actions = batch_actions,
                transition_matrix_torch = transition_matrix_torch,
                transition_matrix = transition_matrix,
                tasks = batch_tasks,
                time_horz_test=args.max_traj_len_test if not is_train else None
            )

            losses_state.update(losses["state_encode"].item(), batch_size)
            losses_action.update(losses["action"].item(), batch_size)
            losses_task.update(losses["task"].item(), batch_size)

            ## metrics for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())
            outputs["state_encode"] = outputs["state_encode"].reshape(batch_size, -1, outputs["state_encode"].shape[-1])
            outputs["state_encode"] = outputs["state_encode"].softmax(dim=-1).argmax(dim=-1).contiguous()
            labels["state"] = labels["state"].reshape(batch_size, -1)
            first_action_acc.update(first_action_accuracy(outputs["state_encode"].cpu().tolist(), labels["state"].cpu().tolist()))
            last_action_acc.update(last_action_accuracy(outputs["state_encode"].cpu().tolist(), labels["state"].cpu().tolist()))

            ## computer accuracy for action prediction
            outputs["viterbi_logits"] = outputs["viterbi_logits"].reshape(batch_size, -1, outputs["action"].shape[-1])
            outputs["viterbi_logits"] = outputs["viterbi_logits"][:, :args.max_traj_len_test, :]
            labels["action"] = labels["action"].reshape(batch_size, -1)[:, :args.max_traj_len_test]
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["viterbi_logits"].contiguous().view(-1, outputs["viterbi_logits"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len_test) 
            action_acc1.update(acc1.item(), batch_size)
            action_acc5.update(acc5.item(), batch_size)
            action_sr.update(sr.item(), batch_size)
            action_miou.update(MIoU, batch_size)
            
            # metrics for task prediction
            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item(), batch_size)

            # metrics for viterbi decoding
            pred_viterbi = outputs["pred_viterbi"][:, 0:args.max_traj_len_test].cpu().numpy()
            labels_viterbi = labels["action"].reshape(batch_size, -1)[:, :args.max_traj_len_test].cpu().numpy().astype("int")
            sr_viterbi = success_rate(pred_viterbi, labels_viterbi, True)
            miou_viterbi = acc_iou(pred_viterbi, labels_viterbi, False).mean()
            acc_viterbi = mean_category_acc(pred_viterbi, labels_viterbi)
            viterbi_sr.update(sr_viterbi, batch_size)
            viterbi_acc1.update(acc_viterbi, batch_size)
            viterbi_miou.update(miou_viterbi, batch_size)
            
            y_pred.extend(pred_viterbi)
            y_true.extend(batch_actions.cpu().numpy().astype("int"))
        
        # Calculate the classic mIoU
        mean_intersection_over_union = miou(y_pred, y_true)

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tFirst Action Acc: {:.2f}% Last Action Acc: {:.2f}%"\
                    .format(first_action_acc.avg, last_action_acc.avg))
        logger.info("\tAction Loss: {:.2f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        logger.info("\tViterbi, SR: {:.2f}% Acc: {:.2f}% MIoU1: {:.2f}% MIoU2: {:.2f}%"\
                    .format(viterbi_sr.avg,
                            viterbi_acc1.avg,
                            viterbi_miou.avg,
                            mean_intersection_over_union))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%"\
                    .format(losses_task.avg, task_acc.avg))

        if is_train:
            writer.add_scalar('valid_loss/state', losses_state.avg, e+1)
            writer.add_scalar('valid_loss/action', losses_action.avg, e+1)
            writer.add_scalar('valid_loss/task', losses_task.avg, e+1)
            
            writer.add_scalar('valid_state/acc', state_acc.avg, e+1)

            writer.add_scalar('valid_action/sr', action_sr.avg, e+1)
            writer.add_scalar('valid_action/miou', action_miou.avg, e+1)
            writer.add_scalar('valid_action/acc1', action_acc1.avg, e+1)
            writer.add_scalar('valid_action/acc5', action_acc5.avg, e+1)

            writer.add_scalar('valid_action/viterbi_sr', viterbi_sr.avg, e+1)
            writer.add_scalar('valid_action/viterbi_miou', viterbi_miou.avg, e+1)
            writer.add_scalar('valid_action/viterbi_acc1', viterbi_acc1.avg, e+1)
            
            writer.add_scalar('valid_task/acc', task_acc.avg, e+1)
        
    if not is_train:
        predictions = {}
        print(len(y_true), len(y_pred))
        for i in range(len(y_true)):
            predictions[i] = {
                "pred": y_pred[i].tolist(),
                "true": y_true[i].tolist()
            }
        with open(os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len_test}_predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4)
        
        metrics_predictions = {
            "viterbi-DVL": {
                "SR": action_sr.avg,
                "mAcc": action_acc1.avg,
                "mIoU": action_miou.avg
            },
            "viterbi-DVL+VD": {
                "SR": viterbi_sr.avg,
                "mAcc": viterbi_acc1.avg,
                "mIoU": viterbi_miou.avg
            }
        }
        with open(os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len_test}_metrics_{args.type_of_model}.json"), "w") as f:
            json.dump(metrics_predictions, f, indent=4)

    return viterbi_sr.avg, viterbi_acc1.avg, viterbi_miou.avg, action_sr.avg, action_acc1.avg, action_miou.avg


def evaluate(args):
    log_file_path = os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len}_log_eval.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')
    
    elif args.dataset == "crosstask_105":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_105_state_prompt_features.npy')
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')
        
    
    logger.info("Loading training data...")
    train_dataset = PlanningDataset(
        video_list=args.train_json,
        horizon=args.max_traj_len_test,
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
    
    # This is the transition matrix used for Viterbi Decoding
    transition_matrix = train_dataset.transition_matrix
    np.save(os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len_test}_transition_matrix.npy"), transition_matrix)
    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))
    transition_matrix_torch = torch.from_numpy(transition_matrix).float().to(device)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
        args=args
    ).to(device)
    
    model_path = os.path.join(args.saved_path, args.dataset + "_" + str(args.seed), f"T{args.max_traj_len}_model_best_{args.type_of_model}.pth")
    model.load_state_dict(torch.load(model_path))
    # Print model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params}")
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
    model.eval()
    
    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()

    eval(
        args,
        valid_loader, 
        model,
        logger, 
        state_prompt_features, 
        transition_matrix, 
        transition_matrix_torch,
        -1,
        device
    )


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
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')
    
    elif args.dataset == "crosstask_105":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/crosstask_105_state_prompt_features.npy')
    
    elif args.dataset == "coin":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')

    elif args.dataset == "niv":
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')
        
    
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
    
    # This is the transition matrix used for Viterbi Decoding
    transition_matrix = train_dataset.transition_matrix

    transition_matrix_torch = torch.from_numpy(transition_matrix).float().to(device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))

    writer = SummaryWriter(logger_path)

    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
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

    max_SR_viterbi = 0
    avg_metrics_viterbi = 0
    max_SR_viterbi_soft = 0
    avg_metrics_viterbi = 0

    for e in range(0, args.epochs):
        model.train()
        # losses
        losses_state  = AverageMeter()
        losses_action = AverageMeter()
        losses_task = AverageMeter()

        # metrics for action
        action_acc1 = AverageMeter()
        action_acc5 = AverageMeter()
        action_sr   = AverageMeter()
        action_miou = AverageMeter()
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
                transition_matrix_torch=transition_matrix_torch,
                tasks=batch_tasks
            )
            
            # Compute loss for state encoding and action prediction
            total_loss = losses["state_encode"] +  losses["action"] + losses["task"]
            total_loss.backward()
            optimizer.step()
            
            losses_action.update(losses["action"].item())
            losses_state.update(losses["state_encode"].item())
            losses_task.update(losses["task"].item())

            # Compute accuracy for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            # Compute accuracy for action prediction using viterbi-soft logits
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["viterbi_logits"].contiguous().view(-1, outputs["viterbi_logits"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len) 
            action_acc1.update(acc1.item())
            action_acc5.update(acc5.item())
            action_sr.update(sr.item())
            action_miou.update(MIoU)
            
            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item())

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tAction Loss: {:.10f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%".format(losses_task.avg, task_acc.avg))

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/lr', lr, e+1)

        writer.add_scalar('train_loss/state', losses_state.avg, e+1)
        writer.add_scalar('train_loss/action', losses_action.avg, e+1)
        writer.add_scalar('train_state/acc', state_acc.avg, e+1)
        writer.add_scalar('train_loss/task', losses_task.avg, e+1)

        writer.add_scalar('train_action/sr', action_sr.avg, e+1)
        writer.add_scalar('train_action/miou', action_miou.avg, e+1)
        writer.add_scalar('train_action/acc1', action_acc1.avg, e+1)
        writer.add_scalar('train_action/acc5', action_acc5.avg, e+1)
        
        writer.add_scalar('train_task/acc', task_acc.avg, e+1)

        if args.last_epoch < 0 or e < args.last_epoch:
            scheduler.step()

        ## validation
        if (e+1) % validate_interval == 0:
            model.eval()
            viterbi_sr, viterbi_acc1, viterbi_miou, viterbi_soft_sr, viterbi_soft_acc1, viterbi_soft_miou = eval(args, 
                      valid_loader, 
                      model, 
                      logger, 
                      state_prompt_features, 
                      transition_matrix, 
                      transition_matrix_torch,
                      e, 
                      device,
                      writer=writer, 
                      is_train=True)
            
            # save the last model to logger path
            torch.save(
                model.state_dict(), 
                os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_last.pth"  
                )
            )
            # save the best viterbi model
            if viterbi_sr > max_SR_viterbi or (viterbi_sr == max_SR_viterbi and (viterbi_acc1 + viterbi_miou) / 2 > avg_metrics_viterbi):
                max_SR_viterbi = viterbi_sr
                avg_metrics_viterbi = (viterbi_acc1 + viterbi_miou) / 2
                log_save_path = os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_best_viterbi.pth"  
                )
                checkpoint_save_path = os.path.join(
                        args.saved_path, 
                        args.dataset + "_" + str(args.seed),
                        f"T{args.max_traj_len}_model_best_viterbi.pth"
                    )
                torch.save(model.state_dict(), checkpoint_save_path)        
                os.system(f"cp {checkpoint_save_path} {log_save_path}")
            
            # save the best viterbi-soft model
            if viterbi_soft_sr > max_SR_viterbi_soft or (viterbi_soft_sr == max_SR_viterbi_soft and (viterbi_soft_acc1 + viterbi_soft_miou) / 2 > avg_metrics_viterbi):
                max_SR_viterbi_soft = viterbi_soft_sr
                avg_metrics_viterbi = (viterbi_soft_acc1 + viterbi_soft_miou) / 2
                log_save_path = os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_best_viterbi_soft.pth"  
                )
                checkpoint_save_path = os.path.join(
                        args.saved_path, 
                        args.dataset + "_" + str(args.seed),
                        f"T{args.max_traj_len}_model_best_viterbi_soft.pth"
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
