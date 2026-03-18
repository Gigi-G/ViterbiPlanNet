import glob
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloader.dataloader import PlanningDataset
from utils.args import get_args
from train_mlp import head


def get_best_checkpoint(checkpoint_dir, args):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'epoch*_T{args.horizon}_{args.seed}.pth.tar'))
    all_ckpt = sorted(all_ckpt)
    return all_ckpt[-1] if all_ckpt else ''


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    mlp_root = args.mlp_checkpoint_root if args.mlp_checkpoint_root else f"{args.checkpoint_root}_mlp"
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.verbose:
        print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False

    valid_dataset = PlanningDataset(
        video_list=args.valid_json,
        horizon=args.horizon_test,
        num_action=args.action_dim,
        aug_range=args.aug_range,
        M=args.M,
        mode='valid',
        PKG_labels=False,
    )

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)

    model = head(args.observation_dim, args.class_dim).cuda(args.gpu)

    checkpoint_dir = os.path.join(mlp_root, f'{args.dataset}_{args.split}_max_mlp')
    checkpoint_path = get_best_checkpoint(checkpoint_dir, args)
    if not checkpoint_path:
        raise FileNotFoundError(f'No MLP checkpoint found in {checkpoint_dir}')

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.rank}', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    model.eval()

    predictions = []
    correct = 0

    for batch_states, _, batch_tasks in valid_loader:
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
            task_pred = torch.argmax(task_logits, dim=1)
            predictions.extend(task_pred.cpu().numpy().tolist())
            correct += torch.sum(task_pred == task_class).item()

    with open(args.valid_json, 'r') as f:
        video_info_dict = json.load(f)

    filtered = []
    for item in video_info_dict:
        if 'vid' in item['id']:
            vid = item['id']['vid']
        else:
            vid = item['id']['feature'].split('/')[-1].split('.')[0]
        if vid not in valid_loader.dataset.no_valid_videos:
            filtered.append(item)

    assert len(filtered) == len(predictions), 'Prediction count does not match filtered JSON entries.'

    for index, item in enumerate(filtered):
        item['id']['task_id'] = int(predictions[index])

    os.makedirs('./data_lists', exist_ok=True)
    output_json = f'./data_lists/{args.dataset}_pred_T{args.horizon_test}_{args.seed}.json'
    with open(output_json, 'w') as f:
        json.dump(filtered, f, indent=4)

    print(f'Predicted-task JSON saved to {output_json}')
    print('task acc:', correct / len(predictions))


if __name__ == '__main__':
    main()
