import argparse

## add argument
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        default='model', type=str, 
                        help='model name')
    
    parser.add_argument('--mlp_ratio', 
                        default=2, type=int, metavar='MLP_RATIO',
                        help='mlp ratio in ff (default: 2)')
    
    parser.add_argument('--text_input_dim', 
                        default=768, type=int, metavar='DIM',
                        help='dimension (default: 768)')
    
    parser.add_argument('--img_input_dim', 
                        default=768, type=int, metavar='DIM',
                        help='dimension (default: 512)')
    
    parser.add_argument('--embed_dim', 
                        default=128, type=int, metavar='DIM',
                        help='dimension (default: 128)')
    
    parser.add_argument('--max_traj_len', 
                        default=3, type=int, metavar='MAXTRAJ',
                        help='max length (default: 54)')
    
    parser.add_argument('--max_traj_len_test', 
                        default=3, type=int, metavar='MAXTRAJTEST',
                        help='max length (default: 3)')
    
    parser.add_argument('--dataset', 
                        default='crosstask_howto100m', type=str, 
                        help='features')
    
    parser.add_argument('--num_action',
                        default=133, type=int,
                        help='number of action classes (crosstask: 133, coin: 778)')
    
    parser.add_argument('--num_tasks',
                        default=18, type=int,
                        help='number of tasks (crosstask: 18, coin: 778)')
    
    parser.add_argument('--epochs', 
                        default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--batch_size', '-b', 
                        default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    
    parser.add_argument('--dropout',
                        default=0.1, type=float,
                        help='dropout rate')
    
    parser.add_argument('--lr', '--learning-rate', 
                        default=0.01, type=float, metavar='LR', 
                        help='initial learning rate')
    
    parser.add_argument('--step_size', 
                        default=20, type=int, metavar='LRSteps', 
                        help='epochs to decay learning rate')
    
    parser.add_argument('--lr_decay',
                        default=0.65, type=float,
                        help='learning weight decay')
    
    parser.add_argument('--M',
                        default=1, type=int,
                        metavar='W', help='augmentation factor (default: 1)')
    
    parser.add_argument('--aug_range',
                        default=0, type=int,
                        metavar='W', help='augmentation range (default: 0)')
    
    parser.add_argument('--train_json', 
                        default='', type=str, 
                        help='train json file')
    
    parser.add_argument('--valid_json', 
                        default='', type=str, 
                        help='valid json file')

    parser.add_argument('--eval', 
                        action='store_true',
                        help='evaluation mode')
    
    parser.add_argument('--saved_path', 
                        default='./logs/', type=str, 
                        help='descriptions dir')

    parser.add_argument('--last_epoch',
                        default=-1, type=int,
                        help='last epoch for adjusting learning rate')

    parser.add_argument('--seed', 
                        default=7, type=int, metavar='M',
                        help='random seed')
    
    parser.add_argument('--type_of_model',
                         default='viterbi', type=str,
                         help='type of model (viterbi, viterbi_soft)')

    return parser.parse_args()
