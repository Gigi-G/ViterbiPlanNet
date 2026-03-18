DIFFUSION_CHECKPOINT_ROOT='checkpoints'
MLP_CHECKPOINT_ROOT='checkpoints_mlp'
BATCH_SIZE=256
EPOCHS_DIFFUSION=200
EPOCHS_MLP=200

if [[ "${DIFFUSION_CHECKPOINT_ROOT}" == "${MLP_CHECKPOINT_ROOT}" ]]; then
    echo "Error: diffusion and MLP checkpoint roots must be different."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size ${BATCH_SIZE} \
    --batch_size_val ${BATCH_SIZE} \
    --epochs ${EPOCHS_DIFFUSION} \
    --horizon 3 \
    --horizon_test 3 \
    --M 2 \
    --aug_range 0 \
    --dataset coin \
    --split base \
    --action_dim 778 \
    --class_dim 180 \
    --observation_dim 1536 \
    --train_json '../../dataset/COIN/coin_train_70_3.json' \
    --valid_json '../../dataset/COIN/coin_test_30_3.json' \
    --checkpoint_root "${DIFFUSION_CHECKPOINT_ROOT}" \
    --diffusion_checkpoint_root "${DIFFUSION_CHECKPOINT_ROOT}" \
    --saved_path './logs/' \
    --log_root 'logs' \
    --cudnn_benchmark 1 \
    --pin_memory \
    --resume \
    --evaluate \
    --seed 7 && \
CUDA_VISIBLE_DEVICES=0 python train_mlp.py \
    --batch_size ${BATCH_SIZE} \
    --batch_size_val ${BATCH_SIZE} \
    --epochs ${EPOCHS_MLP} \
    --horizon 3 \
    --horizon_test 3 \
    --M 2 \
    --aug_range 0 \
    --dataset coin \
    --split base \
    --action_dim 778 \
    --class_dim 180 \
    --observation_dim 1536 \
    --train_json '../../dataset/COIN/coin_train_70_3.json' \
    --valid_json '../../dataset/COIN/coin_test_30_3.json' \
    --checkpoint_root "${MLP_CHECKPOINT_ROOT}" \
    --mlp_checkpoint_root "${MLP_CHECKPOINT_ROOT}" \
    --log_root 'logs' \
    --cudnn_benchmark 1 \
    --pin_memory \
    --resume \
    --evaluate \
    --seed 7 && \
CUDA_VISIBLE_DEVICES=0 python predict_tasks.py \
    --batch_size ${BATCH_SIZE} \
    --batch_size_val ${BATCH_SIZE} \
    --horizon 3 \
    --horizon_test 3 \
    --M 2 \
    --aug_range 0 \
    --dataset coin \
    --split base \
    --action_dim 778 \
    --class_dim 180 \
    --observation_dim 1536 \
    --train_json '../../dataset/COIN/coin_train_70_3.json' \
    --valid_json '../../dataset/COIN/coin_test_30_3.json' \
    --checkpoint_root "${MLP_CHECKPOINT_ROOT}" \
    --mlp_checkpoint_root "${MLP_CHECKPOINT_ROOT}" \
    --log_root 'logs' \
    --cudnn_benchmark 1 \
    --pin_memory \
    --resume \
    --evaluate \
    --seed 7 && \
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --batch_size ${BATCH_SIZE} \
    --batch_size_val ${BATCH_SIZE} \
    --epochs ${EPOCHS_DIFFUSION} \
    --horizon 3 \
    --horizon_test 3 \
    --M 2 \
    --aug_range 0 \
    --dataset coin \
    --split base \
    --action_dim 778 \
    --class_dim 180 \
    --observation_dim 1536 \
    --train_json '../../dataset/COIN/coin_train_70_3.json' \
    --valid_json './data_lists/coin_pred_T3_7.json' \
    --checkpoint_root "${DIFFUSION_CHECKPOINT_ROOT}" \
    --diffusion_checkpoint_root "${DIFFUSION_CHECKPOINT_ROOT}" \
    --saved_path './logs/' \
    --log_root 'logs' \
    --cudnn_benchmark 1 \
    --pin_memory \
    --resume \
    --evaluate \
    --seed 7