WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=":" \
torchrun training/train.py \
--pretrained_model_name_or_path hiieu/halong_embedding \
--max_length 256 \
--epochs 3 \
--learning_rate 2e-5 \
--save_steps 50 \
--batch_size 8 \
--train_file intent_dataset/train.json