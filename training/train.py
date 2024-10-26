import argparse
import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
from modeling import LabelingModel
from data_loader import QGDataset, QGDataCollator
from training.trainer import ATrainer
from save_model_checkpoint import save_model_checkpoint

torch.autograd.set_detect_anomaly(True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer

def get_model(checkpoint: str, device: str, tokenizer: AutoTokenizer):
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, config=config, device_map=device)
    return model


parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_name_or_path", type=str, default="hiieu/halong_embedding")
parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--save_checkpoint", type=str, default="./model-checkpoint")
parser.add_argument("--save_model", type=str, default="model-checkpoint/checkpoint-450/model.safetensors")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--train_file", type=str, default="dataset1/dataset.json")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModel.from_pretrained(args.pretrained_model_name_or_path, device_map=device, torch_dtype="auto")
    
    model.embeddings.requires_grad_(False)

    gpu_count = torch.cuda.device_count()

    train_set = QGDataset(
        json_file=args.train_file
    )

    trainer  = ATrainer(
        model=LabelingModel(model, pooling_type="mean"),
        train_dataset=train_set,
        eval_dataset=None,
        loss_kwargs={
            "tau": 20
        },
        tokenizer=tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            # gradient_accumulation_steps=args.gradient_accumulation_steps,
            # gradient_checkpointing=bool(args.gradient_checkpointing),
            # warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            # bf16=bool(args.bf16),
            # fp16=bool(args.fp16),
            logging_steps=args.logging_steps,
            save_strategy="steps",
            # save_safetensors=True,
            eval_steps=None,
            save_steps=args.save_steps,
            output_dir=args.save_checkpoint,
            save_total_limit=5,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if gpu_count > 1 else None,
            logging_dir=args.log_dir,
        ),
        data_collator=QGDataCollator(tokenizer, args.max_length),
        save_model=save_model_checkpoint(args.save_model)
    )
    trainer.train()
