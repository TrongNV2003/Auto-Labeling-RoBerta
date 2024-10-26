from safetensors.torch import load_file
import torch
from modeling import LabelingModel
from transformers import AutoModel, AutoTokenizer

# class save_model_checkpoint:
#     def __init__(self, safetensors_path: str):
#         self.safetensors_path = safetensors_path

#     def __call__(self):
#         state_dict = load_file(self.safetensors_path)

#         MODEL = "hiieu/halong_embedding"
#         base_model = AutoModel.from_pretrained(MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(MODEL)

#         model = LabelingModel(base_model, "mean")
#         model.load_state_dict(state_dict)

#         save_dir = "embedding-encoder-model"
#         model.save_pretrained(save_dir)
#         tokenizer.save_pretrained(save_dir)

state_dict = load_file("")

MODEL = "hiieu/halong_embedding"
base_model = AutoModel.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = LabelingModel(base_model, "mean")
model.load_state_dict(state_dict)

save_dir = "embedding-encoder-model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)