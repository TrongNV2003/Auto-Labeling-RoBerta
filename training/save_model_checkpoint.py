from safetensors.torch import load_file
import torch
from modeling import LabelingModel

state_dict = load_file("/content/Labeling/labeling/bkai-embedding-encoder/checkpoint-450/model.safetensors")

MODEL = "hiieu/halong_embedding"
base_model = AutoModel.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = LabelingModel(base_model, "mean")
model.load_state_dict(state_dict)

save_dir = "embedding-encoder"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)