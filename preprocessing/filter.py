import json


with open("dataset1/1f_neg_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = []

for item in data:
    if item["label"] == 1:
        dataset.append(item)

with open("dataset1/dataset.json", 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
    