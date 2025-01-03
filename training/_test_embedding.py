import json
import torch
from transformers import AutoTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

torch.cuda.empty_cache()

class EmbeddingSimilarityLabeler:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.label_embeddings = {}

    def add_label(self, label, description):
        inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.label_embeddings[label] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        text_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        similarities = {label: cosine_similarity(text_embedding, emb)[0][0] for label, emb in self.label_embeddings.items()}

        top_label = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:1]
        return top_label[0] if top_label else None


# Khởi tạo model
MODEL = "embedding-encoder-model"
labeler = EmbeddingSimilarityLabeler(MODEL)

# Đọc và thêm nhãn từ file
def add_label_file(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    for label_item in label_data:
        label_intent = label_item["label_intent"]
        description = label_item["description"]
        labeler.add_label(label_intent, description)

# Dự đoán nhãn cho dữ liệu khách hàng
def predict_label(data_file, output_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        labeled_text = item.get("labeled_text", "").strip()
        if not labeled_text:
            continue

        top_label = labeler.predict(labeled_text)
        if top_label:
            item['label_intent'] = top_label[0]
            item['similarity_score'] = float(top_label[1])
        else:
            item['label_intent'] = "Không có nhãn phù hợp"
            item['similarity_score'] = 0.0

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def eval(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    for item in data:
        label_intent = item.get("labeled_intent", "").strip()
        true_label = item.get("label_intent", "").strip()

        if label_intent == true_label:
            correct += 1

    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == "__main__":
    label_file = 'intent_dataset/label.json'  # File chứa nhãn
    data_file = 'intent_dataset/eval.json'     # File dữ liệu cần gán nhãn
    output_file = 'intent_dataset/updated_data.json'  # File kết quả

    # Thêm nhãn từ file
    add_label_file(label_file)

    # Gán nhãn cho dữ liệu
    predict_label(data_file, output_file)

    eval(output_file)