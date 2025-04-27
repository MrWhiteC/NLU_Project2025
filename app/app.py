from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import spacy
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from classModel import QACGBertForSequenceClassification
from datasets import load_from_disk

nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the Flask app
app = Flask(__name__)

# Load the model, tokenizer, and spaCy NER model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("ibm-research/CTI-BERT")
nlp = spacy.load("en_core_web_sm")
model = torch.load("models/full_model_edit.pth", map_location=device)
model.eval()

# Load the datasets
loaded_dataset = load_from_disk('dataset_edit/dataset_edit')
dataset = loaded_dataset
dataset["train"] = dataset["train"].rename_column("label", "labels")
dataset["validation"] = dataset["validation"].rename_column("label", "labels")
dataset["test"] = dataset["test"].rename_column("label", "labels")

train_data = load_from_disk("tokenized_data_edit/train")
val_data = load_from_disk("tokenized_data_edit/validation")
test_data = load_from_disk("tokenized_data_edit/test")

# Ensure consistent column names
train_data = train_data.rename_column("label", "labels")
val_data = val_data.rename_column("label", "labels")
test_data = test_data.rename_column("label", "labels")

# Extract unique labels (MITRE techniques) from both train and validation datasets
labels = list(set(dataset['train']['labels']).union(set(dataset['validation']['labels'])))  # Extract unique labels
label_map = {label: i for i, label in enumerate(labels)}  # Create a label map
label_map_rev = {v: k for k, v in label_map.items()}  # Reverse label map for predictions

# Extract unique aspects (entities)
unique_aspects = list(set(aspects for aspects in dataset['train']['entity']))  # Get unique aspects (entities)
context_map = {asp: i+1 for i, asp in enumerate(unique_aspects)}  # Map each unique aspect to an index (+1 to reserve 0 for padding)

# Prediction function (same as your previous one)
def predict_single(text, model, tokenizer, context_map, label_map_rev, device, max_len=64):
    model.eval()

    sentences = sent_tokenize(text)
    sentence_predictions = []

    for sentence in sentences:
        # Tokenize input
        inputs = tokenizer(sentence, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        seq_len = input_ids.size(1)

        # Entity extraction using spaCy
        doc = nlp(sentence)
        aspects = list(set([ent.text for ent in doc.ents if ent.text in context_map]))

        # Encode context
        context_ids = [context_map.get(a, 0) for a in aspects]
        if not context_ids:
            context_ids = [0]
        context_ids = torch.tensor([context_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                seq_lens=[seq_len],
                context_ids=context_ids,
            )

        logits = outputs
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = np.argmax(probs)
        pred_label = label_map_rev[pred_idx]

        # Get top 5 confidence scores
        top_indices = np.argsort(probs)[-5:][::-1]
        top_confidences = [(label_map_rev[i], f"{probs[i]*100:.2f}%") for i in top_indices]

        sentence_predictions.append((sentence, pred_label, top_confidences))

    return sentence_predictions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Get predictions for the input text
        sentence_predictions = predict_single(text, model, tokenizer, context_map, label_map_rev, device)

        # Render the predictions in a format suitable for HTML
        return render_template('index.html', text=text, predictions=sentence_predictions)

if __name__ == '__main__':
    app.run(debug=True)
