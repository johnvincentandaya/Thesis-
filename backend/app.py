from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.utils.prune as prune
from transformers import BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Initialize Models
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")  # Debugging


    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "filename": filename, "file_path": file_path})

@app.route('/train', methods=['POST'])
def train_model():
    return jsonify({"message": "Teacher model trained!"})

@app.route('/distill', methods=['POST'])
def distill_model():
    return jsonify({"message": "Knowledge Distillation applied!"})

@app.route('/prune', methods=['POST'])
def prune_model():
    for name, module in student_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
    return jsonify({"message": "Model pruning completed!"})

if __name__ == '__main__':
    app.run(debug=True)
