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
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")  # Debugging

    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    return jsonify({"success": True, "message": "File uploaded successfully", "filename": filename, "file_path": file_path})

@app.route('/train', methods=['POST'])
def train_model():
    # Simulate training logic
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"error": "Uploaded file not found"}), 400

    # Simulate training process
    print(f"Training model with file: {uploaded_file}")
    return jsonify({"message": "Training completed successfully!"})

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    # Simulate evaluation logic
    return jsonify({
        "message": "Evaluation completed successfully!",
        "results": [
            {"metric": "Accuracy", "value": "92%"},
            {"metric": "Precision", "value": "90%"},
            {"metric": "Recall", "value": "88%"}
        ]
    })

@app.route('/visualize', methods=['POST'])
def visualize_model():
    # Simulate visualization data
    return jsonify({"message": "Visualization data generated!"})

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
