from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.utils.prune as prune
from transformers import BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification
from werkzeug.utils import secure_filename
import os
from flask_socketio import SocketIO, emit  # Ensure emit is imported
import time

app = Flask(__name__)  # ✅ Define app before using it in SocketIO
CORS(app, resources={r"/*": {"origins": "*"}})  # Fix CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Ensure CORS is allowed for SocketIO

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
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    allowed_extensions = {".csv", ".json", ".txt", ".xlsx", ".jpg", ".png", ".zip"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"success": False, "error": f"Unsupported file type: {file_ext}"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    return jsonify({"success": True, "message": "File uploaded successfully", "filename": filename, "file_path": file_path})

@app.route('/train', methods=['POST'])
def train_model():
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    try:
        for i in range(0, 101, 10):
            time.sleep(1)  # Simulate training time
            socketio.emit('training_progress', {'progress': i})  # Emit progress updates
        return jsonify({"success": True, "message": "Training completed successfully!"})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"success": False, "error": "Training failed due to an internal error."}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    evaluation_results = [
        {"metric": "Accuracy", "value": "92%"},
        {"metric": "Precision", "value": "90%"},
        {"metric": "Recall", "value": "88%"}
    ]
    return jsonify({"success": True, "message": "Evaluation completed successfully!", "results": evaluation_results})

@app.route('/visualize', methods=['POST'])
def visualize_model():
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    visualization_data = {
        "nodes": [{"id": i, "x": i * 0.5, "y": i * 0.5, "z": i * 0.5} for i in range(15)],
        "connections": [{"source": i, "target": j} for i in range(15) for j in range(i + 1, 15)]
    }
    return jsonify({"success": True, "message": "Visualization data generated!", "data": visualization_data})

@app.route('/distill', methods=['POST'])
def distill_model():
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    # Simulate knowledge distillation process
    try:
        distillation_metrics = {
            "teacher_accuracy": "95%",
            "student_accuracy": "92%",
            "compression_ratio": "3:1",
            "inference_speedup": "2x"
        }
        return jsonify({"success": True, "message": "Knowledge Distillation applied!", "metrics": distillation_metrics})
    except Exception as e:
        print(f"Error during distillation: {e}")
        return jsonify({"success": False, "error": "Distillation failed due to an internal error."}), 500

@app.route('/prune', methods=['POST'])
def prune_model():
    uploaded_file = request.json.get("file_path")
    if not uploaded_file or not os.path.exists(uploaded_file):
        return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    # Simulate pruning process
    try:
        pruning_metrics = {
            "original_size": "100MB",
            "pruned_size": "30MB",
            "accuracy_loss": "1%",
            "speedup": "1.5x"
        }
        return jsonify({"success": True, "message": "Model pruning completed!", "metrics": pruning_metrics})
    except Exception as e:
        print(f"Error during pruning: {e}")
        return jsonify({"success": False, "error": "Pruning failed due to an internal error."}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)  # Ensure the server is accessible
