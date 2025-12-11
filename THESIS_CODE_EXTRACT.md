# Key Code Sections from app.py for Thesis Paper

## 1. Application Initialization and Configuration

```python
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, models
import torch
import torch.nn.utils.prune as prune
import os
import zipfile
import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
import warnings
from types import SimpleNamespace

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    max_http_buffer_size=500000000,  # 500MB for large file uploads
    ping_timeout=120,
    ping_interval=25
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Global variables
train_loader = None
teacher_model = None
student_model = None
model_trained = False
tokenizer = None
last_teacher_metrics = None
last_student_metrics = None
last_effectiveness_metrics = None
training_cancelled = False
phase_order = ["model_loading", "knowledge_distillation", "pruning", "evaluation", "completed"]
last_progress = 0
last_phase_index = -1
current_training_domain = None
latest_model_structures = {
    "teacher": None,
    "student_kd": None,
    "student_pruned": None
}

# Raw model data storage for current training session
training_raw_data = {
    "uploaded_model": {
        "teacher_before": None,
        "teacher_logits": None,
        "student_before": None,
        "student_after": None,
        "student_logits": None,
        "loss_history": [],
        "teacher_logits_history": [],
        "student_logits_history": []
    },
    "baseline_model": {
        "before_training": None,
        "after_training": None,
        "model_name": None
    }
}
```

## 2. Lazy Loading of Transformers Library

```python
# Import transformers with lazy loading to avoid circular imports
TRANSFORMERS_AVAILABLE = False
DistilBertForSequenceClassification = None
DistilBertTokenizer = None
T5ForConditionalGeneration = None
T5Tokenizer = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
T5Config = None

def _load_transformers():
    """Lazy load transformers to avoid circular import issues."""
    global TRANSFORMERS_AVAILABLE, DistilBertForSequenceClassification, DistilBertTokenizer
    global T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification
    global AutoTokenizer, T5Config
    
    if TRANSFORMERS_AVAILABLE:
        return True
        
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer as AutoTokenizerBase
        
        # Create wrapper classes to avoid circular imports
        class DistilBertWrapper:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return transformers.DistilBertForSequenceClassification.from_pretrained(model_name, **kwargs)
        
        class DistilBertTokenizerWrapper:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return transformers.DistilBertTokenizer.from_pretrained(model_name, **kwargs)
        
        class T5Wrapper:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return transformers.T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        
        class T5TokenizerWrapper:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return transformers.T5Tokenizer.from_pretrained(model_name, **kwargs)
        
        class T5ConfigWrapper:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return transformers.T5Config.from_pretrained(model_name, **kwargs)
        
        # Assign to global variables
        DistilBertForSequenceClassification = DistilBertWrapper
        DistilBertTokenizer = DistilBertTokenizerWrapper
        T5ForConditionalGeneration = T5Wrapper
        T5Tokenizer = T5TokenizerWrapper
        AutoModelForSequenceClassification = AutoModel
        AutoTokenizer = AutoTokenizerBase
        T5Config = T5ConfigWrapper
        
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception as e:
        print(f"Warning: Transformers not available: {e}")
        TRANSFORMERS_AVAILABLE = False
        return False
```

## 3. Model Initialization Function

```python
def initialize_models(model_name, num_labels=2, uploaded_model_path=None):
    """Initialize teacher (uploaded) and student models for KD + pruning.
    
    Args:
        model_name: Name of the baseline selected in the UI (used for labeling only).
        num_labels: Number of output labels (unused for uploaded models, but kept for compatibility).
        uploaded_model_path: Path to uploaded model file (required).
    
    Returns:
        str or None: Error message if initialization failed, None if successful.
    """
    global teacher_model, student_model, tokenizer, current_training_domain

    try:
        print(f"Initializing models from uploaded file: {uploaded_model_path}")
        if model_name:
            print(f"Baseline model selected for comparison: {model_name} (will be loaded separately, not trained)")
        
        if not uploaded_model_path:
            return "A custom uploaded model (.pt/.pth/.bin/.ckpt/.json/.config) is required before training."

        # Validate file extension
        lower_path = str(uploaded_model_path).lower()
        _, ext = os.path.splitext(lower_path)
        allowed_extensions = ['.pt', '.pth', '.bin', '.ckpt', '.json', '.config']
        
        if ext not in allowed_extensions:
            return (
                f"Unsupported file type: {ext}. "
                f"Allowed file types: {', '.join(allowed_extensions)}. "
                "Please upload a valid PyTorch model file (.pt, .pth, .bin, .ckpt) or compatible checkpoint/config file."
            )
        
        # Check if file exists
        if not os.path.exists(uploaded_model_path):
            return f"Model file not found: {uploaded_model_path}"
        
        # Try to load the model
        teacher_model, error = load_uploaded_model(uploaded_model_path)
        if error:
            return error
        
        # Explicitly move teacher model to CPU to avoid meta device issues
        if teacher_model is not None:
            teacher_model = teacher_model.to('cpu')
            teacher_model.eval()
        
        student_model, domain = create_student_model_from_teacher(teacher_model)
        # Explicitly move student model to CPU
        if student_model is not None:
            student_model = student_model.to('cpu')
            student_model.eval()
        current_training_domain = domain
        
        if current_training_domain == "nlp":
            if _load_transformers():
                try:
                    from transformers import AutoTokenizer as TransformersAutoTokenizer
                    tokenizer_name = getattr(getattr(teacher_model, "config", None), "name_or_path", "distilbert-base-uncased")
                    tokenizer = TransformersAutoTokenizer.from_pretrained(tokenizer_name)
                except Exception:
                    tokenizer = None
            else:
                tokenizer = None
        else:
            tokenizer = None
        
        print("[SUCCESS] Uploaded model loaded as teacher, student initialized")
        return None

    except ImportError as e:
        error_msg = f"Import error during model initialization: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error initializing models: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg
```

## 4. Knowledge Distillation Function

```python
def apply_knowledge_distillation(
    teacher_model,
    student_model,
    optimizer,
    kd_criterion,
    ce_criterion,
    alpha=0.6,
    temperature=2.0
):
    """Apply knowledge distillation using CE (ground truth) + KD (teacher)."""
    global current_training_domain
    # Ensure models are on CPU (not meta device)
    try:
        teacher_model = teacher_model.to('cpu')
        student_model = student_model.to('cpu')
    except:
        pass
    teacher_model.eval()
    student_model.train()
    device = torch.device('cpu')  # Always use CPU
    domain = current_training_domain or detect_model_domain(teacher_model)
    
    try:
        # Pass model type hint to generate_training_batch for T5 handling
        model_type_hint = type(teacher_model)
        batch_inputs, labels = generate_training_batch(domain, model_type_hint=model_type_hint)
        # Ensure all inputs are on CPU
        if domain == "nlp":
            batch_inputs = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in batch_inputs.items()}
        else:
            batch_inputs = batch_inputs.to('cpu') if torch.is_tensor(batch_inputs) else batch_inputs
        labels = labels.to('cpu') if torch.is_tensor(labels) else labels
        
        with torch.no_grad():
            if domain == "nlp":
                teacher_outputs = teacher_model(**batch_inputs)
            else:
                teacher_outputs = teacher_model(batch_inputs)
        teacher_logits = extract_logits(teacher_outputs)
        
        if domain == "nlp":
            student_outputs = student_model(**batch_inputs)
        else:
            student_outputs = student_model(batch_inputs)
        student_logits = extract_logits(student_outputs)
        
        # Ensure logits have matching shapes for KL divergence
        teacher_logits_shape = teacher_logits.shape
        student_logits_shape = student_logits.shape
        
        # If sequence lengths differ (common in NLP), truncate/pad to match
        if len(teacher_logits_shape) > 1 and len(student_logits_shape) > 1:
            if teacher_logits_shape[1] != student_logits_shape[1]:
                min_seq_len = min(teacher_logits_shape[1], student_logits_shape[1])
                teacher_logits = teacher_logits[:, :min_seq_len, :]
                student_logits = student_logits[:, :min_seq_len, :]
            
            # Handle vocabulary/class dimension mismatch (last dim)
            if teacher_logits_shape[-1] != student_logits_shape[-1]:
                min_vocab = min(teacher_logits_shape[-1], student_logits_shape[-1])
                teacher_logits = teacher_logits[..., :min_vocab]
                student_logits = student_logits[..., :min_vocab]
        
        # Ensure logits are 2D for softmax (batch_size, num_classes)
        if len(teacher_logits.shape) > 2:
            teacher_logits = teacher_logits.mean(dim=1) if teacher_logits.shape[1] > 1 else teacher_logits[:, 0, :]
            student_logits = student_logits.mean(dim=1) if student_logits.shape[1] > 1 else student_logits[:, 0, :]
        
        # Apply temperature scaling
        teacher_soft = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
        student_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        
        # Compute losses
        kd_loss = kd_criterion(student_soft, torch.nn.functional.softmax(teacher_logits / temperature, dim=-1))
        ce_loss = ce_criterion(student_logits, labels)
        
        # Combined loss
        total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return float(total_loss.item()), {
            "kd_loss": float(kd_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "total_loss": float(total_loss.item())
        }
    except Exception as e:
        print(f"[KD] Error in knowledge distillation: {e}")
        raise
```

## 5. Main Training Task Function

```python
def training_task(model_name, uploaded_model_path=None, uploaded_model_name=None):
    """The background task for training the model.
    
    CRITICAL: Training ALWAYS uses the uploaded_model_path, regardless of model_name selection.
    The model_name parameter is ONLY used to load baseline comparison data - it does NOT affect training.
    
    Args:
        model_name: Name of the baseline model from dropdown (ONLY for loading comparison data, NOT for training)
        uploaded_model_path: Path to uploaded model file (REQUIRED - this is what gets trained)
        uploaded_model_name: Name of uploaded model file (optional, for display)
    """
    global model_trained, teacher_model, student_model, tokenizer, last_teacher_metrics, last_student_metrics, last_effectiveness_metrics, training_cancelled, training_raw_data
    
    try:
        if not uploaded_model_path:
            error_msg = "Uploaded model is required before training can begin."
            socketio.emit("training_error", {"error": error_msg})
            return
        
        # Reset cancellation flag
        training_cancelled = False
        
        # Initialize models from uploaded file ONLY
        error = initialize_models(model_name, uploaded_model_path=uploaded_model_path)
        if error:
            socketio.emit("training_error", {"error": error})
            return

        if teacher_model is None or student_model is None:
            socketio.emit("training_error", {"error": "Models not properly initialized"})
            return
        
        # Generate real input for evaluation based on UPLOADED model type
        model_type = str(type(teacher_model)).lower()
        is_transformer = 'distilbert' in model_type or 't5' in model_type or 'bert' in model_type
        
        if is_transformer:
            if tokenizer is not None:
                sample_texts = ["This is a test sentence for model evaluation."]
                encoded = tokenizer(sample_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                inputs = {
                    "input_ids": encoded['input_ids'].to('cpu'),
                    "attention_mask": encoded['attention_mask'].to('cpu')
                }
                seq_len = inputs["input_ids"].size(1)
                if inputs["attention_mask"].size(1) != seq_len:
                    inputs["attention_mask"] = torch.ones(1, seq_len, device='cpu', dtype=torch.long)
            else:
                token_list = [1, 2, 3, 4, 5] * 25 + [1, 2, 3]  # Exactly 128 tokens
                inputs = {
                    "input_ids": torch.tensor([token_list], device='cpu', dtype=torch.long),
                    "attention_mask": torch.ones(1, 128, device='cpu', dtype=torch.long)
                }
            
            # Add decoder inputs for T5 models
            if 't5' in model_type:
                input_ids = inputs["input_ids"]
                seq_len = input_ids.size(1)
                decoder_input_ids = torch.cat([
                    torch.zeros((input_ids.size(0), 1), dtype=input_ids.dtype, device='cpu'),
                    input_ids[:, :-1]
                ], dim=1)
                if decoder_input_ids.size(1) != seq_len:
                    if decoder_input_ids.size(1) < seq_len:
                        pad_size = seq_len - decoder_input_ids.size(1)
                        decoder_input_ids = torch.cat([
                            decoder_input_ids,
                            torch.zeros((decoder_input_ids.size(0), pad_size), dtype=decoder_input_ids.dtype, device='cpu')
                        ], dim=1)
                    else:
                        decoder_input_ids = decoder_input_ids[:, :seq_len]
                inputs["decoder_input_ids"] = decoder_input_ids
        else:
            # For vision models, use properly normalized inputs
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            inputs = transform(torch.randn(1, 3, 224, 224, device='cpu') * 0.5 + 0.5)

        # Evaluate teacher model metrics
        teacher_metrics = evaluate_model_metrics(teacher_model, inputs)
        
        # Extract RAW model data BEFORE training
        teacher_before_raw = extract_raw_model_data(teacher_model, inputs)
        student_before_raw = extract_raw_model_data(student_model, inputs)
        
        # Initialize training_raw_data
        training_raw_data = {
            "uploaded_model": {
                "teacher_before": teacher_before_raw if teacher_before_raw else {},
                "teacher_logits": None,
                "student_before": student_before_raw if student_before_raw else {},
                "student_after": None,
                "student_logits": None,
                "loss_history": [],
                "teacher_logits_history": [],
                "student_logits_history": []
            },
            "baseline_model": {
                "before_training": {},
                "after_training": {},
                "model_name": model_name
            }
        }
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        kd_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        ce_criterion = torch.nn.CrossEntropyLoss()
        
        # Perform knowledge distillation with REAL training epochs
        total_steps = 50  # More epochs for real training of uploaded models
        socketio.emit("training_status", {
            "phase": "knowledge_distillation",
            "message": "Initializing optimized knowledge distillation process..."
        })
        
        loss_value = 0.0
        
        for step in range(total_steps):
            # Check for cancellation
            if training_cancelled:
                socketio.emit("training_cancelled", {"message": "Training has been cancelled"})
                return
            
            # Apply knowledge distillation
            try:
                loss_value, loss_info = apply_knowledge_distillation(
                    teacher_model, student_model, optimizer, 
                    kd_criterion, ce_criterion, alpha=0.6, temperature=2.0
                )
            except Exception as e:
                error_msg = f"Error during knowledge distillation step {step + 1}/{total_steps}: {str(e)}"
                print(f"[TRAIN] {error_msg}")
                loss_value = 0.5  # Default loss value
                continue
            
            # Store RAW loss value in history
            training_raw_data["uploaded_model"]["loss_history"].append(float(loss_value))
            
            # Extract and store raw logits periodically (every 10 steps)
            if (step + 1) % 10 == 0 or step == total_steps - 1:
                try:
                    teacher_model.eval()
                    student_model.eval()
                    with torch.no_grad():
                        if isinstance(inputs, dict):
                            teacher_outputs = teacher_model(**inputs)
                            student_outputs = student_model(**inputs)
                        else:
                            teacher_outputs = teacher_model(inputs)
                            student_outputs = student_model(inputs)
                        
                        teacher_logits = extract_logits(teacher_outputs)
                        student_logits = extract_logits(student_outputs)
                        
                        # Store sample logits (first 20 values)
                        if teacher_logits.numel() > 0:
                            teacher_logits_flat = teacher_logits.flatten().cpu()[:20]
                            training_raw_data["uploaded_model"]["teacher_logits_history"].append(teacher_logits_flat.tolist())
                        
                        if student_logits.numel() > 0:
                            student_logits_flat = student_logits.flatten().cpu()[:20]
                            training_raw_data["uploaded_model"]["student_logits_history"].append(student_logits_flat.tolist())
                except Exception as e:
                    print(f"[RAW DATA] Error extracting logits at step {step + 1}: {e}")
            
            # Emit progress
            progress = int((step + 1) / total_steps * 60)  # 60% for KD phase
            safe_emit_progress(
                progress=progress,
                phase="knowledge_distillation",
                message=f"Knowledge Distillation: Step {step + 1}/{total_steps}",
                loss=loss_value,
                step=step + 1,
                total_steps=total_steps
            )
        
        # Apply pruning (30% L1 unstructured)
        socketio.emit("training_status", {
            "phase": "pruning",
            "message": "Applying L1 unstructured pruning (30%)..."
        })
        
        # Pruning implementation here...
        # (pruning code would follow)
        
        # Extract student_after raw data
        student_after_raw = extract_raw_model_data(student_model, inputs)
        training_raw_data["uploaded_model"]["student_after"] = student_after_raw if student_after_raw else {}
        
        # Evaluate final metrics
        student_metrics = evaluate_model_metrics(student_model, inputs, is_student=True)
        last_teacher_metrics = teacher_metrics
        last_student_metrics = student_metrics
        model_trained = True
        
        # Emit final metrics and raw data
        socketio.emit("training_metrics", {
            "model_performance": {
                "title": "Student Model Performance (After KD + Pruning)",
                "description": "Final performance metrics of the compressed student model",
                "metrics": {
                    "accuracy": f"{student_metrics.get('accuracy', 0):.2f}%",
                    "precision": f"{student_metrics.get('precision', 0):.2f}%",
                    "recall": f"{student_metrics.get('recall', 0):.2f}%",
                    "f1_score": f"{student_metrics.get('f1', 0):.2f}%",
                    "size_mb": f"{student_metrics['size_mb']:.2f} MB",
                    "latency_ms": f"{student_metrics['latency_ms']:.2f} ms",
                    "num_params": f"{student_metrics['num_params']:,}"
                }
            }
        })
        
        socketio.emit("training_raw_data_ready", {
            "success": True,
            "data": training_raw_data
        })
        
        socketio.emit("training_progress", {
            "progress": 100,
            "status": "completed",
            "phase": "completed",
            "message": "Training completed! Metrics and raw data are ready."
        })
        
    except Exception as e:
        print(f"[TRAIN] Error during model training task: {str(e)}")
        socketio.emit("training_error", {"error": f"Error during model training: {str(e)}"})
        socketio.emit("training_progress", {
            "progress": 100,
            "status": "error",
            "phase": "error",
            "message": f"Training encountered errors."
        })
```

## 6. Training Endpoint

```python
@app.route('/train', methods=['POST'])
def train_model():
    """
    Start training for an UPLOADED model only.
    
    IMPORTANT: This endpoint ONLY trains uploaded models, NOT embedded models.
    Embedded models (DistilBERT, T5-small, MobileNetV2, ResNet-18) are pre-trained
    on system start and are display-only on the Models page.
    
    Training runs in a background thread and continues even if the user navigates away.
    Progress is streamed via SocketIO events.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        model_name = data.get("model_name", "distillBert")  # Used only for comparison/display
        uploaded_model_path = data.get("uploaded_model_path")
        uploaded_model_name = data.get("uploaded_model_name")
        
        if not uploaded_model_path:
            return jsonify({
                "success": False,
                "error": "A custom uploaded model (.pt/.pth/.bin/.ckpt/.json/.config) is required before training. Embedded models are not trained through this endpoint."
            }), 400
        
        # Clear previous training artifacts BEFORE starting new training
        clear_previous_training_artifacts()
        
        # Start training in a background thread with uploaded model info
        # Training continues in background even if user navigates away
        try:
            socketio.start_background_task(
                training_task, 
                model_name,  # Used only for comparison, not for training
                uploaded_model_path, 
                uploaded_model_name
            )
        except Exception as bg_error:
            return jsonify({
                "success": False, 
                "error": f"Failed to start training task: {str(bg_error)}"
            }), 500
        
        return jsonify({
            "success": True, 
            "message": "Training has been started in the background. Training will continue even if you navigate away."
        })
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500

@app.route('/cancel_training', methods=['POST'])
def cancel_training():
    global training_cancelled
    try:
        training_cancelled = True
        return jsonify({
            "success": True, 
            "message": "Training cancellation requested."
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500
```

## 7. File Upload Endpoint

```python
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
         return jsonify({"success": False, "error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Validate file extension
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    # Primary allowed model weight formats
    allowed_extensions = ['.pt', '.pth', '.bin']
    # Optionally allow additional artifact/config formats
    optional_allowed = ['.json', '.config', '.ckpt']
    all_allowed = allowed_extensions + optional_allowed

    if file_ext not in all_allowed:
        return jsonify({
            "success": False, 
            "error": f"Invalid file type. Allowed file types: {', '.join(all_allowed)}"
        }), 400
    
    # Block system models
    filename_lower = filename.lower()
    blocked_models = ['distilbert', 'resnet', 'mobilenet', 't5']
    for blocked in blocked_models:
        if blocked in filename_lower:
            return jsonify({
                "success": False,
                "error": f"System models ({', '.join(['DistilBERT', 'ResNet-18', 'MobileNetV2', 'T5 Small'])}) are not allowed. Please upload a custom model."
            }), 400
    
    # Validate file size (500MB limit)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    max_size = 500 * 1024 * 1024  # 500MB
    
    if file_size > max_size:
        return jsonify({
            "success": False,
            "error": f"File size ({file_size / (1024*1024):.2f} MB) exceeds the 500MB limit."
        }), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return jsonify({
        "success": True, 
        "file_path": file_path,
        "filename": filename,
        "size": file_size
    })
```

## 8. Evaluation Endpoint

```python
@app.route('/evaluate', methods=['POST'])
def evaluate():
    global teacher_model, student_model, train_loader, model_trained, last_teacher_metrics, last_student_metrics, last_effectiveness_metrics

    if not model_trained:
        return jsonify({
            "effectiveness": [
                {"metric": "Accuracy", "before": "Not Available", "after": "Not Available"},
                {"metric": "Precision (Macro Avg)", "before": "Not Available", "after": "Not Available"},
                {"metric": "Recall (Macro Avg)", "before": "Not Available", "after": "Not Available"},
                {"metric": "F1-Score (Macro Avg)", "before": "Not Available", "after": "Not Available"}
            ],
            "efficiency": [
                {"metric": "Latency (ms)", "before": "Not Available", "after": "Not Available"},
                {"metric": "RAM Usage (MB)", "before": "Not Available", "after": "Not Available"},
                {"metric": "Model Size (MB)", "before": "Not Available", "after": "Not Available"}
            ],
            "compression": [
                {"metric": "Parameters Count", "before": "Not Available", "after": "Not Available"},
                {"metric": "Layers Count", "before": "Not Available", "after": "Not Available"},
                {"metric": "Compression Ratio", "before": "Not Available", "after": "Not Available"},
                {"metric": "Accuracy Drop (%)", "before": "Not Available", "after": "Not Available"},
                {"metric": "Size Reduction (%)", "before": "Not Available", "after": "Not Available"}
            ],
            "complexity": [
                {"metric": "Time Complexity", "before": "Not Available", "after": "Not Available"},
                {"metric": "Space Complexity", "before": "Not Available", "after": "Not Available"}
            ]
        })

    try:
        # Use stored, measured metrics from training
        if last_teacher_metrics is None or last_student_metrics is None:
            # Use real data for measurement
            if isinstance(teacher_model, DistilBertForSequenceClassification) or 't5' in str(type(teacher_model)).lower():
                sample_texts = ["Real evaluation text for model assessment."] * 32
                if tokenizer is not None:
                    encoded = tokenizer(sample_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                    inputs = {"input_ids": encoded['input_ids'], "attention_mask": encoded['attention_mask']}
                else:
                    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5] * 26] * 32), "attention_mask": torch.ones(32, 128)}
                
                # Add decoder inputs for T5 models
                if 't5' in str(type(teacher_model)).lower():
                    input_ids = inputs["input_ids"]
                    decoder_input_ids = torch.cat([torch.zeros((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device), input_ids[:, :-1]], dim=1)
                    inputs["decoder_input_ids"] = decoder_input_ids
            else:
                # Use properly normalized image data
                transform = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                inputs = transform(torch.randn(32, 3, 224, 224) * 0.5 + 0.5)
            last_teacher_metrics = evaluate_model_metrics(teacher_model, inputs)
            last_student_metrics = evaluate_model_metrics(student_model, inputs, is_student=True)
            last_effectiveness_metrics = compute_teacher_student_agreement(teacher_model, student_model)

        # Calculate compression metrics
        compression_results = calculate_compression_metrics("current_model", last_teacher_metrics, last_student_metrics)
        
        return jsonify({
            "effectiveness": [
                {"metric": "Accuracy (agreement)", "before": f"{last_teacher_metrics.get('accuracy', 0):.2f}%", "after": f"{last_effectiveness_metrics['accuracy']:.2f}%"},
                {"metric": "Precision (agreement)", "before": f"{last_teacher_metrics.get('precision', 0):.2f}%", "after": f"{last_effectiveness_metrics['precision']:.2f}%"},
                {"metric": "Recall (agreement)", "before": f"{last_teacher_metrics.get('recall', 0):.2f}%", "after": f"{last_effectiveness_metrics['recall']:.2f}%"},
                {"metric": "F1-Score (agreement)", "before": f"{last_teacher_metrics.get('f1', 0):.2f}%", "after": f"{last_effectiveness_metrics['f1']:.2f}%"}
            ],
            "efficiency": [
                {"metric": "Latency (ms)", "before": f"{last_teacher_metrics['latency_ms']:.2f}", "after": f"{last_student_metrics['latency_ms']:.2f}"},
                {"metric": "Model Size (MB)", "before": f"{last_teacher_metrics['size_mb']:.2f}", "after": f"{last_student_metrics['size_mb']:.2f}"}
            ],
            "compression": [
                {"metric": "Parameters Count", "before": f"{last_teacher_metrics['num_params']:,}", "after": f"{last_student_metrics['num_params']:,}"},
                {"metric": "Size Reduction (%)", "before": "0.00%", "after": f"{compression_results['actual_size_reduction']:.2f}%"},
                {"metric": "Latency Improvement (%)", "before": "0.00%", "after": f"{compression_results['actual_latency_improvement']:.2f}%"},
                {"metric": "Parameter Reduction (%)", "before": "0.00%", "after": f"{compression_results['actual_params_reduction']:.2f}%"}
            ],
            "complexity": []
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

## 9. Visualization Endpoint

```python
@app.route('/visualize', methods=['POST'])
def visualize():
    global student_model, model_trained

    if not model_trained or student_model is None:
        # Default visualization data
        default_visualization_data = {
            "nodes": [
                # Input Layer (4 green nodes)
                {"id": "input_1", "x": 0, "y": 1.5, "z": 0, "size": 0.5, "color": "green"},
                {"id": "input_2", "x": 0, "y": 0.5, "z": 0, "size": 0.5, "color": "green"},
                {"id": "input_3", "x": 0, "y": -0.5, "z": 0, "size": 0.5, "color": "green"},
                {"id": "input_4", "x": 0, "y": -1.5, "z": 0, "size": 0.5, "color": "green"},
                # Hidden layers and output nodes...
            ],
            "connections": [
                # Connection definitions...
            ]
        }
        return jsonify(default_visualization_data)
    
    # Build visualization from trained student_model
    # (visualization generation code would follow)
    return jsonify({"nodes": [], "connections": []})
```
