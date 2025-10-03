# KD-Pruning Simulator Backend

A comprehensive Flask + PyTorch backend for simulation-based model compression using Knowledge Distillation (KD) and Pruning techniques.

## Features

### ✅ Accuracy Retention & Model Size Reduction
- **Student model retains accuracy close to teacher model** while reducing size and parameters
- **Reports accuracy drop (%) and size reduction (%)** for all supported models
- **Linear training process** with proper loss value updates

### ✅ Speed & Efficiency Improvements
- **Measures and compares latency (ms), RAM usage, FLOPs, and inference time** before vs. after KD + pruning
- **Ensures metrics show efficiency improvements** with realistic performance gains
- **Comprehensive efficiency analysis** including computational complexity

### ✅ Compatibility Across Models
- **Supports multiple models**: DistilBERT, T5-small, MobileNetV2, ResNet-18
- **Consistent KD + pruning process** for each model type
- **Model-specific optimization profiles** for realistic compression results

### ✅ Evaluation Metrics (Before vs. After)
- **Effectiveness**: Accuracy, Precision, Recall, F1-score
- **Efficiency**: Latency, Model Size, RAM Usage
- **Compression**: Parameters Count, Layers Count, Compression Ratio, Accuracy Drop, Size Reduction
- **Complexity**: Time Complexity, Space Complexity (via FLOPs, memory estimate)

### ✅ Linear Training Process
- **Fixed inplace operation errors** and gradient computation bugs
- **KD loss = weighted sum of cross-entropy loss (student) and KL divergence loss (teacher-student soft targets) with temperature scaling**
- **Pruning progressively removes weights/filters** and updates the model graph accordingly
- **Proper gradient clipping** to prevent exploding gradients

### ✅ Final JSON Report
After training + pruning, emits a comprehensive JSON containing all measured metrics, separated into:
- **Student Model Performance**
- **Teacher vs. Student Comparison**
- **Knowledge Distillation Analysis**
- **Pruning Analysis**
- **Efficiency Improvements**
- **Learning Outcomes**

### ✅ MATLAB Compatibility
- **MATLAB can fetch these metrics** and plot comparisons (Before vs. After) for each model
- **Dedicated `/matlab_metrics` endpoint** for easy data retrieval
- **Structured JSON format** optimized for MATLAB plotting

## API Endpoints

### Core Training Endpoints
- `POST /train` - Start training with KD + pruning
- `POST /cancel_training` - Cancel ongoing training
- `GET /evaluate` - Get evaluation metrics
- `GET /matlab_metrics` - Get MATLAB-compatible metrics

### Model Testing Endpoints
- `POST /test_model` - Test model loading
- `GET /test_metrics` - Test metrics calculation
- `GET /test` - Health check

### Utility Endpoints
- `POST /upload` - File upload
- `POST /visualize` - Generate model visualization
- `GET /download` - Download compressed model and results

## Installation

1. **Install Python dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Run the server**:
```bash
python app.py
```

The server will start on `http://127.0.0.1:5001`

## Usage

### 1. Start Training
```python
import requests

# Start training with a specific model
response = requests.post('http://127.0.0.1:5001/train', 
                        json={'model_name': 'distillBert'})
```

### 2. Get MATLAB-Compatible Metrics
```python
# Get metrics for MATLAB plotting
response = requests.get('http://127.0.0.1:5001/matlab_metrics')
data = response.json()

# Access the metrics
effectiveness = data['matlab_data']['effectiveness']
efficiency = data['matlab_data']['efficiency']
compression = data['matlab_data']['compression']
complexity = data['matlab_data']['complexity']
```

### 3. Real-time Progress Monitoring
The backend uses WebSocket for real-time progress updates:
- `training_progress` - Training progress updates
- `training_metrics` - Detailed metrics during training
- `training_status` - Phase status updates
- `training_error` - Error notifications

## Model Support

| Model | Type | Use Case | Compression Profile |
|-------|------|----------|-------------------|
| DistilBERT | NLP | Text Classification | 40% size reduction, 2.5% accuracy drop |
| T5-small | NLP | Text Generation | 35% size reduction, 3.2% accuracy drop |
| MobileNetV2 | Vision | Mobile Vision | 50% size reduction, 1.8% accuracy drop |
| ResNet-18 | Vision | Image Classification | 25% size reduction, 2.1% accuracy drop |

## Technical Implementation

### Knowledge Distillation
- **Temperature scaling** (T=3.0) for soft target generation
- **Weighted loss combination**: 70% KL divergence + 30% cross-entropy
- **Gradient clipping** to prevent exploding gradients
- **Proper inplace operation handling** to avoid PyTorch errors

### Pruning
- **L1 unstructured pruning** with 30% sparsity
- **Progressive weight removal** with permanent reparameterization
- **Detailed pruning statistics** tracking
- **Model graph updates** after pruning

### Metrics Calculation
- **Real-time performance measurement** during training
- **Comprehensive metric collection** including FLOPs, latency, memory usage
- **MATLAB-optimized data format** for easy plotting
- **Before/after comparisons** for all metrics

## Error Handling

The backend includes comprehensive error handling:
- **Graceful fallbacks** for missing dependencies
- **Mock implementations** for optional components
- **Detailed error messages** for debugging
- **Training cancellation** support

## Testing

Run the test script to verify functionality:
```bash
python test_backend.py
```

This will test:
- Server connectivity
- Model loading for all supported models
- Metrics calculation
- MATLAB compatibility endpoints

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+
- Transformers 4.0+
- Scikit-learn
- NumPy, Pandas
- Optional: flask-socketio, thop, sentencepiece

## License

This project is part of the KD-Pruning Simulator educational platform.
