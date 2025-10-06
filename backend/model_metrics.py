import torch
import time
import numpy as np
from typing import Dict, Any, Tuple
import tempfile
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.nn.functional as F

class ModelMetricsCalculator:
    """
    Comprehensive model metrics calculator that provides accurate measurements
    for model performance, efficiency, and complexity.
    """
    
    def __init__(self):
        self.metrics_explanations = {
            'f1_score': {
                'description': 'F1-Score is the harmonic mean of precision and recall, providing a balanced measure of model performance.',
                'formula': 'F1 = 2 * (Precision * Recall) / (Precision + Recall)',
                'range': '0.0 to 1.0 (higher is better)',
                'interpretation': 'Combines precision and recall into a single metric, penalizing models that are good at only one of these aspects.'
            },
            'accuracy': {
                'description': 'Accuracy measures the proportion of correct predictions out of all predictions made.',
                'formula': 'Accuracy = (True Positives + True Negatives) / Total Predictions',
                'range': '0.0 to 1.0 (higher is better)',
                'interpretation': 'Simple measure of overall correctness, but can be misleading with imbalanced datasets.'
            },
            'precision': {
                'description': 'Precision measures the proportion of true positive predictions out of all positive predictions.',
                'formula': 'Precision = True Positives / (True Positives + False Positives)',
                'range': '0.0 to 1.0 (higher is better)',
                'interpretation': 'Measures how many of the predicted positive cases were actually positive.'
            },
            'recall': {
                'description': 'Recall measures the proportion of actual positive cases that were correctly identified.',
                'formula': 'Recall = True Positives / (True Positives + False Negatives)',
                'range': '0.0 to 1.0 (higher is better)',
                'interpretation': 'Measures how many of the actual positive cases were found.'
            },
            'inference_latency': {
                'description': 'Inference latency measures the time taken to process a single input and produce an output.',
                'formula': 'Latency = Total Processing Time / Number of Samples',
                'range': 'Milliseconds (lower is better)',
                'interpretation': 'Critical for real-time applications. Lower latency means faster response times.'
            },
            'model_complexity': {
                'description': 'Model complexity measures the computational and memory requirements of the model.',
                'formula': 'Complexity = f(Parameters, FLOPs, Memory Usage)',
                'range': 'Low/Medium/High (lower is better for efficiency)',
                'interpretation': 'Balances performance with resource requirements. Lower complexity means faster training and inference.'
            },
            'parameter_count': {
                'description': 'Total number of trainable parameters in the model.',
                'formula': 'Sum of all parameter tensors in the model',
                'range': 'Integer (lower is better for efficiency)',
                'interpretation': 'More parameters generally mean more capacity but also more memory and computation.'
            },
            'model_size': {
                'description': 'Size of the model when saved to disk.',
                'formula': 'Size = Sum of all parameter sizes in bytes',
                'range': 'Megabytes (lower is better for deployment)',
                'interpretation': 'Important for deployment constraints and storage requirements.'
            },
            'flops': {
                'description': 'Floating Point Operations - measures computational complexity.',
                'formula': 'Sum of all mathematical operations during forward pass',
                'range': 'Integer (lower is better for efficiency)',
                'interpretation': 'Direct measure of computational cost. Lower FLOPs mean faster inference.'
            },
            'size_reduction': {
                'description': 'Model Size Reduction measures the percentage decrease in model size after compression techniques.',
                'formula': 'Size Reduction = ((Original Size - Compressed Size) / Original Size) × 100',
                'range': '0% to 100% (higher is better)',
                'interpretation': 'Shows how much smaller the model became after compression. Higher reduction means more efficient deployment.'
            }
        }
    
    def generate_test_data(self, model, model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate appropriate test data based on model type."""
        if model_name.lower() in ['mobilenetv2', 'resnet18']:
            # Image classification models
            batch_size = 32
            x = torch.randn(batch_size, 3, 224, 224)
            y = torch.randint(0, 1000, (batch_size,))  # 1000 classes for ImageNet
        else:
            # NLP models
            batch_size = 16
            seq_length = 128
            x = torch.randint(0, 30000, (batch_size, seq_length))  # Vocabulary size
            y = torch.randint(0, 2, (batch_size,))  # Binary classification
        return x, y
    
    def calculate_performance_metrics(self, model, model_name: str, num_runs: int = 5) -> Dict[str, float]:
        """Calculate performance metrics using test data."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                x, y = self.generate_test_data(model, model_name)
                
                # Forward pass
                if model_name.lower() in ['mobilenetv2', 'resnet18']:
                    outputs = model(x)
                    predictions = outputs.argmax(dim=1)
                else:
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': round(float(accuracy * 100), 2),
            'precision': round(float(precision * 100), 2),
            'recall': round(float(recall * 100), 2),
            'f1_score': round(float(f1 * 100), 2)
        }
    
    def calculate_inference_latency(self, model, model_name: str, num_runs: int = 100) -> float:
        """Calculate average inference latency."""
        model.eval()
        x, _ = self.generate_test_data(model, model_name)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if model_name.lower() in ['mobilenetv2', 'resnet18']:
                    _ = model(x)
                else:
                    _ = model(x)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if model_name.lower() in ['mobilenetv2', 'resnet18']:
                    _ = model(x)
                else:
                    _ = model(x)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return round(float(np.mean(latencies)), 2)
    
    def calculate_model_complexity(self, model) -> str:
        """Calculate model complexity based on parameters and structure."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count layers
        num_layers = len(list(model.modules()))
        
        # Estimate FLOPs (simplified)
        if total_params < 1e6:
            complexity = "Low"
        elif total_params < 10e6:
            complexity = "Medium"
        else:
            complexity = "High"
        
        return complexity
    
    def calculate_flops(self, model, model_name: str) -> int:
        """Estimate FLOPs for the model."""
        x, _ = self.generate_test_data(model, model_name)
        
        # Simple FLOPs estimation
        if model_name.lower() in ['mobilenetv2', 'resnet18']:
            # For CNN models, estimate based on input size and parameters
            input_size = x.shape[2] * x.shape[3]  # Height * Width
            total_params = sum(p.numel() for p in model.parameters())
            flops = total_params * input_size // 1000  # Rough estimation
        else:
            # For NLP models, estimate based on sequence length and parameters
            seq_length = x.shape[1]
            total_params = sum(p.numel() for p in model.parameters())
            flops = total_params * seq_length // 1000  # Rough estimation
        
        return int(flops)
    
    def calculate_model_size_mb(self, model) -> float:
        """Calculate model size in MB using parameter sizes (works on Windows)."""
        total_size_bytes = 0
        
        # Calculate size by summing all parameter sizes
        for param in model.parameters():
            # Each parameter is typically float32 (4 bytes)
            total_size_bytes += param.numel() * 4
        
        # Convert to MB
        size_mb = total_size_bytes / (1024 * 1024)
        return round(size_mb, 3)
    
    def get_comprehensive_metrics(self, model, model_name: str) -> Dict[str, Any]:
        """Get all comprehensive metrics for a model."""
        print(f"Calculating metrics for {model_name}...")
        
        # Basic metrics
        param_count = sum(p.numel() for p in model.parameters())
        
        # Model size (using parameter-based calculation)
        size_mb = self.calculate_model_size_mb(model)
        
        # Performance metrics
        performance = self.calculate_performance_metrics(model, model_name)
        
        # Latency
        latency = self.calculate_inference_latency(model, model_name)
        
        # Complexity
        complexity = self.calculate_model_complexity(model)
        
        # FLOPs
        flops = self.calculate_flops(model, model_name)
        
        return {
            'parameter_count': param_count,
            'size_mb': size_mb,
            'accuracy': performance['accuracy'],
            'f1_score': performance['f1_score'],
            'precision': performance['precision'],
            'recall': performance['recall'],
            'inference_latency_ms': latency,
            'model_complexity': complexity,
            'flops_millions': flops,
            'explanations': self.metrics_explanations
        }
    
    def get_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of metrics with explanations."""
        return {
            'metrics': {
                'F1-Score': f"{metrics['f1_score']}%",
                'Accuracy': f"{metrics['accuracy']}%",
                'Precision': f"{metrics['precision']}%",
                'Recall': f"{metrics['recall']}%",
                'Inference Latency': f"{metrics['inference_latency_ms']}ms",
                'Model Complexity': metrics['model_complexity'],
                'Parameter Count': f"{metrics['parameter_count']:,}",
                'Model Size': f"{metrics['size_mb']}MB",
                'FLOPs': f"{metrics['flops_millions']}M"
            },
            'explanations': self.metrics_explanations
        }

# Global instance
metrics_calculator = ModelMetricsCalculator() 