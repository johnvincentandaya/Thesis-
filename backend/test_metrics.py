#!/usr/bin/env python3
"""
Test script to demonstrate real metrics computation
"""

import torch
import time
import numpy as np
from model_metrics import metrics_calculator

def test_model_metrics():
    """Test metrics computation for each model type."""
    
    print("=" * 80)
    print("REAL METRICS COMPUTATION DEMONSTRATION")
    print("=" * 80)
    
    # Test each model type
    models_to_test = [
        ("distilbert", "DistilBERT"),
        ("t5-small", "T5-small"), 
        ("mobilenetv2", "MobileNetV2"),
        ("resnet18", "ResNet-18")
    ]
    
    for model_name, display_name in models_to_test:
        print(f"\n🔍 Testing {display_name}...")
        print("-" * 50)
        
        try:
            # Load the model
            if model_name == "distilbert":
                from transformers import DistilBertForSequenceClassification, DistilBertConfig
                teacher_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
                student_config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
                student_config.dim = 384
                student_config.n_layers = 3
                student_model = DistilBertForSequenceClassification(student_config)
                
            elif model_name == "t5-small":
                from transformers import T5ForConditionalGeneration, T5Config
                teacher_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                student_config = T5Config(
                    d_model=256, num_layers=6, d_ff=1024, 
                    num_heads=4, vocab_size=teacher_model.config.vocab_size
                )
                student_model = T5ForConditionalGeneration(student_config)
                
            elif model_name == "mobilenetv2":
                from torchvision.models import mobilenet_v2
                teacher_model = mobilenet_v2(pretrained=True)
                student_model = mobilenet_v2(width_mult=0.5)
                
            elif model_name == "resnet18":
                from torchvision.models import resnet18
                teacher_model = resnet18(pretrained=True)
                student_model = resnet18()
            
            # Compute teacher metrics
            print(f"📊 Computing Teacher Model Metrics...")
            teacher_metrics = metrics_calculator.get_comprehensive_metrics(teacher_model, model_name)
            
            # Compute student metrics
            print(f"📊 Computing Student Model Metrics...")
            student_metrics = metrics_calculator.get_comprehensive_metrics(student_model, model_name)
            
            # Display results
            print(f"\n✅ {display_name} Results:")
            print(f"   Teacher Model:")
            print(f"     - Parameters: {teacher_metrics['parameter_count']:,}")
            print(f"     - Size: {teacher_metrics['size_mb']} MB")
            print(f"     - Accuracy: {teacher_metrics['accuracy']}%")
            print(f"     - F1-Score: {teacher_metrics['f1_score']}%")
            print(f"     - Latency: {teacher_metrics['inference_latency_ms']} ms")
            print(f"     - Complexity: {teacher_metrics['model_complexity']}")
            
            print(f"   Student Model:")
            print(f"     - Parameters: {student_metrics['parameter_count']:,}")
            print(f"     - Size: {student_metrics['size_mb']} MB")
            print(f"     - Accuracy: {student_metrics['accuracy']}%")
            print(f"     - F1-Score: {student_metrics['f1_score']}%")
            print(f"     - Latency: {student_metrics['inference_latency_ms']} ms")
            print(f"     - Complexity: {student_metrics['model_complexity']}")
            
            # Calculate improvements
            size_reduction = ((teacher_metrics['size_mb'] - student_metrics['size_mb']) / teacher_metrics['size_mb']) * 100
            param_reduction = ((teacher_metrics['parameter_count'] - student_metrics['parameter_count']) / teacher_metrics['parameter_count']) * 100
            latency_improvement = ((teacher_metrics['inference_latency_ms'] - student_metrics['inference_latency_ms']) / teacher_metrics['inference_latency_ms']) * 100
            
            print(f"   Improvements:")
            print(f"     - Size Reduction: {size_reduction:.1f}%")
            print(f"     - Parameter Reduction: {param_reduction:.1f}%")
            print(f"     - Latency Improvement: {latency_improvement:.1f}%")
            
        except Exception as e:
            print(f"❌ Error testing {display_name}: {str(e)}")
    
    print(f"\n" + "=" * 80)
    print("METRICS COMPUTATION EXPLANATION")
    print("=" * 80)
    
    print(f"""
🔬 How Real Metrics Are Computed:

1. **Performance Metrics (Accuracy, F1-Score, Precision, Recall):**
   - Generate synthetic test data appropriate for each model type
   - Run multiple forward passes (5 runs) to get stable measurements
   - Use scikit-learn to calculate actual metrics from predictions vs ground truth
   - For NLP models: Use text-like data with vocabulary
   - For Vision models: Use image-like data (224x224x3)

2. **Inference Latency:**
   - Warm up the model with 10 forward passes
   - Measure 100 inference runs with high-precision timing
   - Calculate average latency in milliseconds
   - Real measurement, not estimation

3. **Model Size:**
   - Actually save the model to a temporary file
   - Measure the file size in bytes
   - Convert to MB for display
   - Real disk space measurement

4. **Parameter Count:**
   - Sum all trainable parameters using PyTorch's numel()
   - Real parameter count, not estimation

5. **Model Complexity:**
   - Based on actual parameter count:
     * < 1M parameters = Low
     * 1M-10M parameters = Medium  
     * > 10M parameters = High

6. **FLOPs (Floating Point Operations):**
   - Estimate based on model architecture and input size
   - For CNNs: parameters × input_size
   - For Transformers: parameters × sequence_length

7. **Model Size Reduction:**
   - Calculate: ((Teacher_Size - Student_Size) / Teacher_Size) × 100
   - Real percentage reduction in model size

All metrics are computed in real-time when you click "Train & Visualize"!
""")

if __name__ == "__main__":
    test_model_metrics() 