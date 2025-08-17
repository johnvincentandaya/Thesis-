# Testing Knowledge Distillation and Pruning Algorithm in MATLAB

This directory contains MATLAB implementations of the Knowledge Distillation (KD) and Pruning algorithm that mirrors the functionality of your Python backend.

## Overview

The algorithm combines two powerful model compression techniques:
1. **Knowledge Distillation**: A smaller student model learns from a larger teacher model
2. **L1 Unstructured Pruning**: Removes less important weights to reduce model size

## Files Description

### 1. `kd_pruning_algorithm.m` (Full Implementation)
- **Purpose**: Complete implementation with Deep Learning Toolbox support
- **Features**: 
  - Full knowledge distillation training loop
  - L1 unstructured pruning
  - Comprehensive metrics calculation
  - Real-time training progress
- **Requirements**: Deep Learning Toolbox, Statistics and Machine Learning Toolbox

### 2. `simplified_kd_pruning.m` (Basic Implementation)
- **Purpose**: Simplified version that works without additional toolboxes
- **Features**:
  - Basic pruning functionality
  - Simplified metrics calculation
  - No external dependencies
- **Requirements**: Basic MATLAB installation only

### 3. `test_kd_pruning.m` (Test Script)
- **Purpose**: Demonstrates how to use both implementations
- **Features**:
  - Creates synthetic neural network
  - Generates test data
  - Runs both algorithm versions
  - Creates visualizations
  - Saves results

## How to Test the Algorithm

### Prerequisites
- MATLAB R2018b or later
- Basic understanding of neural networks
- (Optional) Deep Learning Toolbox for full functionality

### Step-by-Step Testing

#### 1. Basic Testing (Recommended to start)
```matlab
% Navigate to the matlab_testing directory
cd matlab_testing

% Run the test script
test_kd_pruning
```

This will:
- Create a simple teacher model
- Generate synthetic training data
- Apply the simplified KD-pruning algorithm
- Display results and create visualizations

#### 2. Advanced Testing (If you have Deep Learning Toolbox)
```matlab
% The test script will automatically try the full version
% If successful, you'll see more detailed training progress
```

#### 3. Custom Testing
```matlab
% Create your own teacher model
my_teacher = create_custom_model();

% Generate your own training data
my_data = generate_custom_data();

% Apply the algorithm
[student, metrics] = simplified_kd_pruning(my_teacher, my_data, 0.3);

% Analyze results
display_results(metrics);
```

## Understanding the Results

The algorithm provides comprehensive metrics:

### Model Comparison
- **Size**: Model file size in MB
- **Parameters**: Number of learnable parameters
- **Latency**: Inference time in milliseconds
- **Accuracy**: Classification accuracy percentage

### Improvements
- **Size Reduction**: Percentage decrease in model size
- **Parameter Reduction**: Percentage decrease in parameters
- **Latency Improvement**: Percentage improvement in speed
- **Accuracy Impact**: Change in accuracy (can be positive or negative)

## Algorithm Parameters

### Pruning Ratio
- **Range**: 0.0 to 1.0
- **Meaning**: Percentage of weights to remove
- **Example**: 0.3 means 30% of weights will be pruned
- **Trade-off**: Higher pruning = smaller model but potentially lower accuracy

### Temperature (Full version only)
- **Range**: 1.0 to 10.0
- **Meaning**: Controls knowledge transfer smoothness
- **Effect**: Higher temperature = softer probability distributions
- **Recommended**: 2.0 to 5.0

### Training Epochs (Full version only)
- **Range**: 1 to 100+
- **Meaning**: Number of training iterations
- **Effect**: More epochs = better knowledge transfer but longer training
- **Recommended**: Start with 5-10 epochs

## Expected Results

### Typical Compression Results
- **Size Reduction**: 25-50% depending on pruning ratio
- **Parameter Reduction**: 25-50% depending on pruning ratio
- **Latency Improvement**: 20-40% depending on hardware
- **Accuracy Impact**: -1% to -5% (small drop for significant gains)

### Example Output
```
=== Simplified Model Compression Results ===
Size Reduction: 30.00%
Parameter Reduction: 30.00%
Latency Improvement: 25.00%
Accuracy Impact: -2.50%
```

## Troubleshooting

### Common Issues

#### 1. "Function not found" errors
- **Solution**: Make sure all .m files are in the same directory
- **Check**: Run `dir *.m` to see available functions

#### 2. Deep Learning Toolbox errors
- **Solution**: Use the simplified version (`simplified_kd_pruning`)
- **Alternative**: Install Deep Learning Toolbox if available

#### 3. Memory issues with large models
- **Solution**: Reduce model size or use smaller training data
- **Check**: Monitor memory usage with `memory` command

#### 4. Slow performance
- **Solution**: Reduce number of test samples
- **Optimization**: Use `gpuArray` if GPU is available

### Performance Tips
- Start with small models for testing
- Use synthetic data for initial validation
- Monitor memory usage during execution
- Save results frequently for large experiments

## Extending the Algorithm

### Adding New Pruning Methods
```matlab
function pruned_model = apply_custom_pruning(model, pruning_ratio)
    % Implement your custom pruning strategy here
    % Example: L2 pruning, structured pruning, etc.
end
```

### Custom Metrics
```matlab
function custom_metrics = calculate_custom_metrics(model)
    % Add your own evaluation metrics
    % Example: FLOPs, memory usage, etc.
end
```

### Integration with Real Models
```matlab
% Load pre-trained models
teacher = load('pretrained_model.mat');

% Convert to compatible format
teacher_model = convert_model_format(teacher);

% Apply KD-pruning
[student, metrics] = simplified_kd_pruning(teacher_model, data, 0.3);
```

## Comparison with Python Implementation

| Feature | Python (PyTorch) | MATLAB (Simplified) |
|---------|------------------|---------------------|
| Knowledge Distillation | Full implementation | Basic simulation |
| Pruning | L1 unstructured | L1 unstructured |
| Training Loop | Complete with optimizer | Simplified |
| Metrics | Comprehensive | Basic but functional |
| Dependencies | PyTorch, Flask | Basic MATLAB only |
| Performance | Production-ready | Educational/demo |

## Next Steps

1. **Run the basic test** to understand the algorithm
2. **Experiment with different pruning ratios** to see trade-offs
3. **Try with your own models** if you have them
4. **Extend the implementation** for your specific needs
5. **Compare results** with the Python backend

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify MATLAB version compatibility
3. Ensure all files are in the same directory
4. Try the simplified version first

The MATLAB implementation provides a solid foundation for understanding and testing the KD-pruning algorithm without external dependencies.
