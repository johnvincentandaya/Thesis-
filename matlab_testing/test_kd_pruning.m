%% Test Script for Knowledge Distillation and Pruning Algorithm
% This script demonstrates how to test the KD-pruning algorithm in MATLAB
% It creates a simple neural network and applies the compression techniques

clear; clc; close all;

fprintf('=== Testing KD-Pruning Algorithm in MATLAB ===\n\n');

%% Step 1: Create a Simple Teacher Model
fprintf('1. Creating a simple teacher model...\n');

% Create a simple feedforward neural network
input_size = 10;
hidden_size = 20;
output_size = 3;

% Initialize weights randomly
W1 = randn(hidden_size, input_size) * 0.1;
b1 = zeros(hidden_size, 1);
W2 = randn(output_size, hidden_size) * 0.1;
b2 = zeros(output_size, 1);

% Create teacher model structure
teacher_model = struct();
teacher_model.W1 = W1;
teacher_model.b1 = b1;
teacher_model.W2 = W2;
teacher_model.b2 = b2;
teacher_model.Learnables = table();

% Add learnable parameters to the table
teacher_model.Learnables = table({W1; b1; W2; b2}, ...
    'RowNames', {'W1'; 'b1'; 'W2'; 'b2'}, ...
    'VariableNames', {'Value'});

fprintf('   Teacher model created with %d parameters\n', ...
    numel(W1) + numel(b1) + numel(W2) + numel(b2));

%% Step 2: Generate Synthetic Training Data
fprintf('\n2. Generating synthetic training data...\n');

num_samples = 1000;
num_features = input_size;
num_classes = output_size;

% Generate random input data
X = randn(num_samples, num_features);

% Generate labels (simple classification task)
labels = randi([1, num_classes], num_samples, 1);

% Create training data structure
training_data = struct();
for i = 1:num_samples
    training_data(i).inputs = X(i, :)';
    training_data(i).labels = labels(i);
end

fprintf('   Generated %d training samples\n', num_samples);

%% Step 3: Test the KD-Pruning Algorithm
fprintf('\n3. Running KD-Pruning algorithm...\n');

% Algorithm parameters
pruning_ratio = 0.3;  % Prune 30% of weights
temperature = 2.0;     % Temperature for knowledge distillation
epochs = 5;           % Number of training epochs

try
    % Run the algorithm
    [student_model, metrics] = kd_pruning_algorithm(...
        teacher_model, training_data, pruning_ratio, temperature, epochs);
    
    fprintf('\nAlgorithm completed successfully!\n');
    
    % Display results
    display_results(metrics);
    
catch ME
    fprintf('Error running algorithm: %s\n', ME.message);
    fprintf('This might be due to missing Deep Learning Toolbox functions\n');
    fprintf('Consider using the simplified version or installing required toolboxes\n');
end

%% Step 4: Alternative Simplified Testing (if Deep Learning Toolbox not available)
fprintf('\n4. Running simplified version for basic testing...\n');

% Simplified version that doesn't require Deep Learning Toolbox
[student_model_simple, metrics_simple] = simplified_kd_pruning(...
    teacher_model, training_data, pruning_ratio);

fprintf('Simplified algorithm completed!\n');
display_results(metrics_simple);

%% Step 5: Visualization
fprintf('\n5. Creating visualizations...\n');

% Create comparison plots
create_comparison_plots(metrics_simple);

fprintf('\n=== Testing Complete ===\n');

%% Helper Functions

function display_results(metrics)
    % Display the results in a formatted way
    
    fprintf('\n--- Results Summary ---\n');
    fprintf('Teacher Model:\n');
    fprintf('  Size: %.2f MB\n', metrics.teacher.size_mb);
    fprintf('  Parameters: %d\n', metrics.teacher.num_params);
    fprintf('  Latency: %.2f ms\n', metrics.teacher.latency_ms);
    fprintf('  Accuracy: %.2f%%\n', metrics.teacher.accuracy);
    
    fprintf('\nStudent Model:\n');
    fprintf('  Size: %.2f MB\n', metrics.student.size_mb);
    fprintf('  Parameters: %d\n', metrics.student.num_params);
    fprintf('  Latency: %.2f ms\n', metrics.student.latency_ms);
    fprintf('  Accuracy: %.2f%%\n', metrics.student.accuracy);
    
    fprintf('\nImprovements:\n');
    fprintf('  Size Reduction: %.2f%%\n', metrics.improvements.size_reduction_percent);
    fprintf('  Parameter Reduction: %.2f%%\n', metrics.improvements.param_reduction_percent);
    fprintf('  Latency Improvement: %.2f%%\n', metrics.improvements.latency_improvement_percent);
    fprintf('  Accuracy Impact: %.2f%%\n', metrics.improvements.accuracy_impact);
end

function create_comparison_plots(metrics)
    % Create comparison plots
    
    % Model size comparison
    figure('Name', 'Model Size Comparison', 'Position', [100, 100, 800, 600]);
    
    subplot(2, 2, 1);
    sizes = [metrics.teacher.size_mb, metrics.student.size_mb];
    labels = {'Teacher', 'Student'};
    bar(sizes);
    title('Model Size (MB)');
    set(gca, 'XTickLabel', labels);
    ylabel('Size (MB)');
    
    % Parameter count comparison
    subplot(2, 2, 2);
    params = [metrics.teacher.num_params, metrics.student.num_params];
    bar(params);
    title('Parameter Count');
    set(gca, 'XTickLabel', labels);
    ylabel('Number of Parameters');
    
    % Latency comparison
    subplot(2, 2, 3);
    latencies = [metrics.teacher.latency_ms, metrics.student.latency_ms];
    bar(latencies);
    title('Inference Latency');
    set(gca, 'XTickLabel', labels);
    ylabel('Latency (ms)');
    
    % Accuracy comparison
    subplot(2, 2, 4);
    accuracies = [metrics.teacher.accuracy, metrics.student.accuracy];
    bar(accuracies);
    title('Model Accuracy');
    set(gca, 'XTickLabel', labels);
    ylabel('Accuracy (%)');
    
    % Add improvement percentages
    sgtitle('Knowledge Distillation and Pruning Results', 'FontSize', 14);
    
    % Save the figure
    saveas(gcf, 'kd_pruning_results.png');
    fprintf('Results visualization saved as kd_pruning_results.png\n');
end
