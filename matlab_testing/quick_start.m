%% Quick Start Script for KD-Pruning Algorithm
% This script provides a simple way to test the algorithm
% Just run this file and follow the prompts

clear; clc; close all;

fprintf('=== Quick Start: KD-Pruning Algorithm ===\n\n');

%% Step 1: Choose Algorithm Version
fprintf('Choose algorithm version:\n');
fprintf('1. Simplified version (no additional toolboxes required)\n');
fprintf('2. Full version (requires Deep Learning Toolbox)\n');
fprintf('3. Both versions (recommended for comparison)\n\n');

choice = input('Enter your choice (1, 2, or 3): ');

if isempty(choice)
    choice = 3; % Default to both
    fprintf('Using default: Both versions\n\n');
end

%% Step 2: Set Algorithm Parameters
fprintf('Set algorithm parameters:\n');
pruning_ratio = input('Pruning ratio (0.1 to 0.5, default 0.3): ');
if isempty(pruning_ratio)
    pruning_ratio = 0.3;
end

if choice >= 2
    temperature = input('Temperature for KD (1.0 to 5.0, default 2.0): ');
    if isempty(temperature)
        temperature = 2.0;
    end
    
    epochs = input('Training epochs (1 to 20, default 5): ');
    if isempty(epochs)
        epochs = 5;
    end
end

fprintf('\nParameters set:\n');
fprintf('  Pruning ratio: %.1f%%\n', pruning_ratio * 100);
if choice >= 2
    fprintf('  Temperature: %.1f\n', temperature);
    fprintf('  Epochs: %d\n', epochs);
end
fprintf('\n');

%% Step 3: Create Test Model and Data
fprintf('Creating test model and data...\n');

% Create a simple neural network
input_size = 10;
hidden_size = 20;
output_size = 3;

% Initialize weights
W1 = randn(hidden_size, input_size) * 0.1;
b1 = zeros(hidden_size, 1);
W2 = randn(output_size, hidden_size) * 0.1;
b2 = zeros(output_size, 1);

% Create teacher model
teacher_model = struct();
teacher_model.W1 = W1;
teacher_model.b1 = b1;
teacher_model.W2 = W2;
teacher_model.b2 = b2;
teacher_model.Learnables = table();

% Add learnable parameters
teacher_model.Learnables = table({W1; b1; W2; b2}, ...
    'RowNames', {'W1'; 'b1'; 'W2'; 'b2'}, ...
    'VariableNames', {'Value'});

% Generate synthetic data
num_samples = 500;
X = randn(num_samples, input_size);
labels = randi([1, output_size], num_samples, 1);

training_data = struct();
for i = 1:num_samples
    training_data(i).inputs = X(i, :)';
    training_data(i).labels = labels(i);
end

fprintf('  Teacher model: %d parameters\n', numel(W1) + numel(b1) + numel(W2) + numel(b2));
fprintf('  Training data: %d samples\n', num_samples);
fprintf('\n');

%% Step 4: Run Selected Algorithm(s)
results = struct();

if choice == 1 || choice == 3
    fprintf('Running simplified version...\n');
    try
        [student_simple, metrics_simple] = simplified_kd_pruning(...
            teacher_model, training_data, pruning_ratio);
        results.simplified = metrics_simple;
        fprintf('✓ Simplified version completed successfully!\n\n');
    catch ME
        fprintf('✗ Simplified version failed: %s\n\n', ME.message);
        results.simplified = [];
    end
end

if choice == 2 || choice == 3
    fprintf('Running full version...\n');
    try
        [student_full, metrics_full] = kd_pruning_algorithm(...
            teacher_model, training_data, pruning_ratio, temperature, epochs);
        results.full = metrics_full;
        fprintf('✓ Full version completed successfully!\n\n');
    catch ME
        fprintf('✗ Full version failed: %s\n\n', ME.message);
        fprintf('  This might be due to missing Deep Learning Toolbox\n');
        fprintf('  The simplified version should still work\n\n');
        results.full = [];
    end
end

%% Step 5: Display Results
fprintf('=== Results Summary ===\n\n');

if isfield(results, 'simplified') && ~isempty(results.simplified)
    fprintf('Simplified Version Results:\n');
    display_quick_results(results.simplified);
    fprintf('\n');
end

if isfield(results, 'full') && ~isempty(results.full)
    fprintf('Full Version Results:\n');
    display_quick_results(results.full);
    fprintf('\n');
end

%% Step 6: Create Visualization
fprintf('Creating visualization...\n');
try
    if isfield(results, 'simplified') && ~isempty(results.simplified)
        create_quick_plot(results.simplified, 'Simplified Version');
    elseif isfield(results, 'full') && ~isempty(results.full)
        create_quick_plot(results.full, 'Full Version');
    end
    fprintf('✓ Visualization created and saved\n');
catch ME
    fprintf('✗ Visualization failed: %s\n', ME.message);
end

%% Step 7: Save Results
fprintf('\nSaving results...\n');
try
    save('kd_pruning_results.mat', 'results', 'teacher_model', 'student_simple', 'pruning_ratio');
    if exist('student_full', 'var')
        save('kd_pruning_results.mat', 'student_full', '-append');
    end
    fprintf('✓ Results saved to kd_pruning_results.mat\n');
catch ME
    fprintf('✗ Failed to save results: %s\n', ME.message);
end

fprintf('\n=== Quick Start Complete ===\n');
fprintf('You can now:\n');
fprintf('1. Analyze the results above\n');
fprintf('2. View the generated plots\n');
fprintf('3. Load results later: load(''kd_pruning_results.mat'')\n');
fprintf('4. Modify parameters and run again\n');

%% Helper Functions

function display_quick_results(metrics)
    fprintf('  Size Reduction: %.1f%%\n', metrics.improvements.size_reduction_percent);
    fprintf('  Parameter Reduction: %.1f%%\n', metrics.improvements.param_reduction_percent);
    fprintf('  Latency Improvement: %.1f%%\n', metrics.improvements.latency_improvement_percent);
    fprintf('  Accuracy Impact: %.1f%%\n', metrics.improvements.accuracy_impact);
end

function create_quick_plot(metrics, title_str)
    figure('Name', title_str, 'Position', [100, 100, 600, 400]);
    
    % Create comparison bar chart
    categories = {'Size (MB)', 'Parameters', 'Latency (ms)', 'Accuracy (%)'};
    teacher_values = [metrics.teacher.size_mb, metrics.teacher.num_params/1000, ...
                     metrics.teacher.latency_ms, metrics.teacher.accuracy];
    student_values = [metrics.student.size_mb, metrics.student.num_params/1000, ...
                     metrics.student.latency_ms, metrics.student.accuracy];
    
    x = 1:length(categories);
    width = 0.35;
    
    bar(x - width/2, teacher_values, width, 'DisplayName', 'Teacher');
    hold on;
    bar(x + width/2, student_values, width, 'DisplayName', 'Student');
    
    xlabel('Metrics');
    ylabel('Values');
    title([title_str, ' - Teacher vs Student Comparison']);
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', categories);
    legend('Location', 'best');
    grid on;
    
    % Save the plot
    filename = sprintf('quick_results_%s.png', lower(strrep(title_str, ' ', '_')));
    saveas(gcf, filename);
end
