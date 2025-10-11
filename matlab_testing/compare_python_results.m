%% Compare Python Backend Results with MATLAB Implementation
% This script loads JSON results from Python backend and compares them
% with MATLAB KD-pruning algorithm results

clear; clc; close all;

fprintf('=== Comparing Python Backend vs MATLAB Results ===\n\n');

%% Step 1: Load Python Results
fprintf('1. Loading Python backend results...\n');

% Define the exports directory path
% Try multiple possible locations
possible_paths = {
    '../backend/exports',
    'backend/exports', 
    'exports',
    './exports',
    '../exports'
};

exports_dir = '';
for i = 1:length(possible_paths)
    if exist(possible_paths{i}, 'dir')
        exports_dir = possible_paths{i};
        break;
    end
end

% Check if exports directory exists
if isempty(exports_dir) || ~exist(exports_dir, 'dir')
    fprintf('❌ Exports directory not found in any of these locations:\n');
    for i = 1:length(possible_paths)
        fprintf('   - %s\n', possible_paths{i});
    end
    fprintf('\n📋 To fix this:\n');
    fprintf('1. Run your Python backend first: python app.py\n');
    fprintf('2. Train at least one model in the web interface\n');
    fprintf('3. Upload the generated JSON files to MATLAB Online\n');
    fprintf('4. Place them in a folder called "exports"\n');
    fprintf('5. Run this script again\n\n');
    fprintf('💡 Alternative: Run quick_start for basic testing without Python data\n');
    return;
end

% Get all JSON files from exports directory
json_files = dir(fullfile(exports_dir, '*.json'));

if isempty(json_files)
    fprintf('❌ No JSON files found in exports directory.\n');
    fprintf('   Please run Python backend training first.\n');
    return;
end

fprintf('   Found %d JSON files:\n', length(json_files));
for i = 1:length(json_files)
    fprintf('   - %s\n', json_files(i).name);
end

%% Step 2: Load and Parse Python Results
python_results = struct();

for i = 1:length(json_files)
    try
        % Load JSON file
        filepath = fullfile(exports_dir, json_files(i).name);
        json_data = jsondecode(fileread(filepath));
        
        % Extract model name
        model_name = json_data.model_name;
        fprintf('\n2. Processing %s results...\n', model_name);
        
        % Store results
        python_results.(lower(strrep(model_name, '-', '_'))) = json_data;
        
        % Display key metrics
        student_metrics = json_data.student_metrics;
        compression = json_data.compression_results;
        
        fprintf('   Python Results for %s:\n', model_name);
        fprintf('   - Size: %.2f MB\n', student_metrics.size_mb);
        fprintf('   - Latency: %.2f ms\n', student_metrics.latency_ms);
        fprintf('   - Accuracy: %.2f%%\n', student_metrics.accuracy);
        fprintf('   - Size Reduction: %.1f%%\n', compression.size_reduction_percent);
        fprintf('   - Latency Improvement: %.1f%%\n', compression.latency_improvement_percent);
        
    catch ME
        fprintf('❌ Error loading %s: %s\n', json_files(i).name, ME.message);
    end
end

%% Step 3: Run MATLAB KD-Pruning Tests
fprintf('\n3. Running MATLAB KD-pruning tests...\n');

% Test parameters
pruning_ratio = 0.3;
models_to_test = {'distilbert', 't5_small', 'mobilenetv2', 'resnet18'};

matlab_results = struct();

for model_idx = 1:length(models_to_test)
    model_name = models_to_test{model_idx};
    fprintf('\n   Testing MATLAB %s...\n', model_name);
    
    try
        % Create test model and data
        [teacher_model, training_data] = create_test_model_and_data(model_name);
        
        % Run MATLAB KD-pruning algorithm
        [student_model, metrics] = simplified_kd_pruning(teacher_model, training_data, pruning_ratio);
        
        % Store MATLAB results
        matlab_results.(model_name) = metrics;
        
        % Display MATLAB results
        fprintf('   MATLAB Results for %s:\n', model_name);
        fprintf('   - Size: %.2f MB\n', metrics.student.size_mb);
        fprintf('   - Latency: %.2f ms\n', metrics.student.latency_ms);
        fprintf('   - Accuracy: %.2f%%\n', metrics.student.accuracy);
        fprintf('   - Size Reduction: %.1f%%\n', metrics.improvements.size_reduction_percent);
        fprintf('   - Latency Improvement: %.1f%%\n', metrics.improvements.latency_improvement_percent);
        
    catch ME
        fprintf('❌ Error testing MATLAB %s: %s\n', model_name, ME.message);
    end
end

%% Step 4: Compare Results
fprintf('\n4. Comparing Python vs MATLAB Results...\n');

comparison_results = struct();

for model_idx = 1:length(models_to_test)
    model_name = models_to_test{model_idx};
    
    if isfield(python_results, model_name) && isfield(matlab_results, model_name)
        fprintf('\n=== %s Comparison ===\n', upper(model_name));
        
        % Get results
        py_data = python_results.(model_name);
        mat_data = matlab_results.(model_name);
        
        % Compare key metrics
        fprintf('Metric                | Python    | MATLAB    | Difference\n');
        fprintf('----------------------|-----------|------------|-----------\n');
        
        % Size comparison
        py_size = py_data.student_metrics.size_mb;
        mat_size = mat_data.student.size_mb;
        size_diff = abs(py_size - mat_size);
        fprintf('Size (MB)            | %8.2f  | %8.2f   | %8.2f\n', py_size, mat_size, size_diff);
        
        % Latency comparison
        py_latency = py_data.student_metrics.latency_ms;
        mat_latency = mat_data.student.latency_ms;
        latency_diff = abs(py_latency - mat_latency);
        fprintf('Latency (ms)         | %8.2f  | %8.2f   | %8.2f\n', py_latency, mat_latency, latency_diff);
        
        % Accuracy comparison
        py_acc = py_data.student_metrics.accuracy;
        mat_acc = mat_data.student.accuracy;
        acc_diff = abs(py_acc - mat_acc);
        fprintf('Accuracy (%%)         | %8.2f  | %8.2f   | %8.2f\n', py_acc, mat_acc, acc_diff);
        
        % Size reduction comparison
        py_size_red = py_data.compression_results.size_reduction_percent;
        mat_size_red = mat_data.improvements.size_reduction_percent;
        size_red_diff = abs(py_size_red - mat_size_red);
        fprintf('Size Reduction (%%)   | %8.1f  | %8.1f   | %8.1f\n', py_size_red, mat_size_red, size_red_diff);
        
        % Store comparison
        comparison_results.(model_name) = struct(...
            'python', py_data, ...
            'matlab', mat_data, ...
            'differences', struct(...
                'size_mb', size_diff, ...
                'latency_ms', latency_diff, ...
                'accuracy', acc_diff, ...
                'size_reduction', size_red_diff ...
            ) ...
        );
        
        % Overall assessment
        if size_diff < 5 && latency_diff < 10 && acc_diff < 5
            fprintf('✅ Results are CONSISTENT between Python and MATLAB\n');
        else
            fprintf('⚠️  Results show some DIFFERENCES between Python and MATLAB\n');
        end
        
    else
        fprintf('❌ Missing data for %s comparison\n', model_name);
    end
end

%% Step 5: Generate Comparison Report
fprintf('\n5. Generating comparison report...\n');

% Create comparison plots
create_comparison_plots(comparison_results);

% Save comparison results
save('python_matlab_comparison.mat', 'python_results', 'matlab_results', 'comparison_results');
fprintf('✅ Comparison results saved to python_matlab_comparison.mat\n');

fprintf('\n=== Comparison Complete ===\n');
fprintf('This analysis helps validate that your MATLAB implementation\n');
fprintf('produces consistent results with your Python backend.\n');

%% Helper Functions

function [teacher_model, training_data] = create_test_model_and_data(model_name)
    % Create a simple test model and data for MATLAB testing
    
    % Model parameters based on the model type
    switch lower(model_name)
        case 'distilbert'
            input_size = 10;
            hidden_size = 20;
            output_size = 2;
        case 't5_small'
            input_size = 12;
            hidden_size = 24;
            output_size = 3;
        case 'mobilenetv2'
            input_size = 8;
            hidden_size = 16;
            output_size = 2;
        case 'resnet18'
            input_size = 6;
            hidden_size = 12;
            output_size = 2;
        otherwise
            input_size = 10;
            hidden_size = 20;
            output_size = 2;
    end
    
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
    num_samples = 200;
    X = randn(num_samples, input_size);
    labels = randi([1, output_size], num_samples, 1);
    
    training_data = struct();
    for i = 1:num_samples
        training_data(i).inputs = X(i, :)';
        training_data(i).labels = labels(i);
    end
end

function create_comparison_plots(comparison_results)
    % Create comparison plots between Python and MATLAB results
    
    models = fieldnames(comparison_results);
    if isempty(models)
        fprintf('No comparison data available for plotting\n');
        return;
    end
    
    % Create figure
    figure('Name', 'Python vs MATLAB Results Comparison', 'Position', [100, 100, 1200, 800]);
    
    % Plot 1: Size Comparison
    subplot(2, 3, 1);
    py_sizes = [];
    mat_sizes = [];
    model_names = {};
    
    for i = 1:length(models)
        model = models{i};
        if isfield(comparison_results, model)
            py_sizes = [py_sizes, comparison_results.(model).python.student_metrics.size_mb];
            mat_sizes = [mat_sizes, comparison_results.(model).matlab.student.size_mb];
            model_names{end+1} = upper(model);
        end
    end
    
    x = 1:length(model_names);
    width = 0.35;
    bar(x - width/2, py_sizes, width, 'DisplayName', 'Python');
    hold on;
    bar(x + width/2, mat_sizes, width, 'DisplayName', 'MATLAB');
    xlabel('Models');
    ylabel('Size (MB)');
    title('Model Size Comparison');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', model_names);
    legend;
    grid on;
    
    % Plot 2: Latency Comparison
    subplot(2, 3, 2);
    py_latencies = [];
    mat_latencies = [];
    
    for i = 1:length(models)
        model = models{i};
        if isfield(comparison_results, model)
            py_latencies = [py_latencies, comparison_results.(model).python.student_metrics.latency_ms];
            mat_latencies = [mat_latencies, comparison_results.(model).matlab.student.latency_ms];
        end
    end
    
    bar(x - width/2, py_latencies, width, 'DisplayName', 'Python');
    hold on;
    bar(x + width/2, mat_latencies, width, 'DisplayName', 'MATLAB');
    xlabel('Models');
    ylabel('Latency (ms)');
    title('Latency Comparison');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', model_names);
    legend;
    grid on;
    
    % Plot 3: Accuracy Comparison
    subplot(2, 3, 3);
    py_accuracies = [];
    mat_accuracies = [];
    
    for i = 1:length(models)
        model = models{i};
        if isfield(comparison_results, model)
            py_accuracies = [py_accuracies, comparison_results.(model).python.student_metrics.accuracy];
            mat_accuracies = [mat_accuracies, comparison_results.(model).matlab.student.accuracy];
        end
    end
    
    bar(x - width/2, py_accuracies, width, 'DisplayName', 'Python');
    hold on;
    bar(x + width/2, mat_accuracies, width, 'DisplayName', 'MATLAB');
    xlabel('Models');
    ylabel('Accuracy (%)');
    title('Accuracy Comparison');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', model_names);
    legend;
    grid on;
    
    % Plot 4: Size Reduction Comparison
    subplot(2, 3, 4);
    py_size_red = [];
    mat_size_red = [];
    
    for i = 1:length(models)
        model = models{i};
        if isfield(comparison_results, model)
            py_size_red = [py_size_red, comparison_results.(model).python.compression_results.size_reduction_percent];
            mat_size_red = [mat_size_red, comparison_results.(model).matlab.improvements.size_reduction_percent];
        end
    end
    
    bar(x - width/2, py_size_red, width, 'DisplayName', 'Python');
    hold on;
    bar(x + width/2, mat_size_red, width, 'DisplayName', 'MATLAB');
    xlabel('Models');
    ylabel('Size Reduction (%)');
    title('Size Reduction Comparison');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', model_names);
    legend;
    grid on;
    
    % Plot 5: Differences Summary
    subplot(2, 3, 5);
    differences = [];
    for i = 1:length(models)
        model = models{i};
        if isfield(comparison_results, model)
            diff = comparison_results.(model).differences;
            differences = [differences, (diff.size_mb + diff.latency_ms/10 + diff.accuracy)/3];
        end
    end
    
    bar(x, differences);
    xlabel('Models');
    ylabel('Average Difference');
    title('Overall Differences');
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', model_names);
    grid on;
    
    % Plot 6: Correlation
    subplot(2, 3, 6);
    scatter(py_sizes, mat_sizes, 100, 'filled');
    xlabel('Python Size (MB)');
    ylabel('MATLAB Size (MB)');
    title('Size Correlation');
    grid on;
    
    % Add correlation coefficient
    if length(py_sizes) > 1
        corr_coeff = corrcoef(py_sizes, mat_sizes);
        text(0.1, 0.9, sprintf('R = %.3f', corr_coeff(1,2)), 'Units', 'normalized');
    end
    
    % Save the plot
    saveas(gcf, 'python_matlab_comparison.png');
    fprintf('✅ Comparison plots saved as python_matlab_comparison.png\n');
end
