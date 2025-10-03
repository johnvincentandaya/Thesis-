%% MATLAB Backend Integration for KD-Pruning Simulator
% This script demonstrates how to fetch metrics from the Flask backend
% and create comparison plots for the KD-Pruning simulation results.

function matlab_backend_integration()
    % Configuration
    backend_url = 'http://127.0.0.1:5001';
    
    fprintf('KD-Pruning Simulator - MATLAB Integration\n');
    fprintf('==========================================\n\n');
    
    % Check if backend is running
    if ~check_backend_status(backend_url)
        fprintf('Error: Backend server is not running.\n');
        fprintf('Please start the Flask backend first: python app.py\n');
        return;
    end
    
    % Fetch metrics from backend
    fprintf('Fetching metrics from backend...\n');
    metrics_data = fetch_metrics(backend_url);
    
    if isempty(metrics_data)
        fprintf('Error: Could not fetch metrics. Make sure a model has been trained.\n');
        return;
    end
    
    % Create comparison plots
    fprintf('Creating comparison plots...\n');
    create_effectiveness_plots(metrics_data);
    create_efficiency_plots(metrics_data);
    create_compression_plots(metrics_data);
    create_complexity_plots(metrics_data);
    
    fprintf('All plots created successfully!\n');
end

function status = check_backend_status(url)
    % Check if the backend server is running
    try
        response = webread([url '/test']);
        status = true;
        fprintf('✓ Backend server is running\n');
    catch
        status = false;
        fprintf('✗ Backend server is not responding\n');
    end
end

function metrics_data = fetch_metrics(url)
    % Fetch comprehensive metrics from the backend
    try
        response = webread([url '/matlab_metrics']);
        
        if response.success
            metrics_data = response.matlab_data;
            fprintf('✓ Metrics fetched successfully\n');
            fprintf('Model: %s\n', response.model_name);
            fprintf('Accuracy Retention: %s\n', response.summary.accuracy_retention);
            fprintf('Size Reduction: %s\n', response.summary.size_reduction);
            fprintf('Speed Improvement: %s\n', response.summary.speed_improvement);
            fprintf('Parameter Reduction: %s\n', response.summary.parameter_reduction);
        else
            fprintf('Error: %s\n', response.error);
            metrics_data = [];
        end
    catch ME
        fprintf('Error fetching metrics: %s\n', ME.message);
        metrics_data = [];
    end
end

function create_effectiveness_plots(data)
    % Create plots for effectiveness metrics (Accuracy, Precision, Recall, F1)
    if isempty(data.effectiveness)
        fprintf('No effectiveness data available\n');
        return;
    end
    
    figure('Name', 'Effectiveness Metrics Comparison', 'Position', [100, 100, 1200, 800]);
    
    metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
    before_values = [];
    after_values = [];
    
    for i = 1:length(data.effectiveness)
        before_str = data.effectiveness{i}.before;
        after_str = data.effectiveness{i}.after;
        
        % Extract numeric values (remove % symbol)
        before_val = str2double(regexprep(before_str, '%', ''));
        after_val = str2double(regexprep(after_str, '%', ''));
        
        before_values(i) = before_val;
        after_values(i) = after_val;
    end
    
    % Create grouped bar chart
    subplot(2,2,1);
    x = 1:length(metrics);
    width = 0.35;
    bar(x - width/2, before_values, width, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'Teacher Model');
    hold on;
    bar(x + width/2, after_values, width, 'FaceColor', [0.8, 0.4, 0.2], 'DisplayName', 'Student Model');
    set(gca, 'XTickLabel', metrics);
    ylabel('Percentage (%)');
    title('Effectiveness Metrics: Teacher vs Student');
    legend('Location', 'best');
    grid on;
    
    % Create difference plot
    subplot(2,2,2);
    differences = before_values - after_values;
    bar(x, differences, 'FaceColor', [0.6, 0.2, 0.2]);
    set(gca, 'XTickLabel', metrics);
    ylabel('Accuracy Drop (%)');
    title('Accuracy Drop After Compression');
    grid on;
    
    % Create retention plot
    subplot(2,2,3);
    retention = (after_values ./ before_values) * 100;
    bar(x, retention, 'FaceColor', [0.2, 0.8, 0.2]);
    set(gca, 'XTickLabel', metrics);
    ylabel('Retention (%)');
    title('Performance Retention After Compression');
    ylim([0, 100]);
    grid on;
    
    % Create radar chart
    subplot(2,2,4);
    theta = linspace(0, 2*pi, length(metrics)+1);
    before_radar = [before_values, before_values(1)];
    after_radar = [after_values, after_values(1)];
    
    polarplot(theta, before_radar, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    polarplot(theta, after_radar, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
    title('Effectiveness Radar Chart');
    legend('Teacher Model', 'Student Model', 'Location', 'best');
    thetaticks(rad2deg(theta(1:end-1)));
    thetaticklabels(metrics);
end

function create_efficiency_plots(data)
    % Create plots for efficiency metrics (Latency, RAM Usage, Model Size)
    if isempty(data.efficiency)
        fprintf('No efficiency data available\n');
        return;
    end
    
    figure('Name', 'Efficiency Metrics Comparison', 'Position', [200, 200, 1200, 600]);
    
    metrics = {'Latency (ms)', 'RAM Usage (MB)', 'Model Size (MB)'};
    before_values = [];
    after_values = [];
    
    for i = 1:length(data.efficiency)
        before_str = data.efficiency{i}.before;
        after_str = data.efficiency{i}.after;
        
        % Extract numeric values
        before_val = str2double(before_str);
        after_val = str2double(after_str);
        
        before_values(i) = before_val;
        after_values(i) = after_val;
    end
    
    % Create grouped bar chart
    subplot(1,2,1);
    x = 1:length(metrics);
    width = 0.35;
    bar(x - width/2, before_values, width, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'Before');
    hold on;
    bar(x + width/2, after_values, width, 'FaceColor', [0.8, 0.4, 0.2], 'DisplayName', 'After');
    set(gca, 'XTickLabel', metrics);
    ylabel('Value');
    title('Efficiency Metrics: Before vs After');
    legend('Location', 'best');
    grid on;
    
    % Create improvement plot
    subplot(1,2,2);
    improvements = ((before_values - after_values) ./ before_values) * 100;
    bar(x, improvements, 'FaceColor', [0.2, 0.8, 0.2]);
    set(gca, 'XTickLabel', metrics);
    ylabel('Improvement (%)');
    title('Efficiency Improvements');
    grid on;
end

function create_compression_plots(data)
    % Create plots for compression metrics
    if isempty(data.compression)
        fprintf('No compression data available\n');
        return;
    end
    
    figure('Name', 'Compression Metrics', 'Position', [300, 300, 1200, 800]);
    
    % Extract compression data
    param_before = str2double(regexprep(data.compression{1}.before, ',', ''));
    param_after = str2double(regexprep(data.compression{1}.after, ',', ''));
    
    layers_before = str2double(data.compression{2}.before);
    layers_after = str2double(data.compression{2}.after);
    
    comp_ratio = str2double(data.compression{3}.after);
    acc_drop = str2double(data.compression{4}.after);
    size_red = str2double(data.compression{5}.after);
    
    % Parameters comparison
    subplot(2,2,1);
    bar([1, 2], [param_before, param_after], 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', {'Before', 'After'});
    ylabel('Number of Parameters');
    title('Parameter Count Reduction');
    grid on;
    
    % Layers comparison
    subplot(2,2,2);
    bar([1, 2], [layers_before, layers_after], 'FaceColor', [0.8, 0.4, 0.2]);
    set(gca, 'XTickLabel', {'Before', 'After'});
    ylabel('Number of Layers');
    title('Layer Count Reduction');
    grid on;
    
    % Compression ratio pie chart
    subplot(2,2,3);
    pie([comp_ratio-1, 1], {'Compressed', 'Original'});
    title(sprintf('Compression Ratio: %.2fx', comp_ratio));
    
    % Trade-off analysis
    subplot(2,2,4);
    scatter(acc_drop, size_red, 100, 'filled', 'MarkerFaceColor', [0.6, 0.2, 0.2]);
    xlabel('Accuracy Drop (%)');
    ylabel('Size Reduction (%)');
    title('Accuracy vs Size Trade-off');
    grid on;
end

function create_complexity_plots(data)
    % Create plots for complexity metrics
    if isempty(data.complexity)
        fprintf('No complexity data available\n');
        return;
    end
    
    figure('Name', 'Complexity Analysis', 'Position', [400, 400, 1000, 600]);
    
    % Extract complexity data
    time_before = data.complexity{1}.before;
    time_after = data.complexity{1}.after;
    space_before = data.complexity{2}.before;
    space_after = data.complexity{2}.after;
    
    % Create text-based comparison
    subplot(1,2,1);
    text(0.1, 0.8, 'Time Complexity', 'FontSize', 14, 'FontWeight', 'bold');
    text(0.1, 0.6, ['Before: ' time_before], 'FontSize', 12);
    text(0.1, 0.4, ['After: ' time_after], 'FontSize', 12);
    text(0.1, 0.2, 'Improvement: Reduced computational complexity', 'FontSize', 10, 'Color', 'green');
    axis off;
    title('Time Complexity Analysis');
    
    subplot(1,2,2);
    text(0.1, 0.8, 'Space Complexity', 'FontSize', 14, 'FontWeight', 'bold');
    text(0.1, 0.6, ['Before: ' space_before], 'FontSize', 12);
    text(0.1, 0.4, ['After: ' space_after], 'FontSize', 12);
    text(0.1, 0.2, 'Improvement: Reduced memory requirements', 'FontSize', 10, 'Color', 'green');
    axis off;
    title('Space Complexity Analysis');
end

% Run the integration
matlab_backend_integration();
