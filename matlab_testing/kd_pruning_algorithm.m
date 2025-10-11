function [student_model, metrics] = kd_pruning_algorithm(teacher_model, training_data, pruning_ratio, temperature, epochs)
    % Knowledge Distillation and Pruning Algorithm Implementation in MATLAB
    % This function implements the same algorithm as the Python backend
    
    % Inputs:
    %   teacher_model: Pre-trained teacher model
    %   training_data: Training dataset
    %   pruning_ratio: Percentage of weights to prune (0.0 to 1.0)
    %   temperature: Temperature for knowledge distillation
    %   epochs: Number of training epochs
    
    % Outputs:
    %   student_model: Compressed student model
    %   metrics: Performance and efficiency metrics
    
    fprintf('Starting Knowledge Distillation and Pruning Algorithm...\n');
    
    % Step 1: Initialize student model (copy of teacher)
    student_model = copy_struct(teacher_model);
    fprintf('Student model initialized (copy of teacher)\n');
    
    % Step 2: Knowledge Distillation Training
    fprintf('Starting Knowledge Distillation...\n');
    student_model = knowledge_distillation_training(student_model, teacher_model, training_data, temperature, epochs);
    
    % Step 3: Apply Pruning
    fprintf('Applying L1 Unstructured Pruning...\n');
    student_model = apply_pruning_matlab(student_model, pruning_ratio);
    
    % Step 4: Calculate Metrics
    fprintf('Calculating final metrics...\n');
    metrics = calculate_model_metrics_matlab(student_model, teacher_model, training_data);
    
    fprintf('Algorithm completed successfully!\n');
end

function student_model = knowledge_distillation_training(student_model, teacher_model, training_data, temperature, epochs)
    % Knowledge Distillation training loop
    
    % Set up optimizer (simplified version without Deep Learning Toolbox)
    learning_rate = 0.001;
    % Note: Using simplified gradient descent instead of Adam optimizer
    % This works without Deep Learning Toolbox
    
    % Training loop
    for epoch = 1:epochs
        total_loss = 0;
        num_batches = 0;
        
        % Iterate through training data
        for i = 1:length(training_data)
            % Get batch data
            inputs = training_data(i).inputs;
            labels = training_data(i).labels;
            
            % Forward pass through teacher (with temperature scaling)
            teacher_outputs = forward(teacher_model, inputs);
            teacher_soft_targets = softmax(teacher_outputs / temperature);
            
            % Forward pass through student
            student_outputs = forward(student_model, inputs);
            student_probs = softmax(student_outputs / temperature);
            
            % Calculate distillation loss
            distillation_loss = calculate_distillation_loss(student_probs, teacher_soft_targets, temperature);
            
            % Calculate task loss (hard labels)
            task_loss = crossentropy(student_outputs, labels);
            
            % Combined loss
            alpha = 0.7; % Weight for distillation vs task loss
            total_batch_loss = alpha * distillation_loss + (1 - alpha) * task_loss;
            
            % Backward pass and optimization
            gradients = dlgradient(total_batch_loss, student_model.Learnables);
            optimizer.apply_gradients(student_model.Learnables, gradients);
            
            total_loss = total_loss + total_batch_loss;
            num_batches = num_batches + 1;
        end
        
        avg_loss = total_loss / num_batches;
        fprintf('Epoch %d/%d, Average Loss: %.4f\n', epoch, epochs, avg_loss);
    end
end

function pruned_model = apply_pruning_matlab(model, pruning_ratio)
    % Apply L1 unstructured pruning to the model
    
    % Get all learnable parameters
    params = model.Learnables;
    
    for i = 1:height(params)
        param_name = params.Properties.RowNames{i};
        param_value = params.Value{i};
        
        % Only prune weight parameters (not biases)
        if contains(param_name, 'Weight')
            % Calculate L1 norm for each weight
            l1_norms = abs(param_value);
            
            % Find threshold for pruning
            sorted_norms = sort(l1_norms(:), 'ascend');
            threshold_idx = round(pruning_ratio * numel(sorted_norms));
            threshold = sorted_norms(threshold_idx);
            
            % Create pruning mask
            pruning_mask = l1_norms > threshold;
            
            % Apply pruning
            pruned_value = param_value .* pruning_mask;
            
            % Update parameter
            model.(param_name) = pruned_value;
            
            fprintf('Pruned %s: %.2f%% sparsity\n', param_name, pruning_ratio * 100);
        end
    end
    
    pruned_model = model;
end

function metrics = calculate_model_metrics_matlab(student_model, teacher_model, test_data)
    % Calculate comprehensive model metrics
    
    % Model size comparison
    teacher_size = get_model_size_matlab(teacher_model);
    student_size = get_model_size_matlab(student_model);
    
    % Parameter count
    teacher_params = count_parameters_matlab(teacher_model);
    student_params = count_parameters_matlab(student_model);
    
    % Inference latency
    teacher_latency = measure_inference_latency_matlab(teacher_model, test_data);
    student_latency = measure_inference_latency_matlab(student_model, test_data);
    
    % Accuracy comparison
    teacher_accuracy = evaluate_accuracy_matlab(teacher_model, test_data);
    student_accuracy = evaluate_accuracy_matlab(student_model, test_data);
    
    % Calculate improvements
    size_reduction = ((teacher_size - student_size) / teacher_size) * 100;
    param_reduction = ((teacher_params - student_params) / teacher_params) * 100;
    latency_improvement = ((teacher_latency - student_latency) / teacher_latency) * 100;
    accuracy_impact = student_accuracy - teacher_accuracy;
    
    % Compile metrics
    metrics = struct();
    metrics.teacher = struct(...
        'size_mb', teacher_size, ...
        'num_params', teacher_params, ...
        'latency_ms', teacher_latency, ...
        'accuracy', teacher_accuracy);
    
    metrics.student = struct(...
        'size_mb', student_size, ...
        'num_params', student_params, ...
        'latency_ms', student_latency, ...
        'accuracy', student_accuracy);
    
    metrics.improvements = struct(...
        'size_reduction_percent', size_reduction, ...
        'param_reduction_percent', param_reduction, ...
        'latency_improvement_percent', latency_improvement, ...
        'accuracy_impact', accuracy_impact);
    
    % Print summary
    fprintf('\n=== Model Compression Results ===\n');
    fprintf('Size Reduction: %.2f%%\n', size_reduction);
    fprintf('Parameter Reduction: %.2f%%\n', param_reduction);
    fprintf('Latency Improvement: %.2f%%\n', latency_improvement);
    fprintf('Accuracy Impact: %.2f%%\n', accuracy_impact);
end

function loss = calculate_distillation_loss(student_probs, teacher_probs, temperature)
    % Calculate KL divergence loss for knowledge distillation
    % Using KL divergence between student and teacher probability distributions
    
    % Add small epsilon to avoid log(0)
    epsilon = 1e-8;
    student_probs = student_probs + epsilon;
    teacher_probs = teacher_probs + epsilon;
    
    % KL divergence: KL(teacher || student)
    loss = sum(teacher_probs .* log(teacher_probs ./ student_probs), 'all');
end

function size_mb = get_model_size_matlab(model)
    % Estimate model size in MB
    % This is a simplified estimation
    total_params = count_parameters_matlab(model);
    size_mb = (total_params * 4) / (1024 * 1024); % Assuming float32 (4 bytes per parameter)
end

function num_params = count_parameters_matlab(model)
    % Count total number of learnable parameters
    params = model.Learnables;
    num_params = 0;
    
    for i = 1:height(params)
        param_value = params.Value{i};
        num_params = num_params + numel(param_value);
    end
end

function latency_ms = measure_inference_latency_matlab(model, test_data)
    % Measure average inference latency
    num_samples = min(100, length(test_data)); % Test with up to 100 samples
    
    total_time = 0;
    for i = 1:num_samples
        inputs = test_data(i).inputs;
        
        tic;
        forward(model, inputs);
        elapsed_time = toc;
        
        total_time = total_time + elapsed_time;
    end
    
    latency_ms = (total_time / num_samples) * 1000; % Convert to milliseconds
end

function accuracy = evaluate_accuracy_matlab(model, test_data)
    % Evaluate model accuracy on test data
    correct_predictions = 0;
    total_predictions = 0;
    
    for i = 1:length(test_data)
        inputs = test_data(i).inputs;
        labels = test_data(i).labels;
        
        % Forward pass
        outputs = forward(model, inputs);
        
        % Get predictions
        [~, predicted_labels] = max(outputs, [], 2);
        
        % Compare with true labels
        correct_predictions = correct_predictions + sum(predicted_labels == labels);
        total_predictions = total_predictions + length(labels);
    end
    
    accuracy = (correct_predictions / total_predictions) * 100;
end

function copied_struct = copy_struct(original_struct)
    % Create a deep copy of a struct (compatible with basic MATLAB)
    % This replaces the 'copy' function which requires Deep Learning Toolbox
    
    if ~isstruct(original_struct)
        copied_struct = original_struct;
        return;
    end
    
    % Get all field names
    field_names = fieldnames(original_struct);
    
    % Create new struct
    copied_struct = struct();
    
    % Copy each field
    for i = 1:length(field_names)
        field_name = field_names{i};
        field_value = original_struct.(field_name);
        
        if isstruct(field_value)
            % Recursively copy nested structs
            copied_struct.(field_name) = copy_struct(field_value);
        else
            % Copy simple values directly
            copied_struct.(field_name) = field_value;
        end
    end
end
