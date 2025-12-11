import React, { useState, useEffect } from 'react';
import '../styles/TrainingComparison.css';

/**
 * TrainingComparison Component
 * 
 * Renders a side-by-side comparison of:
 * - Uploaded model metrics (after Knowledge Distillation + Pruning)
 * - Baseline reference model metrics (precomputed, no training)
 * - Both with detailed computation explanations
 * 
 * Uses high-contrast colors optimized for dark backgrounds.
 */
const TrainingComparison = ({ computationDetails, socket }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (computationDetails && computationDetails.details) {
      setIsVisible(true);
    }
  }, [computationDetails]);

  if (!isVisible || !computationDetails) {
    return null;
  }

  const { uploaded_model, baseline_reference, hardware, uploaded_model_name, baseline_model_name } =
    computationDetails;

  return (
    <div className="training-comparison-container">
      {/* Header with hardware info */}
      <div className="comparison-header">
        <h2 className="comparison-title">Model Performance Comparison</h2>
        <div className="hardware-info">
          <span className="hw-device">
            Device: <strong>{hardware?.device || 'Unknown'}</strong>
          </span>
          {hardware?.name && (
            <span className="hw-name">
              {' '}
              • <strong>{hardware.name}</strong>
            </span>
          )}
        </div>
      </div>

      {/* Side-by-side comparison cards */}
      <div className="comparison-grid">
        {/* Uploaded Model Card */}
        <div className="model-card uploaded-card">
          <div className="card-header uploaded-header">
            <h3 className="card-title">Uploaded Model</h3>
            <span className="card-subtitle">(After Knowledge Distillation + Pruning)</span>
            {uploaded_model_name && (
              <div className="model-name">{uploaded_model_name}</div>
            )}
          </div>

          <div className="metrics-container">
            {/* Accuracy */}
            <div className="metric-block">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">{uploaded_model?.accuracy?.value}</div>
              <div className="metric-explanation">
                {uploaded_model?.accuracy?.explanation}
              </div>
            </div>

            {/* F1-Score */}
            <div className="metric-block">
              <div className="metric-label">F1-Score</div>
              <div className="metric-value">{uploaded_model?.f1_score?.value}</div>
              <div className="metric-explanation">
                {uploaded_model?.f1_score?.explanation}
              </div>
            </div>

            {/* Size Reduction */}
            <div className="metric-block">
              <div className="metric-label">Model Size Reduction</div>
              <div className="metric-value">{uploaded_model?.model_size_reduction?.value}</div>
              <div className="metric-explanation">
                {uploaded_model?.model_size_reduction?.explanation}
              </div>
            </div>

            {/* Inference Latency */}
            <div className="metric-block">
              <div className="metric-label">Inference Latency</div>
              <div className="metric-value">{uploaded_model?.inference_latency?.value}</div>
              <div className="metric-explanation">
                {uploaded_model?.inference_latency?.explanation}
              </div>
            </div>

            {/* Model Complexity */}
            <div className="metric-block">
              <div className="metric-label">Model Complexity</div>
              <div className="complexity-details">
                <div className="complexity-item">
                  <span className="complexity-key">Parameters:</span>
                  <span className="complexity-val">
                    {uploaded_model?.model_complexity?.num_params?.toLocaleString() || 'N/A'}
                  </span>
                </div>
                <div className="complexity-item">
                  <span className="complexity-key">Effective Params:</span>
                  <span className="complexity-val">
                    {uploaded_model?.model_complexity?.effective_params?.toLocaleString() || 'N/A'}
                  </span>
                </div>
                <div className="complexity-item">
                  <span className="complexity-key">Sparsity:</span>
                  <span className="complexity-val">
                    {uploaded_model?.model_complexity?.sparsity_percent?.toFixed(2) || 'N/A'}%
                  </span>
                </div>
              </div>
              <div className="metric-explanation">
                {uploaded_model?.model_complexity?.explanation}
              </div>
            </div>
          </div>
        </div>

        {/* Baseline Reference Card */}
        {baseline_reference && (
          <div className="model-card baseline-card">
            <div className="card-header baseline-header">
              <h3 className="card-title">Baseline Reference</h3>
              <span className="card-subtitle">(Pretrained, No Training)</span>
              {baseline_model_name && (
                <div className="model-name">{baseline_model_name}</div>
              )}
            </div>

            <div className="metrics-container">
              {/* Accuracy */}
              <div className="metric-block">
                <div className="metric-label">Accuracy</div>
                <div className="metric-value">{baseline_reference?.accuracy?.value}</div>
                <div className="metric-explanation">
                  {baseline_reference?.accuracy?.explanation}
                </div>
              </div>

              {/* F1-Score */}
              <div className="metric-block">
                <div className="metric-label">F1-Score</div>
                <div className="metric-value">{baseline_reference?.f1_score?.value}</div>
                <div className="metric-explanation">
                  {baseline_reference?.f1_score?.explanation}
                </div>
              </div>

              {/* Model Size */}
              <div className="metric-block">
                <div className="metric-label">Model Size</div>
                <div className="metric-value">{baseline_reference?.model_size?.value}</div>
                <div className="metric-explanation">
                  {baseline_reference?.model_size?.explanation}
                </div>
              </div>

              {/* Inference Latency */}
              <div className="metric-block">
                <div className="metric-label">Inference Latency</div>
                <div className="metric-value">{baseline_reference?.inference_latency?.value}</div>
                <div className="metric-explanation">
                  {baseline_reference?.inference_latency?.explanation}
                </div>
              </div>

              {/* Model Complexity */}
              <div className="metric-block">
                <div className="metric-label">Model Complexity</div>
                <div className="complexity-details">
                  <div className="complexity-item">
                    <span className="complexity-key">Parameters:</span>
                    <span className="complexity-val">
                      {baseline_reference?.model_complexity?.num_params?.toLocaleString() || 'N/A'}
                    </span>
                  </div>
                </div>
                <div className="metric-explanation">
                  {baseline_reference?.model_complexity?.explanation}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingComparison;
