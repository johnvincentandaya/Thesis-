import React, { useState, useEffect } from 'react';
import { Card, Table, Tooltip, Button, Modal, Typography, Space, Tag, Alert } from 'antd';
import { InfoCircleOutlined, QuestionCircleOutlined, TrophyOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const MetricsDisplay = ({ beforeMetrics, afterMetrics, loading }) => {
  const [explanations, setExplanations] = useState({});
  const [showExplanations, setShowExplanations] = useState(false);

  useEffect(() => {
    // Fetch metrics explanations
    fetch('http://localhost:5000/metrics_explanations')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setExplanations(data.explanations);
        }
      })
      .catch(err => console.error('Failed to fetch explanations:', err));
  }, []);

  const getMetricColor = (metric, value) => {
    if (metric.includes('Latency') || metric.includes('Size') || metric.includes('Parameters') || metric.includes('FLOPs')) {
      // For these metrics, lower is better
      return 'green';
    } else {
      // For accuracy, F1-score, etc., higher is better
      const numValue = parseFloat(value);
      if (numValue >= 80) return 'green';
      if (numValue >= 60) return 'orange';
      return 'red';
    }
  };

  const getComplexityColor = (complexity) => {
    switch (complexity.toLowerCase()) {
      case 'low': return 'green';
      case 'medium': return 'orange';
      case 'high': return 'red';
      default: return 'blue';
    }
  };

  const calculateSizeReduction = () => {
    if (!beforeMetrics || !afterMetrics) return 0;
    const reduction = ((beforeMetrics.size_mb - afterMetrics.size_mb) / beforeMetrics.size_mb) * 100;
    return Math.round(reduction * 10) / 10; // Round to 1 decimal place
  };

  const renderMetricsTable = () => {
    if (!beforeMetrics || !afterMetrics) return null;

    const sizeReduction = calculateSizeReduction();

    const metricsData = [
      {
        key: 'f1_score',
        metric: 'F1-Score',
        before: `${beforeMetrics.f1_score}%`,
        after: `${afterMetrics.f1_score}%`,
        improvement: ((afterMetrics.f1_score - beforeMetrics.f1_score) / beforeMetrics.f1_score * 100).toFixed(1)
      },
      {
        key: 'accuracy',
        metric: 'Accuracy',
        before: `${beforeMetrics.accuracy}%`,
        after: `${afterMetrics.accuracy}%`,
        improvement: ((afterMetrics.accuracy - beforeMetrics.accuracy) / beforeMetrics.accuracy * 100).toFixed(1)
      },
      {
        key: 'precision',
        metric: 'Precision',
        before: `${beforeMetrics.precision}%`,
        after: `${afterMetrics.precision}%`,
        improvement: ((afterMetrics.precision - beforeMetrics.precision) / beforeMetrics.precision * 100).toFixed(1)
      },
      {
        key: 'recall',
        metric: 'Recall',
        before: `${beforeMetrics.recall}%`,
        after: `${afterMetrics.recall}%`,
        improvement: ((afterMetrics.recall - beforeMetrics.recall) / beforeMetrics.recall * 100).toFixed(1)
      },
      {
        key: 'inference_latency',
        metric: 'Inference Latency',
        before: `${beforeMetrics.inference_latency_ms}ms`,
        after: `${afterMetrics.inference_latency_ms}ms`,
        improvement: ((beforeMetrics.inference_latency_ms - afterMetrics.inference_latency_ms) / beforeMetrics.inference_latency_ms * 100).toFixed(1)
      },
      {
        key: 'model_complexity',
        metric: 'Model Complexity',
        before: beforeMetrics.model_complexity,
        after: afterMetrics.model_complexity,
        improvement: 'N/A'
      },
      {
        key: 'parameter_count',
        metric: 'Parameter Count',
        before: beforeMetrics.parameter_count.toLocaleString(),
        after: afterMetrics.parameter_count.toLocaleString(),
        improvement: ((beforeMetrics.parameter_count - afterMetrics.parameter_count) / beforeMetrics.parameter_count * 100).toFixed(1)
      },
      {
        key: 'model_size',
        metric: 'Model Size',
        before: `${beforeMetrics.size_mb}MB`,
        after: `${afterMetrics.size_mb}MB`,
        improvement: ((beforeMetrics.size_mb - afterMetrics.size_mb) / beforeMetrics.size_mb * 100).toFixed(1)
      },
      {
        key: 'flops',
        metric: 'FLOPs',
        before: `${beforeMetrics.flops_millions}M`,
        after: `${afterMetrics.flops_millions}M`,
        improvement: ((beforeMetrics.flops_millions - afterMetrics.flops_millions) / beforeMetrics.flops_millions * 100).toFixed(1)
      },
      {
        key: 'size_reduction',
        metric: 'Model Size Reduction',
        before: 'N/A',
        after: `${sizeReduction}%`,
        improvement: 'N/A'
      }
    ];

    const columns = [
      {
        title: 'Metric',
        dataIndex: 'metric',
        key: 'metric',
        render: (text, record) => (
          <Space>
            <Text strong>{text}</Text>
            {explanations[record.key] && (
              <Tooltip title="Click for explanation">
                <QuestionCircleOutlined 
                  style={{ color: '#1890ff', cursor: 'pointer' }}
                  onClick={() => setShowExplanations(record.key)}
                />
              </Tooltip>
            )}
          </Space>
        )
      },
      {
        title: 'Before (Teacher)',
        dataIndex: 'before',
        key: 'before',
        render: (text, record) => {
          if (text === 'N/A') return <Text type="secondary">N/A</Text>;
          const color = record.key === 'model_complexity' 
            ? getComplexityColor(text)
            : getMetricColor(record.key, text);
          return <Tag color={color}>{text}</Tag>;
        }
      },
      {
        title: 'After (Student)',
        dataIndex: 'after',
        key: 'after',
        render: (text, record) => {
          const color = record.key === 'model_complexity' 
            ? getComplexityColor(text)
            : getMetricColor(record.key, text);
          return <Tag color={color}>{text}</Tag>;
        }
      },
      {
        title: 'Improvement',
        dataIndex: 'improvement',
        key: 'improvement',
        render: (text, record) => {
          if (text === 'N/A') return <Text type="secondary">N/A</Text>;
          const numValue = parseFloat(text);
          const color = numValue > 0 ? 'green' : numValue < 0 ? 'red' : 'blue';
          const sign = numValue > 0 ? '+' : '';
          return <Tag color={color}>{sign}{text}%</Tag>;
        }
      }
    ];

    return (
      <Table 
        columns={columns} 
        dataSource={metricsData} 
        pagination={false}
        loading={loading}
        size="middle"
        rowClassName={(record) => {
          if (record.key === 'size_reduction') return 'highlight-row';
          return '';
        }}
      />
    );
  };

  const renderExplanationModal = () => {
    if (!showExplanations || !explanations[showExplanations]) return null;

    const explanation = explanations[showExplanations];

    return (
      <Modal
        title={
          <Space>
            <TrophyOutlined style={{ color: '#1890ff' }} />
            {showExplanations.replace('_', ' ').toUpperCase()} - Explanation
          </Space>
        }
        open={!!showExplanations}
        onCancel={() => setShowExplanations(false)}
        footer={[
          <Button key="close" onClick={() => setShowExplanations(false)}>
            Close
          </Button>
        ]}
        width={700}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div>
            <Title level={5}>Description</Title>
            <Paragraph>{explanation.description}</Paragraph>
          </div>
          
          <div>
            <Title level={5}>Formula</Title>
            <Paragraph code>{explanation.formula}</Paragraph>
          </div>
          
          <div>
            <Title level={5}>Range</Title>
            <Paragraph>{explanation.range}</Paragraph>
          </div>
          
          <div>
            <Title level={5}>Interpretation</Title>
            <Paragraph>{explanation.interpretation}</Paragraph>
          </div>

          {showExplanations === 'size_reduction' && (
            <Alert
              message="Why Model Size Reduction Matters"
              description="Smaller models are easier to deploy on mobile devices, require less storage space, and can be transmitted faster over networks. This is crucial for real-world applications where resources are limited."
              type="info"
              showIcon
            />
          )}
        </Space>
      </Modal>
    );
  };

  const renderCompressionSummary = () => {
    if (!beforeMetrics || !afterMetrics) return null;

    const sizeReduction = calculateSizeReduction();
    const paramReduction = ((beforeMetrics.parameter_count - afterMetrics.parameter_count) / beforeMetrics.parameter_count * 100).toFixed(1);
    const latencyImprovement = ((beforeMetrics.inference_latency_ms - afterMetrics.inference_latency_ms) / beforeMetrics.inference_latency_ms * 100).toFixed(1);

    return (
      <Card className="mt-4" style={{ backgroundColor: '#f6ffed', borderColor: '#b7eb8f' }}>
        <Card.Body>
          <Title level={4} style={{ color: '#52c41a' }}>
            <TrophyOutlined className="me-2" />
            Compression Results Summary
          </Title>
          <div className="row text-center">
            <div className="col-md-4">
              <div className="h3 text-success">{sizeReduction}%</div>
              <div className="text-muted">Size Reduction</div>
            </div>
            <div className="col-md-4">
              <div className="h3 text-primary">{paramReduction}%</div>
              <div className="text-muted">Parameter Reduction</div>
            </div>
            <div className="col-md-4">
              <div className="h3 text-info">{latencyImprovement}%</div>
              <div className="text-muted">Faster Inference</div>
            </div>
          </div>
          <Alert
            message="Compression Success!"
            description="The model has been successfully compressed while maintaining good performance. This demonstrates the effectiveness of Knowledge Distillation and Pruning techniques."
            type="success"
            showIcon
            className="mt-3"
          />
        </Card.Body>
      </Card>
    );
  };

  return (
    <Card 
      title={
        <Space>
          <Title level={4} style={{ margin: 0 }}>Model Performance Metrics</Title>
          <Tooltip title="Comprehensive metrics showing model performance before and after Knowledge Distillation and Pruning">
            <InfoCircleOutlined style={{ color: '#1890ff' }} />
          </Tooltip>
        </Space>
      }
      style={{ marginTop: 24, marginBottom: 24 }}
    >
      {renderMetricsTable()}
      {renderCompressionSummary()}
      {renderExplanationModal()}
    </Card>
  );
};

export default MetricsDisplay; 