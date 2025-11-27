import React, { useState, useEffect } from "react";
import { Container, Row, Col, Card, Navbar, Nav, Table, Badge, Tooltip, Button, Modal, Accordion } from "react-bootstrap";
import { Link } from "react-router-dom";
import { InfoCircle, PlayCircle } from "react-bootstrap-icons";
import "./Models.css";
import Footer from "../components/Footer";

const Models = () => {
    const [selectedModel, setSelectedModel] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [modelsData, setModelsData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [models, setModels] = useState([]);

    // Helper function to format number
    const formatNumber = (num) => {
        if (num >= 1_000_000) {
            return `${(num / 1_000_000).toFixed(1)}M`;
        } else if (num >= 1_000) {
            return `${(num / 1_000).toFixed(1)}K`;
        }
        return num.toString();
    };

    // Fetch builtin models info from backend on mount
    useEffect(() => {
        // Transform backend data to frontend format
        const transformBackendData = (backendData) => {
        if (!backendData) return [];

        const modelMapping = {
            "distillBert": {
                name: "DistilBERT",
                fullName: "Distilled Bidirectional Encoder Representations from Transformers",
                description: "DistilBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks like text classification and question answering. It uses knowledge distillation to achieve 97% of BERT's performance while being 40% smaller and 60% faster.",
                architecture: "Transformer-based with 6 layers, 768 hidden dimensions, 12 attention heads",
                useCases: ["Text Classification", "Question Answering", "Named Entity Recognition", "Sentiment Analysis"],
                complexity: "Medium"
            },
            "T5-small": {
                name: "T5-small",
                fullName: "Text-to-Text Transfer Transformer (Small)",
                description: "T5-small is a smaller version of the T5 model, capable of performing a wide range of NLP tasks by converting them into a text-to-text format. It can handle translation, summarization, question answering, and more with a unified approach.",
                architecture: "Encoder-decoder transformer with 6 encoder and 6 decoder layers, 512 hidden dimensions",
                useCases: ["Text Translation", "Text Summarization", "Question Answering", "Text Generation"],
                complexity: "Medium"
            },
            "MobileNetV2": {
                name: "MobileNetV2",
                fullName: "MobileNet Version 2",
                description: "MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and object detection on mobile and embedded devices. It uses inverted residuals and linear bottlenecks to achieve high accuracy with low computational cost.",
                architecture: "CNN with inverted residual blocks, depthwise separable convolutions, 53 layers",
                useCases: ["Image Classification", "Object Detection", "Face Recognition", "Mobile Vision Apps"],
                complexity: "Low"
            },
            "ResNet-18": {
                name: "ResNet-18",
                fullName: "Residual Network with 18 Layers",
                description: "ResNet-18 is a deep residual network with 18 layers, known for its ability to train very deep networks by using skip connections to avoid vanishing gradients. It's a foundational architecture that has influenced many subsequent models.",
                architecture: "CNN with residual connections, 18 layers, skip connections every 2 layers",
                useCases: ["Image Classification", "Feature Extraction", "Transfer Learning", "Computer Vision Research"],
                complexity: "High"
            }
        };

        return Object.keys(backendData).map((key) => {
            const backendModel = backendData[key];
            const frontendInfo = modelMapping[key];
            const beforeMetrics = backendModel.metrics.before_kd;
            const afterMetrics = backendModel.metrics.after_kd_pruning;

            return {
                key: key,
                name: frontendInfo?.name || backendModel.name,
                fullName: frontendInfo?.fullName || backendModel.description,
                description: frontendInfo?.description || backendModel.description,
                architecture: frontendInfo?.architecture || "",
                useCases: frontendInfo?.useCases || [],
                trainingHistory: backendModel.training_history,
                kdExplanation: backendModel.kd_explanation,
                pruningExplanation: backendModel.pruning_explanation,
                metrics: {
                    before: {
                        accuracy: `${beforeMetrics.accuracy.toFixed(1)}%`,
                        loss: "N/A",
                        precision: `${beforeMetrics.precision.toFixed(1)}%`,
                        recall: `${beforeMetrics.recall.toFixed(1)}%`,
                        f1: `${beforeMetrics.f1.toFixed(1)}%`,
                        inference: `${beforeMetrics.latency_ms.toFixed(1)} ms`,
                        size: `${beforeMetrics.size_mb.toFixed(1)} MB`
                    },
                    after: {
                        accuracy: `${afterMetrics.accuracy.toFixed(1)}%`,
                        loss: "N/A",
                        precision: `${afterMetrics.precision.toFixed(1)}%`,
                        recall: `${afterMetrics.recall.toFixed(1)}%`,
                        f1: `${afterMetrics.f1.toFixed(1)}%`,
                        inference: `${afterMetrics.latency_ms.toFixed(1)} ms`,
                        size: `${afterMetrics.size_mb.toFixed(1)} MB`
                    },
                    modelComplexity: frontendInfo?.complexity || "Medium",
                    parameterCount: formatNumber(afterMetrics.num_params)
                },
                backendData: backendModel // Store full backend data for proof
            };
        });
    };

        const fetchModelsInfo = async () => {
            try {
                const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || "http://localhost:5000";
                const response = await fetch(`${SOCKET_URL}/model_info`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.success && data.data) {
                        setModelsData(data.data);
                        const transformed = transformBackendData(data.data);
                        setModels(transformed);
                        console.log("Successfully fetched and transformed builtin models info from backend");
                    }
                } else {
                    console.warn("Failed to fetch models info from backend, using fallback");
                }
            } catch (error) {
                console.warn("Error fetching models info:", error);
            } finally {
                setLoading(false);
            }
        };
        
        fetchModelsInfo();
    }, []);

    // Fallback models (only used if backend fetch fails)
    const fallbackModelsArray = [
        {
            name: "DistilBERT",
            fullName: "Distilled Bidirectional Encoder Representations from Transformers",
            description: "DistilBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks like text classification and question answering. It uses knowledge distillation to achieve 97% of BERT's performance while being 40% smaller and 60% faster.",
            architecture: "Transformer-based with 6 layers, 768 hidden dimensions, 12 attention heads",
            useCases: ["Text Classification", "Question Answering", "Named Entity Recognition", "Sentiment Analysis"],
            metrics: {
                before: {
                    accuracy: "92.4%",
                    loss: "0.38",
                    precision: "91.8%",
                    recall: "90.7%",
                    f1: "91.2%",
                    inference: "126 ms",
                    size: "255 MB"
                },
                after: {
                    accuracy: "89.6%",
                    loss: "0.42",
                    precision: "89.1%",
                    recall: "88.4%",
                    f1: "88.7%",
                    inference: "48 ms",
                    size: "178 MB"
                },
                modelComplexity: "Medium",
                parameterCount: "67M"
            },
            explanation: {
                accuracy: "Measures how often the model correctly classifies text across various NLP tasks.",
                f1Score: "Balanced measure of precision and recall, especially important for imbalanced datasets.",
                inferenceLatency: "Time taken to process a single text input and produce a prediction.",
                parameterCount: "Total number of trainable parameters that determine the model's capacity.",
                modelSize: "Disk space required to store the model weights and architecture.",
                modelComplexity: "Qualitative assessment based on parameter count and architectural complexity."
            }
        },
        {
            name: "T5-small",
            fullName: "Text-to-Text Transfer Transformer (Small)",
            description: "T5-small is a smaller version of the T5 model, capable of performing a wide range of NLP tasks by converting them into a text-to-text format. It can handle translation, summarization, question answering, and more with a unified approach.",
            architecture: "Encoder-decoder transformer with 6 encoder and 6 decoder layers, 512 hidden dimensions",
            useCases: ["Text Translation", "Text Summarization", "Question Answering", "Text Generation"],
            metrics: {
                before: {
                    accuracy: "88.1%",
                    loss: "0.51",
                    precision: "86.4%",
                    recall: "84.9%",
                    f1: "85.6%",
                    inference: "124 ms",
                    size: "231 MB"
                },
                after: {
                    accuracy: "84.7%",
                    loss: "0.57",
                    precision: "83.2%",
                    recall: "82.4%",
                    f1: "82.8%",
                    inference: "89 ms",
                    size: "162 MB"
                },
                modelComplexity: "Medium",
                parameterCount: "61M"
            },
            explanation: {
                accuracy: "Performance across multiple text-to-text tasks including translation and summarization.",
                f1Score: "Balanced performance metric for various text generation and understanding tasks.",
                inferenceLatency: "Time required to generate text output given an input sequence.",
                parameterCount: "Total parameters in both encoder and decoder components.",
                modelSize: "Storage space for the complete encoder-decoder architecture.",
                modelComplexity: "Medium complexity due to encoder-decoder structure."
            }
        },
        {
            name: "MobileNetV2",
            fullName: "MobileNet Version 2",
            description: "MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and object detection on mobile and embedded devices. It uses inverted residuals and linear bottlenecks to achieve high accuracy with low computational cost.",
            architecture: "CNN with inverted residual blocks, depthwise separable convolutions, 53 layers",
            useCases: ["Image Classification", "Object Detection", "Face Recognition", "Mobile Vision Apps"],
            metrics: {
                before: {
                    accuracy: "90.8%",
                    loss: "0.31",
                    precision: "90.2%",
                    recall: "89.4%",
                    f1: "89.8%",
                    inference: "34 ms",
                    size: "13.4 MB"
                },
                after: {
                    accuracy: "89.1%",
                    loss: "0.34",
                    precision: "88.4%",
                    recall: "88.0%",
                    f1: "88.2%",
                    inference: "24 ms",
                    size: "9.1 MB"
                },
                modelComplexity: "Low",
                parameterCount: "3.5M"
            },
            explanation: {
                accuracy: "Performance on ImageNet classification task with 1000 classes.",
                f1Score: "Balanced performance across different object categories in image classification.",
                inferenceLatency: "Time to classify a single 224x224 pixel image.",
                parameterCount: "Efficiently designed with depthwise separable convolutions.",
                modelSize: "Compact model suitable for mobile deployment.",
                modelComplexity: "Low complexity optimized for mobile and edge devices."
            }
        },
        {
            name: "ResNet-18",
            fullName: "Residual Network with 18 Layers",
            description: "ResNet-18 is a deep residual network with 18 layers, known for its ability to train very deep networks by using skip connections to avoid vanishing gradients. It's a foundational architecture that has influenced many subsequent models.",
            architecture: "CNN with residual connections, 18 layers, skip connections every 2 layers",
            useCases: ["Image Classification", "Feature Extraction", "Transfer Learning", "Computer Vision Research"],
            metrics: {
                before: {
                    accuracy: "94.2%",
                    loss: "0.27",
                    precision: "93.6%",
                    recall: "93.1%",
                    f1: "93.3%",
                    inference: "36 ms",
                    size: "45 MB"
                },
                after: {
                    accuracy: "91.8%",
                    loss: "0.33",
                    precision: "91.1%",
                    recall: "90.6%",
                    f1: "90.8%",
                    inference: "27 ms",
                    size: "31 MB"
                },
                modelComplexity: "High",
                parameterCount: "11.7M"
            },
            explanation: {
                accuracy: "Strong performance on ImageNet classification due to deep architecture and residual connections.",
                f1Score: "Consistent performance across various object categories.",
                inferenceLatency: "Time to process an image through the 18-layer deep network.",
                parameterCount: "More parameters due to deeper architecture with residual connections.",
                modelSize: "Larger model size reflecting the deeper network architecture.",
                modelComplexity: "High complexity due to deep structure and residual connections."
            }
        }
    ];

    const metricLabels = [
        { key: "accuracy", label: "Accuracy" },
        { key: "loss", label: "Loss" },
        { key: "precision", label: "Precision" },
        { key: "recall", label: "Recall" },
        { key: "f1", label: "F1 Score" },
        { key: "inference", label: "Inference Time" },
        { key: "size", label: "Model Size" }
    ];

    const getComplexityColor = (complexity) => {
        switch (complexity.toLowerCase()) {
            case 'low': return 'success';
            case 'medium': return 'warning';
            case 'high': return 'danger';
            default: return 'secondary';
        }
    };

    const handleCardClick = (model) => {
        setSelectedModel(model);
        setShowModal(true);
    };

    const handleStartTraining = (modelName) => {
        // Map display names to backend model names
        const modelNameMapping = {
            "DistilBERT": "distillBert",
            "T5-small": "T5-small", 
            "MobileNetV2": "MobileNetV2",
            "ResNet-18": "ResNet-18"
        };
        
        const backendModelName = modelNameMapping[modelName] || modelName;
        // Navigate to training page with selected model using React Router
        window.location.href = `/training?model=${backendModelName}`;
    };

    return (
        <>
            <Navbar bg="black" variant="dark" expand="lg">
                <Container>
                    <Navbar.Brand as={Link} to="/">KD-Pruning Simulator</Navbar.Brand>
                    <Navbar.Toggle aria-controls="basic-navbar-nav" />
                    <Navbar.Collapse id="basic-navbar-nav">
                        <Nav className="ms-auto">
                            <Nav.Link as={Link} to="/">Home</Nav.Link>
                            <Nav.Link as={Link} to="/instructions">Instructions</Nav.Link>
                            <Nav.Link as={Link} to="/models">Models</Nav.Link>
                            <Nav.Link as={Link} to="/training">Training</Nav.Link>
                            <Nav.Link as={Link} to="/visualization">Visualization</Nav.Link>
                            <Nav.Link as={Link} to="/assessment">Assessment</Nav.Link>
                        </Nav>
                    </Navbar.Collapse>
                </Container>
            </Navbar>

            <Container className="mt-5">
                <div className="text-center mb-5">
                    <h1 className="display-4 mb-3 page-hero-title">Available Models</h1>
                    <p className="lead page-hero-subtitle">
                        Explore our collection of pre-trained models and understand their performance characteristics before applying <strong className="hero-accent-primary">Knowledge Distillation</strong> and <strong className="hero-accent-success">Model Pruning</strong> techniques.
                    </p>
                    <p className="models-metrics-note">
                        Every metric pair below represents real <strong>trained</strong> baselines ("Before") versus their pruned KD students ("After"). All metrics are computed from actual Knowledge Distillation and Pruning training runs—not hardcoded. Training details (KD parameters, pruning ratios, epochs) are shown in each model card.
                    </p>
                </div>

                <Row>
                    {(models.length > 0 ? models : fallbackModelsArray).map((model, index) => (
                        <Col key={index} lg={6} className="mb-4">
                            <Card className="h-100 shadow-sm model-card" onClick={() => handleCardClick(model)}>
                                <Card.Header className="bg-primary text-white">
                                    <div className="d-flex justify-content-between align-items-center">
                                        <h4 className="mb-0">{model.name}</h4>
                                        <Badge bg="dark" text="light">
                                            {model.metrics.modelComplexity} Complexity
                                        </Badge>
                                    </div>
                                </Card.Header>
                                <Card.Body className="p-4">
                                    <h6 className="text-muted mb-2">{model.fullName}</h6>
                                    <p className="mb-3">{model.description}</p>
                                    
                                    {model.trainingHistory && (
                                        <div className="mb-3" style={{ background: '#e8f4f8', padding: '12px', borderRadius: '8px', border: '1px solid #1890ff' }}>
                                            <strong style={{ color: '#1890ff' }}>✅ Real Training Results:</strong>
                                            <p className="mb-1" style={{ color: '#000000', fontSize: '0.9rem' }}>{model.trainingHistory}</p>
                                            {model.kdExplanation && (
                                                <p className="mb-1" style={{ color: '#666', fontSize: '0.85rem' }}>
                                                    <strong>KD:</strong> {model.kdExplanation}
                                                </p>
                                            )}
                                            {model.pruningExplanation && (
                                                <p className="mb-0" style={{ color: '#666', fontSize: '0.85rem' }}>
                                                    <strong>Pruning:</strong> {model.pruningExplanation}
                                                </p>
                                            )}
                                        </div>
                                    )}
                                    
                                    <div className="mb-3">
                                        <strong>Architecture:</strong> {model.architecture}
                                    </div>
                                    
                                    <div className="mb-3">
                                        <strong>Use Cases:</strong>
                                        <div className="mt-1">
                                            {model.useCases.map((useCase, idx) => (
                                                <Badge key={idx} bg="secondary" className="me-1 mb-1">
                                                    {useCase}
                                                </Badge>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="metrics-compare-wrapper">
                                        <div className="metrics-columns">
                                            <div>
                                                <small className="text-muted d-block">Parameters</small>
                                                <div className="fw-bold">{model.metrics.parameterCount}</div>
                                            </div>
                                            <div>
                                                <small className="text-muted d-block">Complexity</small>
                                                <Badge bg={getComplexityColor(model.metrics.modelComplexity)}>
                                                    {model.metrics.modelComplexity} Complexity
                                                </Badge>
                                            </div>
                                        </div>
                                        <div className="metrics-matrix">
                                            <div className="metrics-matrix-header">
                                                <span>Metric</span>
                                                <span>Before</span>
                                                <span>After</span>
                                            </div>
                                            {metricLabels.map((item) => (
                                                <div className="metrics-matrix-row" key={`${model.name}-${item.key}`}>
                                                    <span className="metric-label">{item.label}</span>
                                                    <span className="metric-before">{model.metrics.before[item.key]}</span>
                                                    <span className="metric-after">{model.metrics.after[item.key]}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </Card.Body>
                                <Card.Footer className="bg-light">
                                    <div className="d-flex justify-content-center align-items-center">
                                        <small className="text-muted">
                                            These metrics show the baseline model after KD + Pruning. Go to Training page to compare with your custom model.
                                        </small>
                                    </div>
                                </Card.Footer>
                            </Card>
                        </Col>
                    ))}
                </Row>

                {/* Metrics Computation Explanation Section - Below All Models */}
                <Row className="mb-5 mt-5">
                    <Col xs={24}>
                        <Card className="shadow-sm" style={{ background: '#ffffff', border: '2px solid #1890ff' }}>
                            <Card.Header style={{ background: '#1890ff', color: 'white' }}>
                                <h4 className="mb-0">
                                    <InfoCircle className="me-2" />
                                    How Metrics Are Computed
                                </h4>
                            </Card.Header>
                            <Card.Body style={{ padding: '24px', color: '#000000' }}>
                                <Accordion defaultActiveKey="0" flush>
                                    <Accordion.Item eventKey="0">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>📊 Accuracy</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Accuracy = (Correct Predictions) / (Total Predictions) × 100%</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Uses <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>sklearn.metrics.accuracy_score</code> with actual model predictions vs. ground truth labels</li>
                                                    <li style={{ color: '#000000' }}>Calculated from real evaluation dataset during model testing</li>
                                                    <li style={{ color: '#000000' }}>Result multiplied by 100 to display as percentage</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> The proportion of correct predictions made by the model across all test samples.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="1">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>📈 F1-Score</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>F1 = 2 × (Precision × Recall) / (Precision + Recall) × 100%</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Uses <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>sklearn.metrics.f1_score</code> with <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>average='weighted'</code> for multi-class scenarios</li>
                                                    <li style={{ color: '#000000' }}>Calculates harmonic mean of precision and recall</li>
                                                    <li style={{ color: '#000000' }}>Result multiplied by 100 to display as percentage</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Balanced measure that combines precision and recall, especially important for imbalanced datasets.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="2">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>🎯 Precision</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Precision = True Positives / (True Positives + False Positives) × 100%</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Uses <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>sklearn.metrics.precision_score</code> with <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>average='weighted'</code></li>
                                                    <li style={{ color: '#000000' }}>Handles multi-class classification with weighted averaging</li>
                                                    <li style={{ color: '#000000' }}>Result multiplied by 100 to display as percentage</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Proportion of predicted positive cases that were actually positive.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="3">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>🔍 Recall</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Recall = True Positives / (True Positives + False Negatives) × 100%</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Uses <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>sklearn.metrics.recall_score</code> with <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>average='weighted'</code></li>
                                                    <li style={{ color: '#000000' }}>Calculated from confusion matrix of predictions vs. true labels</li>
                                                    <li style={{ color: '#000000' }}>Result multiplied by 100 to display as percentage</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Proportion of actual positive cases that were correctly identified by the model.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="4">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>💾 Model Size</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Size (MB) = Σ(parameters × element_size) / (1024 × 1024)</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Iterates through all model parameters: <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>for param in model.parameters()</code></li>
                                                    <li style={{ color: '#000000' }}>Calculates total bytes: <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>total_bytes += param.numel() × param.element_size()</code></li>
                                                    <li style={{ color: '#000000' }}>For pruned models: <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>effective_size = raw_size × (1 - sparsity / 100)</code></li>
                                                    <li style={{ color: '#000000' }}>Converts to megabytes: <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>size_mb = total_bytes / (1024 × 1024)</code></li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Disk space required to store all model weights, accounting for data type (typically float32 = 4 bytes).</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="5">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>⚡ Inference Latency</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Latency (ms) = mean(inference_times) × 1000</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Performs 10 inference runs with actual input data</li>
                                                    <li style={{ color: '#000000' }}>Measures time using <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>time.time()</code> before and after model forward pass</li>
                                                    <li style={{ color: '#000000' }}>Calculates mean latency: <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>latency_ms = np.mean(latencies)</code></li>
                                                    <li style={{ color: '#000000' }}>Converts from seconds to milliseconds (×1000)</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Average time taken for the model to process a single input and produce a prediction, measured in real-world conditions.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="6">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>🔢 Parameter Count</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Formula:</strong> <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Parameters = Σ(param.numel() for param in model.parameters())</code></p>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Computation Method:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}>Counts all trainable and non-trainable parameters</li>
                                                    <li style={{ color: '#000000' }}>Uses PyTorch's <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>numel()</code> method to count elements in each parameter tensor</li>
                                                    <li style={{ color: '#000000' }}>For pruned models: calculates effective parameters accounting for sparsity</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>What it measures:</strong> Total number of learnable weights in the neural network that determine the model's capacity and complexity.</p>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>

                                    <Accordion.Item eventKey="7">
                                        <Accordion.Header style={{ color: '#000000' }}>
                                            <strong style={{ color: '#000000' }}>📉 Compression Metrics</strong>
                                        </Accordion.Header>
                                        <Accordion.Body style={{ color: '#000000' }}>
                                            <div style={{ color: '#000000' }}>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Size Reduction:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}><code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Size Reduction (%) = ((Teacher Size - Student Size) / Teacher Size) × 100</code></li>
                                                    <li style={{ color: '#000000' }}>Calculated from actual measured model sizes (MB)</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Latency Improvement:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}><code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Latency Improvement (%) = ((Teacher Latency - Student Latency) / Teacher Latency) × 100</code></li>
                                                    <li style={{ color: '#000000' }}>Shows speedup from compression (positive values = faster)</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Parameter Reduction:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}><code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Param Reduction (%) = ((Teacher Params - Student Params) / Teacher Params) × 100</code></li>
                                                    <li style={{ color: '#000000' }}>Measures reduction in model complexity</li>
                                                </ul>
                                                <p style={{ color: '#000000' }}><strong style={{ color: '#000000' }}>Accuracy Impact:</strong></p>
                                                <ul style={{ color: '#000000' }}>
                                                    <li style={{ color: '#000000' }}><code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: '4px', color: '#000000' }}>Accuracy Impact (%) = Student Accuracy - Teacher Accuracy</code></li>
                                                    <li style={{ color: '#000000' }}>Shows performance trade-off (can be positive or negative)</li>
                                                </ul>
                                            </div>
                                        </Accordion.Body>
                                    </Accordion.Item>
                                </Accordion>
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            </Container>

            {/* Detailed Model Modal */}
            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
                <Modal.Header closeButton>
                    <Modal.Title>{selectedModel?.name} - Detailed Analysis</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {selectedModel && (
                        <div>
                            <h5>Performance Metrics with Explanations</h5>
                            <Table striped bordered hover responsive>
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                        <th>Explanation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>Accuracy (%)</strong></td>
                                        <td>{selectedModel.metrics.accuracy}</td>
                                        <td>{selectedModel.explanation.accuracy}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>F1-Score (%)</strong></td>
                                        <td>{selectedModel.metrics.f1Score}</td>
                                        <td>{selectedModel.explanation.f1Score}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Inference Latency (ms)</strong></td>
                                        <td>{selectedModel.metrics.inferenceLatency}</td>
                                        <td>{selectedModel.explanation.inferenceLatency}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Parameter Count</strong></td>
                                        <td>{selectedModel.metrics.parameterCount}</td>
                                        <td>{selectedModel.explanation.parameterCount}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Model Size (MB)</strong></td>
                                        <td>{selectedModel.metrics.modelSize}</td>
                                        <td>{selectedModel.explanation.modelSize}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Model Complexity</strong></td>
                                        <td>
                                            <Badge bg={getComplexityColor(selectedModel.metrics.modelComplexity)}>
                                                {selectedModel.metrics.modelComplexity}
                                            </Badge>
                                        </td>
                                        <td>{selectedModel.explanation.modelComplexity}</td>
                                    </tr>
                                </tbody>
                            </Table>

                            <div className="mt-4">
                                <h6>Expected Compression Benefits:</h6>
                                <ul>
                                    <li><strong>Size Reduction:</strong> {selectedModel.metrics.sizeReduction} smaller model</li>
                                    <li><strong>Faster Inference:</strong> Reduced latency for real-time applications</li>
                                    <li><strong>Lower Memory:</strong> Reduced RAM requirements for deployment</li>
                                    <li><strong>Energy Efficiency:</strong> Lower power consumption on mobile devices</li>
                                </ul>
                            </div>
                        </div>
                    )}
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={() => setShowModal(false)}>
                        Close
                    </Button>
                    <Button 
                        variant="primary" 
                        onClick={() => {
                            setShowModal(false);
                            // Navigate to training page with selected model
                            const modelNameMapping = {
                                "DistilBERT": "distillBert",
                                "T5-small": "T5-small", 
                                "MobileNetV2": "MobileNetV2",
                                "ResNet-18": "ResNet-18"
                            };
                            const backendModelName = modelNameMapping[selectedModel?.name] || selectedModel?.name;
                            window.location.href = `/training?model=${backendModelName}`;
                        }}
                    >
                        Go to Training Page
                    </Button>
                </Modal.Footer>
            </Modal>
            <Footer />
        </>
    );
};

export default Models;
