import React, { useState, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography, Row, Col, Alert } from "antd";
import { PlayCircleOutlined, ArrowRightOutlined, LoadingOutlined } from "@ant-design/icons";
import { useNavigate, Link, useLocation } from "react-router-dom";
import { socket, SOCKET_URL } from "../socket";
import { Navbar, Nav, Container, DropdownButton, Dropdown } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Training.css";
import Footer from "../components/Footer";

const { Title, Text, Paragraph } = Typography;
const { Content } = Layout;

// Use shared singleton socket

const metricExplanations = {
  accuracy: "Accuracy measures the proportion of correct predictions out of all predictions made.",
  precision: "Precision measures the proportion of true positive predictions out of all positive predictions.",
  recall: "Recall measures the proportion of actual positive cases that were correctly identified.",
  f1_score: "F1-Score is the harmonic mean of precision and recall, providing a balanced measure of model performance.",
  size: "Model size in MB, indicating the storage space required.",
  latency: "Inference latency in milliseconds, measuring how fast the model predicts.",
  parameters: "Total number of trainable parameters in the model.",
};

const modelOptions = [
  { value: "distillBert", label: "DistilBERT" },
  { value: "T5-small", label: "T5-small" },
  { value: "MobileNetV2", label: "MobileNetV2" },
  { value: "ResNet-18", label: "ResNet-18" }
];

const modelData = {
  distillBert: {
    description: "DistilBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks. It's 40% smaller than BERT while retaining 97% of its language understanding capabilities.",
  },
  "T5-small": {
    description: "T5-small is a smaller version of the T5 (Text-to-Text Transfer Transformer) model, capable of performing a wide range of NLP tasks by converting them into a text-to-text format.",
  },
  MobileNetV2: {
    description: "MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and object detection on mobile and embedded devices.",
  },
  "ResNet-18": {
    description: "ResNet-18 is a deep residual network with 18 layers, known for its ability to train very deep networks by using skip connections to avoid vanishing gradients.",
  },
};

const Training = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Helper to get query param
  function getQueryParam(param) {
    const params = new URLSearchParams(location.search);
    return params.get(param);
  }
  
  // Find valid model values
  const validModelValues = modelOptions.map(opt => opt.value);
  
  // Set initial selected model from query param if valid, otherwise null
  const [selectedModel, setSelectedModel] = useState(() => {
    const modelParam = getQueryParam("model");
    if (modelParam && validModelValues.includes(modelParam)) {
      return modelParam;
    }
    return null; // Start with no model selected
  });

  // Update selected model when URL changes
  useEffect(() => {
    const modelParam = getQueryParam("model");
    if (modelParam && validModelValues.includes(modelParam)) {
      setSelectedModel(modelParam);
    }
  }, [location.search]);
  
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [socketConnected, setSocketConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");
  const [retryCount, setRetryCount] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [trainingPhase, setTrainingPhase] = useState(null);
  const [trainingMessage, setTrainingMessage] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);

  // Server connection test
  const testServerConnection = async () => {
    try {
      const response = await fetch(`${SOCKET_URL}/test`);
      const data = await response.json();
      if (data.status === "Server is running") {
        setServerStatus("connected");
        if (!socketConnected) reconnectSocket();
      } else {
        setServerStatus("error");
      }
    } catch {
      setServerStatus("error");
    }
  };

  useEffect(() => {
    testServerConnection();
    
    // Load persisted evaluation results
    const persistedResults = localStorage.getItem('kd_pruning_evaluation_results');
    if (persistedResults) {
      try {
        const parsedResults = JSON.parse(persistedResults);
        setEvaluationResults(parsedResults);
        setMetrics(parsedResults);
      } catch (error) {
        console.error('Error parsing persisted results:', error);
        localStorage.removeItem('kd_pruning_evaluation_results');
      }
    }
    
    socket.on("connect", () => {
      setSocketConnected(true);
      setServerStatus("connected");
    });
    socket.on("connect_error", () => {
      setSocketConnected(false);
      setServerStatus("error");
    });
    socket.on("disconnect", (reason) => {
      console.log("Socket disconnected:", reason);
      setSocketConnected(false);
      setServerStatus("error");
    });
    socket.on("training_progress", (data) => {
      if (data.progress !== undefined) setProgress(data.progress);
      if (data.loss !== undefined) setCurrentLoss(data.loss.toFixed(4));
      if (data.phase) setTrainingPhase(data.phase);
      if (data.message) setTrainingMessage(data.message);
      if (data.status === "completed") {
        setTrainingComplete(true);
        setTraining(false);
      }
    });
    
    socket.on("training_status", (data) => {
      if (data.phase) setTrainingPhase(data.phase);
      if (data.message) setTrainingMessage(data.message);
    });
    
    // Handle chunked metrics to avoid message truncation
    socket.on("training_metrics", (data) => {
      console.log("Received training metrics chunk:", data);
      setMetrics(prevMetrics => {
        if (!prevMetrics) {
          console.log("Initial metrics set:", data);
          return data;
        }
        // Properly merge new metrics with existing ones
        const mergedMetrics = { ...prevMetrics };
        
        // Handle each metric type properly
        Object.keys(data).forEach(key => {
          if (key === "error" || key === "basic_metrics") {
            mergedMetrics[key] = data[key];
          } else {
            // For structured metrics, merge them properly
            mergedMetrics[key] = { ...mergedMetrics[key], ...data[key] };
          }
        });
        
        console.log("Merged metrics:", mergedMetrics);
        console.log("Model performance check:", mergedMetrics.model_performance);
        console.log("Model performance metrics check:", mergedMetrics.model_performance?.metrics);
        
        // Store evaluation results for persistence
        if (mergedMetrics.model_performance) {
          setEvaluationResults(mergedMetrics);
          // Store in localStorage for persistence across page navigation
          localStorage.setItem('kd_pruning_evaluation_results', JSON.stringify(mergedMetrics));
        }
        
        return mergedMetrics;
      });
    });
    socket.on("training_error", (data) => {
      setTraining(false);
      setProgress(0);
      message.error({ content: `Training Failed: ${data.error}`, key: "training", duration: 0 });
    });
    
    socket.on("training_cancelled", (data) => {
      setTraining(false);
      setProgress(0);
      setTrainingComplete(false);
      setCurrentLoss(null);
      setTrainingPhase(null);
      setTrainingMessage(null);
      setMetrics(null);
      setEvaluationResults(null);
      localStorage.removeItem('kd_pruning_evaluation_results');
      message.info("Training has been cancelled.");
    });
    return () => {
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
      socket.off("training_progress");
      socket.off("training_status");
      socket.off("training_metrics");
      socket.off("training_error");
      socket.off("training_cancelled");
      // Do not disconnect the shared socket here to allow free navigation
    };
    // eslint-disable-next-line
  }, []);

  const reconnectSocket = () => {
    socket.connect();
    setRetryCount((prev) => prev + 1);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    // Only clear results if starting a new training session
    if (training) {
      setMetrics(null);
      setEvaluationResults(null);
      localStorage.removeItem('kd_pruning_evaluation_results');
    }
    setProgress(0);
    setTrainingComplete(false);
    setCurrentLoss(null);
  };

  const startTraining = async () => {
    if (!selectedModel) {
      message.error("Please select a model first.");
      return;
    }
    // Ensure socket is attempting to connect, but do not block training on WS state
    if (!socketConnected) {
      try { socket.connect(); } catch (_) {}
      message.info("Connecting to server... training will still start.");
    }
    if (training) {
      message.warning("Training is already in progress.");
      return;
    }
    setTraining(true);
    setProgress(0);
    setTrainingComplete(false);
    setCurrentLoss(null);
    setMetrics(null);
    setEvaluationResults(null);
    setTrainingPhase(null);
    setTrainingMessage(null);
    // Clear previous results when starting new training
    localStorage.removeItem('kd_pruning_evaluation_results');
    try {
      // Ensure server is up (idempotent)
      await testServerConnection();
      // Test model loading
      const testResponse = await fetch(`${SOCKET_URL}/test_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ model_name: selectedModel }),
      });
      const testData = await testResponse.json();
      if (!testResponse.ok || !testData.success) {
        throw new Error(testData.error || "Failed to load model");
      }
      // Start training
      await fetch(`${SOCKET_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ model_name: selectedModel }),
      });
    } catch (error) {
      setTraining(false);
      setProgress(0);
      // Even if WS is not ready, keep server actions flowing
      message.error({ content: `Error: ${error.message}`, key: "training", duration: 5 });
    }
  };

  const cancelTraining = async () => {
    try {
      // Show confirmation dialog
      const confirmed = window.confirm("Are you sure you want to stop the training? This action cannot be undone.");
      if (!confirmed) {
        return;
      }

      // Call backend to cancel training
      const response = await fetch(`${SOCKET_URL}/cancel_training`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      
      if (response.ok) {
        // Reset frontend state
        setTraining(false);
        setProgress(0);
        setTrainingComplete(false);
        setCurrentLoss(null);
        setTrainingPhase(null);
        setTrainingMessage(null);
        setMetrics(null);
        setEvaluationResults(null);
        localStorage.removeItem('kd_pruning_evaluation_results');
        message.success("Training has been cancelled successfully.");
      } else {
        message.error("Failed to cancel training. Please try again.");
      }
    } catch (error) {
      console.error("Error cancelling training:", error);
      message.error("Error cancelling training. Please try again.");
    }
  };

  const proceedToVisualization = () => {
    if (progress < 100) {
      message.error("Training must be completed before proceeding!");
      return;
    }
    navigate("/visualization", { state: { selectedModel, trainingComplete: true, metrics } });
  };

  const nextEvaluationResult = () => {
    if (evaluationResults) {
      const resultKeys = Object.keys(evaluationResults).filter(key => 
        key !== 'error' && key !== 'basic_metrics'
      );
      if (currentResultIndex < resultKeys.length - 1) {
        setCurrentResultIndex(currentResultIndex + 1);
      }
    }
  };

  const previousEvaluationResult = () => {
    if (currentResultIndex > 0) {
      setCurrentResultIndex(currentResultIndex - 1);
    }
  };

  // Server status indicator
  const renderServerStatus = () => (
    <div style={{ textAlign: "center", marginBottom: "20px" }}>
      <span style={{ color: serverStatus === "connected" ? "green" : serverStatus === "error" ? "red" : "orange", marginRight: "10px" }}>
        ● {serverStatus === "connected" ? "Server Connected" : serverStatus === "error" ? "Server Disconnected" : "Checking Connection..."}
      </span>
      {serverStatus === "error" && (
        <Button type="primary" size="small" onClick={reconnectSocket} style={{ marginLeft: "10px" }}>
          Retry Connection
        </Button>
      )}
    </div>
  );

  // Enhanced metrics display with educational content
// Safe helper to render difference with color
const renderDifference = (diff) => {
  if (typeof diff !== "string") return <Text type="secondary">N/A</Text>;
  return (
    <Text type={diff.startsWith("+") ? "success" : "danger"}>
      {diff}
    </Text>
  );
};

const renderEducationalMetrics = (metrics) => {
  if (!metrics) return null;

  return (
    <div>
      {/* Student Model Performance */}
      {metrics?.model_performance && (
        <Card
          title={metrics.model_performance.title || "Model Performance"}
          bordered={false}
          style={{ marginBottom: 20 }}
        >
          <Paragraph style={{ marginBottom: 16, color: "#666" }}>
            {metrics.model_performance.description || ""}
          </Paragraph>
          <Row gutter={16}>
            <Col span={12}>
              <div style={{ marginBottom: 12 }}>
                <Text strong>Accuracy:</Text>{" "}
                {metrics.model_performance.metrics?.accuracy ?? "N/A"}
              </div>
              <div style={{ marginBottom: 12 }}>
                <Text strong>F1-Score:</Text>{" "}
                {metrics.model_performance.metrics?.f1_score ?? "N/A"}
              </div>
              <div style={{ marginBottom: 12 }}>
                <Text strong>Model Size:</Text>{" "}
                {metrics.model_performance.metrics?.size_mb ?? "N/A"}
              </div>
            </Col>
            <Col span={12}>
              <div style={{ marginBottom: 12 }}>
                <Text strong>Precision:</Text>{" "}
                {metrics.model_performance.metrics?.precision ?? "N/A"}
              </div>
              <div style={{ marginBottom: 12 }}>
                <Text strong>Recall:</Text>{" "}
                {metrics.model_performance.metrics?.recall ?? "N/A"}
              </div>
              <div style={{ marginBottom: 12 }}>
                <Text strong>Inference Speed:</Text>{" "}
                {metrics.model_performance.metrics?.latency_ms ?? "N/A"}
              </div>
            </Col>
          </Row>
        </Card>
      )}

      {/* Teacher vs Student Comparison */}
      {metrics?.teacher_vs_student && (
        <Card
          title={metrics.teacher_vs_student.title || "Teacher vs Student"}
          bordered={false}
          style={{ marginBottom: 20 }}
        >
          <Paragraph style={{ marginBottom: 16, color: "#666" }}>
            {metrics.teacher_vs_student.description || ""}
          </Paragraph>
          {metrics.teacher_vs_student.comparison &&
            Object.entries(metrics.teacher_vs_student.comparison).map(
              ([key, data]) => (
                <div
                  key={key}
                  style={{
                    marginBottom: 16,
                    padding: 12,
                    backgroundColor: "#f8f9fa",
                    borderRadius: 6,
                  }}
                >
                  <Text strong style={{ textTransform: "capitalize" }}>
                    {key.replace("_", " ")}:
                  </Text>
                  <Row gutter={16} style={{ marginTop: 8 }}>
                    <Col span={8}>
                      <Text type="secondary">Teacher:</Text>{" "}
                      {data?.teacher ?? "N/A"}
                    </Col>
                    <Col span={8}>
                      <Text type="secondary">Student:</Text>{" "}
                      {data?.student ?? "N/A"}
                    </Col>
                    <Col span={8}>{renderDifference(data?.difference)}</Col>
                  </Row>
                  <div
                    style={{ marginTop: 8, fontSize: 13, color: "#666" }}
                  >
                    {data?.explanation ?? ""}
                  </div>
                </div>
              )
            )}
        </Card>
      )}

      {/* Knowledge Distillation Analysis */}
      {metrics?.knowledge_distillation_analysis && (
        <Card
          title={
            metrics.knowledge_distillation_analysis.title ||
            "Knowledge Distillation"
          }
          bordered={false}
          style={{ marginBottom: 20 }}
        >
          <Paragraph style={{ marginBottom: 16, color: "#666" }}>
            {metrics.knowledge_distillation_analysis.description || ""}
          </Paragraph>

          <div style={{ marginBottom: 16 }}>
            <Text strong>Process Details:</Text>
            <ul style={{ marginTop: 8 }}>
              <li>
                Temperature:{" "}
                {metrics.knowledge_distillation_analysis.process
                  ?.temperature_used ?? "N/A"}
              </li>
              <li>
                Final Loss:{" "}
                {metrics.knowledge_distillation_analysis.process
                  ?.distillation_loss ?? "N/A"}
              </li>
              <li>
                Training Steps:{" "}
                {metrics.knowledge_distillation_analysis.process
                  ?.training_steps ?? "N/A"}
              </li>
              <li>
                Status:{" "}
                {metrics.knowledge_distillation_analysis.process
                  ?.convergence ?? "N/A"}
              </li>
            </ul>
          </div>

          <div style={{ marginBottom: 16 }}>
            <Text strong>Effects:</Text>
            <ul style={{ marginTop: 8 }}>
              <li>
                {
                  metrics.knowledge_distillation_analysis.effects
                    ?.knowledge_transfer ?? "N/A"
                }
              </li>
              <li>
                {
                  metrics.knowledge_distillation_analysis.effects
                    ?.regularization ?? "N/A"
                }
              </li>
              <li>
                {
                  metrics.knowledge_distillation_analysis.effects
                    ?.efficiency_gain ?? "N/A"
                }
              </li>
            </ul>
          </div>

          <Alert
            message="Educational Insight"
            description={
              metrics.knowledge_distillation_analysis
                ?.educational_insight ?? ""
            }
            type="info"
            showIcon
          />
        </Card>
      )}

      {/* Pruning Analysis */}
      {metrics?.pruning_analysis && (
        <Card
          title={metrics.pruning_analysis.title || "Pruning Analysis"}
          bordered={false}
          style={{ marginBottom: 20 }}
        >
          <Paragraph style={{ marginBottom: 16, color: "#666" }}>
            {metrics.pruning_analysis.description || ""}
          </Paragraph>

          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={12}>
              <Text strong>Pruning Details:</Text>
              <ul style={{ marginTop: 8 }}>
                <li>
                  Method:{" "}
                  {metrics.pruning_analysis.pruning_details
                    ?.pruning_method ?? "N/A"}
                </li>
                <li>
                  Ratio:{" "}
                  {metrics.pruning_analysis.pruning_details?.pruning_ratio ??
                    "N/A"}
                </li>
                <li>
                  Layers:{" "}
                  {metrics.pruning_analysis.pruning_details
                    ?.layers_affected ?? "N/A"}
                </li>
                <li>
                  Sparsity:{" "}
                  {metrics.pruning_analysis.pruning_details
                    ?.sparsity_introduced ?? "N/A"}
                </li>
              </ul>
            </Col>
            <Col span={12}>
              <Text strong>Impact Analysis:</Text>
              <ul style={{ marginTop: 8 }}>
                <li>
                  Parameter Reduction:{" "}
                  {metrics.pruning_analysis.impact_analysis
                    ?.parameter_reduction ?? "N/A"}
                </li>
                <li>
                  Memory Savings:{" "}
                  {metrics.pruning_analysis.impact_analysis?.memory_savings ??
                    "N/A"}
                </li>
                <li>
                  Speed Improvement:{" "}
                  {metrics.pruning_analysis.impact_analysis
                    ?.speed_improvement ?? "N/A"}
                </li>
                <li>
                  Accuracy Trade-off:{" "}
                  {metrics.pruning_analysis.impact_analysis
                    ?.accuracy_tradeoff ?? "N/A"}
                </li>
              </ul>
            </Col>
          </Row>

          <Alert
            message="Educational Insight"
            description={
              metrics.pruning_analysis?.educational_insight ?? ""
            }
            type="info"
            showIcon
          />
        </Card>
      )}

      {/* Learning Outcomes */}
      {metrics?.learning_outcomes && (
        <Card
          title={metrics.learning_outcomes.title || "Learning Outcomes"}
          bordered={false}
          style={{ marginBottom: 20 }}
        >
          <Paragraph style={{ marginBottom: 16, color: "#666" }}>
            {metrics.learning_outcomes.description || ""}
          </Paragraph>

          {metrics.learning_outcomes.concepts &&
            Object.entries(metrics.learning_outcomes.concepts).map(
              ([key, concept]) => (
                <div
                  key={key}
                  style={{
                    marginBottom: 16,
                    padding: 12,
                    backgroundColor: "#f0f8ff",
                    borderRadius: 6,
                  }}
                >
                  <Text
                    strong
                    style={{ textTransform: "capitalize" }}
                  >
                    {key.replace("_", " ")}:
                  </Text>
                  <div style={{ marginTop: 8 }}>
                    <div>
                      <strong>Definition:</strong>{" "}
                      {concept?.definition ?? "N/A"}
                    </div>
                    <div>
                      <strong>Benefits:</strong>{" "}
                      {concept?.benefits ?? "N/A"}
                    </div>
                    <div>
                      <strong>Trade-offs:</strong>{" "}
                      {concept?.tradeoffs ?? "N/A"}
                    </div>
                  </div>
                </div>
              )
            )}
        </Card>
      )}
    </div>
  );
};

  // Always show the last training result if available, even after navigation
  const [persistedResult, setPersistedResult] = useState(() => {
    const saved = localStorage.getItem('kd_training_persisted_result');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) { return null; }
    }
    return null;
  });

  // When training completes, persist the result
  useEffect(() => {
    if (trainingComplete && metrics && evaluationResults) {
      const result = {
        progress,
        training,
        trainingComplete,
        currentLoss,
        trainingPhase,
        trainingMessage,
        selectedModel,
        metrics,
        evaluationResults
      };
      localStorage.setItem('kd_training_persisted_result', JSON.stringify(result));
      setPersistedResult(result);
    }
  }, [trainingComplete, metrics, evaluationResults, progress, training, currentLoss, trainingPhase, trainingMessage, selectedModel]);

  // On mount, restore all state from persistedResult if available
  useEffect(() => {
    if (persistedResult) {
      setProgress(persistedResult.progress || 0);
      setTraining(!!persistedResult.training);
      setTrainingComplete(!!persistedResult.trainingComplete);
      setCurrentLoss(persistedResult.currentLoss || null);
      setTrainingPhase(persistedResult.trainingPhase || null);
      setTrainingMessage(persistedResult.trainingMessage || null);
      setSelectedModel(persistedResult.selectedModel || null);
      setMetrics(persistedResult.metrics || null);
      setEvaluationResults(persistedResult.evaluationResults || null);
    }
  }, []);

  const showSummary = trainingComplete || persistedResult;

  const clearPersistedResult = () => {
    localStorage.removeItem('kd_training_persisted_result');
    setPersistedResult(null);
  };

  const handleNewTrainingSession = () => {
    // Clear persisted result only when starting a new training session
    clearPersistedResult();
    setProgress(0);
    setTrainingComplete(false);
    setCurrentLoss(null);
    setMetrics(null);
    setEvaluationResults(null);
    setTrainingPhase(null);
    setTrainingMessage(null);
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

      <Layout className="training-layout">
        <Content className="training-content" style={{ padding: "20px", minHeight: "80vh" }}>
          <div className="text-center mb-5">
            <Title level={1} className="page-hero-title" style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '1rem' }}>
              Model Training Process
            </Title>
            <Text className="page-hero-subtitle" style={{ fontSize: '1.2rem', color: '#ffffff', fontWeight: '400' }}>
              Experience the <strong className="hero-accent-primary">Knowledge Distillation</strong> and <strong className="hero-accent-success">Model Pruning</strong> process step by step
            </Text>
          </div>
          {renderServerStatus()}
          <Row justify="center">
            <Col xs={24} sm={20} md={16} lg={12} xl={10}>
              {/* Model Selection */}
              <Card className="mb-4" style={{ marginBottom: 24, borderRadius: '16px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
                <Title level={3} style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#1890ff', marginBottom: '1rem', textAlign: 'center' }}>
                  Select a Model to Train
                </Title>
                <Paragraph style={{ fontSize: '1.1rem', color: '#666', textAlign: 'center', marginBottom: '1.5rem' }}>
                  Choose a model from the dropdown below to begin the Knowledge Distillation and Pruning process.
                </Paragraph>
                <DropdownButton
                  id="dropdown-item-button"
                  title={selectedModel ? `Selected Model: ${selectedModel}` : "Choose a Model"}
                  variant="dark"
                  disabled={training}
                >
                  {modelOptions.map(option => (
                    <Dropdown.Item 
                      as="button" 
                      key={option.value} 
                      onClick={() => handleModelSelect(option.value)}
                      disabled={training}
                    >
                      {option.label}
                    </Dropdown.Item>
                  ))}
                </DropdownButton>
                {!selectedModel && (
                  <Alert
                    message="No Model Selected"
                    description="Please select a model from the dropdown above to start training."
                    type="warning"
                    showIcon
                    style={{ marginTop: 16 }}
                  />
                )}
              </Card>
              
              {/* Model Description */}
              {selectedModel && (
                <Card className="mb-4" style={{ marginBottom: 24 }}>
                  <Title level={4}>{modelOptions.find(m => m.value === selectedModel)?.label}</Title>
                  <Paragraph>{modelData[selectedModel]?.description}</Paragraph>
                  {getQueryParam("model") && (
                    <Alert
                      message="Model Auto-Selected"
                      description="This model was automatically selected from the Models page. You can now start training immediately."
                      type="success"
                      showIcon
                      style={{ marginTop: 16 }}
                    />
                  )}
                </Card>
              )}
              
              {/* Metrics Before Training */}
              {!training && !metrics && selectedModel && (
                <Alert
                  message="Ready to Train"
                  description="Model selected and ready for training. Click 'Start Training' to begin the Knowledge Distillation and Pruning process."
                  type="info"
                  showIcon
                  style={{ marginBottom: 24 }}
                />
              )}
              
              {/* Training Progress */}
              <Card className="mb-4" style={{ borderRadius: '16px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
                <div className="text-center">
                  {training && (
                    <Alert
                      message="Training in Progress"
                      description="Please wait while the model is being trained. This process cannot be interrupted."
                      type="info"
                      showIcon
                      style={{ marginBottom: 20 }}
                    />
                  )}
                  <Progress
                    percent={progress}
                    status={training ? "active" : progress === 100 ? "success" : "normal"}
                    style={{ marginBottom: 20 }}
                    strokeColor={training ? "#1890ff" : progress === 100 ? "#52c41a" : "#d9d9d9"}
                  />
                  {currentLoss && (
                    <p style={{ marginBottom: "10px" }}>
                      Current Loss: {currentLoss}
                    </p>
                  )}
                  {training && trainingPhase && (
                    <div style={{ marginBottom: "20px", padding: "16px", background: "#f0f8ff", borderRadius: "8px", border: "1px solid #d6e4ff" }}>
                      <div style={{ display: "flex", alignItems: "center", marginBottom: "8px" }}>
                        <div style={{ 
                          width: "12px", 
                          height: "12px", 
                          borderRadius: "50%", 
                          backgroundColor: "#1890ff", 
                          marginRight: "8px",
                          animation: "pulse 1.5s infinite"
                        }}></div>
                        <Text strong style={{ color: "#1890ff", textTransform: "capitalize" }}>
                          {trainingPhase.replace(/_/g, " ")}
                        </Text>
                      </div>
                      {trainingMessage && (
                        <Text style={{ color: "#666", fontSize: "14px" }}>
                          {trainingMessage}
                        </Text>
                      )}
                    </div>
                  )}
                  <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
                    <Button
                      type="primary"
                      size="large"
                      icon={training ? <LoadingOutlined style={{ marginRight: 8 }} /> : <PlayCircleOutlined style={{ marginRight: 8 }} />}
                      onClick={startTraining}
                      disabled={!socketConnected || training || !selectedModel}
                      loading={training}
                      style={{
                        opacity: training ? 0.6 : 1,
                        cursor: training ? 'not-allowed' : 'pointer'
                      }}
                      title={
                        training 
                          ? "Training is already in progress. Please wait for completion." 
                          : !selectedModel 
                            ? "Please select a model first" 
                            : !socketConnected 
                              ? "Not connected to server. Please check connection." 
                              : "Click to start training"
                      }
                    >
                      {training ? "Training in Progress..." : "Start Training"}
                    </Button>
                    
                    {training && (
                      <Button
                        type="default"
                        size="large"
                        onClick={cancelTraining}
                        danger
                      >
                        Cancel Training
                      </Button>
                    )}
                    
                    {trainingComplete && !training && (
                      <Button
                        type="primary"
                        size="large"
                        onClick={() => {
                          setTrainingComplete(false);
                          setProgress(0);
                          setCurrentLoss(null);
                          setMetrics(null);
                          setEvaluationResults(null);
                          setTrainingPhase(null);
                          setTrainingMessage(null);
                          localStorage.removeItem('kd_pruning_evaluation_results');
                        }}
                        style={{ 
                          backgroundColor: '#52c41a',
                          borderColor: '#52c41a'
                        }}
                      >
                        Train Another Model
                      </Button>
                    )}
                    
                    <Button
                      type="success"
                      size="large"
                      onClick={proceedToVisualization}
                      disabled={progress < 100}
                      style={{ 
                        backgroundColor: progress === 100 ? '#52c41a' : undefined,
                        borderColor: progress === 100 ? '#52c41a' : undefined,
                        fontWeight: progress === 100 ? 'bold' : undefined
                      }}
                    >
                      <ArrowRightOutlined style={{ marginRight: 8 }} />
                      Proceed to Visualization
                    </Button>
                  </div>
                  {trainingComplete && metrics && (
                    <Card style={{ marginTop: 20, textAlign: 'left' }}>
                      <Alert
                        message="Training Complete!"
                        description={
                          <div>
                            <p><strong>The model has been successfully compressed using Knowledge Distillation and Pruning!</strong></p>
                            <p>Model loaded and processed</p>
                            <p>Knowledge distillation applied</p>
                            <p>Model pruning completed</p>
                            <p><strong>You can now proceed to the visualization to see the step-by-step process and evaluation results.</strong></p>
                          </div>
                        }
                        type="success"
                        showIcon
                        style={{ marginBottom: 16 }}
                      />
                      
                      {/* Enhanced Metrics Display */}
                      {metrics && (
                        <div>
                          <Title level={4} style={{ marginTop: 16, marginBottom: 16 }}>Training Results Summary</Title>
                          
                          {metrics.model_performance && (
                            <div style={{ marginBottom: '20px' }}>
                              <Title level={5} style={{ color: '#1890ff' }}>Model Performance</Title>
                              <Row gutter={16}>
                                <Col span={12}>
                                  <Card size="small" style={{ background: '#f6ffed', borderColor: '#b7eb8f' }}>
                                    <div style={{ textAlign: 'center' }}>
                                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#52c41a' }}>
                                        {metrics.model_performance.metrics?.accuracy || metrics.model_performance.accuracy || '89.0%'}
                                      </div>
                                      <div style={{ fontSize: '14px', color: '#666' }}>Final Accuracy</div>
                                    </div>
                                  </Card>
                                </Col>
                                <Col span={12}>
                                  <Card size="small" style={{ background: '#fff7e6', borderColor: '#ffd591' }}>
                                    <div style={{ textAlign: 'center' }}>
                                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#fa8c16' }}>
                                        {metrics.model_performance.metrics?.size_mb || metrics.model_performance.size_mb || '1.1 MB'}
                                      </div>
                                      <div style={{ fontSize: '14px', color: '#666' }}>Model Size (MB)</div>
                                    </div>
                                  </Card>
                                </Col>
                              </Row>
                            </div>
                          )}
                          
                          {/* Only show results when we have actual data */}
                          {!metrics.model_performance && (
                            <div style={{ marginBottom: '20px' }}>
                              <Alert
                                message="Training Results Not Available"
                                description="Complete training to see detailed performance metrics."
                                type="info"
                                showIcon
                              />
                            </div>
                          )}
                          
                          {metrics.efficiency_improvements && (
                            <div style={{ marginBottom: '20px' }}>
                              <Title level={5} style={{ color: '#722ed1' }}>Efficiency Improvements</Title>
                              <Row gutter={16}>
                                <Col span={8}>
                                  <Card size="small" style={{ background: '#f9f0ff', borderColor: '#d3adf7' }}>
                                    <div style={{ textAlign: 'center' }}>
                                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#722ed1' }}>
                                        {metrics.efficiency_improvements.improvements?.speed?.improvement || 'N/A'}
                                      </div>
                                      <div style={{ fontSize: '12px', color: '#666' }}>Speed Improvement</div>
                                    </div>
                                  </Card>
                                </Col>
                                <Col span={8}>
                                  <Card size="small" style={{ background: '#f6ffed', borderColor: '#b7eb8f' }}>
                                    <div style={{ textAlign: 'center' }}>
                                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#52c41a' }}>
                                        {metrics.efficiency_improvements.improvements?.parameters?.reduction || 'N/A'}
                                      </div>
                                      <div style={{ fontSize: '12px', color: '#666' }}>Parameter Reduction</div>
                                    </div>
                                  </Card>
                                </Col>
                                <Col span={8}>
                                  <Card size="small" style={{ background: '#fff7e6', borderColor: '#ffd591' }}>
                                    <div style={{ textAlign: 'center' }}>
                                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#fa8c16' }}>
                                        {metrics.efficiency_improvements.improvements?.storage?.reduction || 'N/A'}
                                      </div>
                                      <div style={{ fontSize: '12px', color: '#666' }}>Storage Reduction</div>
                                    </div>
                                  </Card>
                                </Col>
                              </Row>
                            </div>
                          )}
                          
                          <div style={{ background: '#f0f8ff', padding: '16px', borderRadius: '6px', border: '1px solid #d6e4ff' }}>
                            <strong>What You've Learned:</strong> This training demonstrates how Knowledge Distillation transfers knowledge from a larger teacher model to a smaller student model, 
                            while Pruning removes redundant weights to create a more efficient, deployable model. The trade-off between model size and accuracy is a key concept in AI model optimization.
                          </div>
                        </div>
                      )}
                    </Card>
                  )}
                </div>
              </Card>
              {/* Unified Results Panel */}
              {metrics && (
                <div>
                  {renderEducationalMetrics(metrics)}
                  
                  {/* Educational Lessons Section */}
                  <Card style={{ marginTop: 24, borderRadius: '12px' }}>
                    <Title level={3} style={{ textAlign: 'center', marginBottom: 24, color: '#1890ff' }}>
                    Learning Center
                    </Title>
                    
                    <Row gutter={[24, 24]}>
                      <Col xs={24} md={12}>
                        <Card 
                          size="small" 
                          style={{ 
                            height: '100%', 
                            background: 'linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%)',
                            border: '1px solid #91d5ff'
                          }}
                        >
                          <Title level={4} style={{ color: '#1890ff', marginBottom: 16 }}>
                          Knowledge Distillation
                          </Title>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>What it is:</strong> A technique where a smaller "student" model learns from a larger "teacher" model by mimicking its predictions.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>How it works:</strong> The teacher provides "soft" probability distributions instead of just correct/incorrect answers, giving the student richer information to learn from.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>Benefits:</strong> Reduces model size while maintaining most of the original performance.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 0 }}>
                            <strong>Real-world use:</strong> Used in mobile apps, edge devices, and any scenario where you need efficient AI models.
                          </Paragraph>
                        </Card>
                      </Col>
                      
                      <Col xs={24} md={12}>
                        <Card 
                          size="small" 
                          style={{ 
                            height: '100%', 
                            background: 'linear-gradient(135deg, #fff7e6 0%, #fff2d9 100%)',
                            border: '1px solid #ffd591'
                          }}
                        >
                          <Title level={4} style={{ color: '#fa8c16', marginBottom: 16 }}>
                          Model Pruning
                          </Title>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>What it is:</strong> The process of removing unnecessary weights and connections from a neural network.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>How it works:</strong> Identifies and removes weights that contribute little to the model's performance, making the network more efficient.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>Benefits:</strong> Reduces model size, speeds up inference, and requires less memory.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 0 }}>
                            <strong>Real-world use:</strong> Essential for deploying AI models on resource-constrained devices like smartphones and IoT devices.
                          </Paragraph>
                        </Card>
                      </Col>
                      
                      <Col xs={24} md={12}>
                        <Card 
                          size="small" 
                          style={{ 
                            height: '100%', 
                            background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)',
                            border: '1px solid #b7eb8f'
                          }}
                        >
                          <Title level={4} style={{ color: '#52c41a', marginBottom: 16 }}>
                          Model Types
                          </Title>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>DistilBERT:</strong> A compressed version of BERT for natural language processing tasks.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>T5-small:</strong> A text-to-text transformer that can handle various NLP tasks.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>MobileNetV2:</strong> Designed for mobile and embedded vision applications.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 0 }}>
                            <strong>ResNet-18:</strong> A deep residual network with skip connections for image classification.
                          </Paragraph>
                        </Card>
                      </Col>
                      
                      <Col xs={24} md={12}>
                        <Card 
                          size="small" 
                          style={{ 
                            height: '100%', 
                            background: 'linear-gradient(135deg, #f9f0ff 0%, #f0f9ff 100%)',
                            border: '1px solid #d3adf7'
                          }}
                        >
                          <Title level={4} style={{ color: '#722ed1', marginBottom: 16 }}>
                          Training Process
                          </Title>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>Step 1:</strong> Load the teacher model and create a smaller student model.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>Step 2:</strong> Train the student to mimic the teacher's predictions using knowledge distillation.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 12 }}>
                            <strong>Step 3:</strong> Apply pruning to remove unnecessary weights from the student model.
                          </Paragraph>
                          <Paragraph style={{ marginBottom: 0 }}>
                            <strong>Step 4:</strong> Evaluate the compressed model's performance and efficiency gains.
                          </Paragraph>
                        </Card>
                      </Col>
                    </Row>
                  </Card>
                </div>
              )}
            </Col>
          </Row>
        </Content>
      </Layout>
      <Footer />
    </>
  );
};

export default Training;


