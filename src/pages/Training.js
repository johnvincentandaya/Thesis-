import React, { useState, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography, Row, Col, Alert, Divider } from "antd";
import { ArrowRightOutlined } from "@ant-design/icons";
import { useNavigate, Link, useLocation } from "react-router-dom";
import { socket, SOCKET_URL, checkConnectionHealth, forceReconnect } from "../socket";
import { Navbar, Nav, Container, DropdownButton, Dropdown } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Training.css";
import Footer from "../components/Footer";
import TrainingComparison from "../components/TrainingComparison";
import "../styles/TrainingComparison.css";

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

const ALLOWED_UPLOAD_EXTENSIONS = [".pt", ".pth", ".bin", ".json", ".config", ".ckpt"];
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;  // 500MB
const DEFAULT_MODEL_NAMES = ["distilbert", "resnet18", "resnet-18", "mobilenetv2", "mobilenet-v2", "t5small", "t5-small", "t5_small", "t5 small"];

const modelOptions = [
  { value: "distillBert", label: "DistilBERT" },
  { value: "T5-small", label: "T5-small" },
  { value: "MobileNetV2", label: "MobileNetV2" },
  { value: "ResNet-18", label: "ResNet-18" }
];

const getSelectedModelLabel = (value) => {
  if (!value) return "Not selected";
  return modelOptions.find(opt => opt.value === value)?.label || value;
};

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

const baselineReferenceMetrics = {
  distillBert: {
    label: "DistilBERT",
    accuracy: 89.6,
    f1: 88.7,
    sizeMB: 178,
    latencyMs: 48,
    params: 67_000_000,
  },
  "T5-small": {
    label: "T5-small",
    accuracy: 84.7,
    f1: 82.8,
    sizeMB: 162,
    latencyMs: 89,
    params: 61_000_000,
  },
  MobileNetV2: {
    label: "MobileNetV2",
    accuracy: 89.1,
    f1: 88.2,
    sizeMB: 9.1,
    latencyMs: 24,
    params: 3_500_000,
  },
  "ResNet-18": {
    label: "ResNet-18",
    accuracy: 91.8,
    f1: 90.8,
    sizeMB: 31,
    latencyMs: 27,
    params: 11_700_000,
  },
};

const formatPercentValue = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value.toFixed(2)}%`;
};

const formatParamValue = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "N/A";
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M params`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K params`;
  }
  return `${value} params`;
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

  // --- Model Pre-selection Logic ---
  // Try to get model from navigation state (from Models page), then from query param, else null
  const getInitialSelectedModel = () => {
    // 1. From navigation state (Models page)
    const navModel = location.state && location.state.selectedModel;
    if (navModel && validModelValues.includes(navModel)) {
      return navModel;
    }
    // 2. From query param (URL)
    const modelParam = getQueryParam("model");
    if (modelParam && validModelValues.includes(modelParam)) {
      return modelParam;
    }
    // 3. Fallback: no selection
    return null;
  };

  // Dropdown: Model selection logic (pre-select from navigation state if present)
  const [selectedModel, setSelectedModel] = useState(getInitialSelectedModel);

  // --- Socket and Server State ---
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [socketConnected, setSocketConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");
  const [retryCount, setRetryCount] = useState(0);
  const [retryLoading, setRetryLoading] = useState(false);
  const [currentLoss, setCurrentLoss] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [trainingPhase, setTrainingPhase] = useState(null);
  const [trainingMessage, setTrainingMessage] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);
  const [visualizationUnlocked, setVisualizationUnlocked] = useState(() => {
    return localStorage.getItem('visualization_unlocked') === 'true';
  });
  const [computationDetails, setComputationDetails] = useState(null);

  // --- Pagination State ---
  // Results container: 4 pages
  const [resultsPage, setResultsPage] = useState(0);

  // --- Error State ---
  const [trainingError, setTrainingError] = useState(null);
  const cachedUploadMeta = (() => {
    try {
      const saved = localStorage.getItem('kd_uploaded_model_meta');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  })();
  const [uploadStatus, setUploadStatus] = useState(cachedUploadMeta ? "success" : "idle");
  const [uploadError, setUploadError] = useState(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadedModelPath, setUploadedModelPath] = useState(cachedUploadMeta?.path || null);
  const [uploadedModelName, setUploadedModelName] = useState(cachedUploadMeta?.name || null);
  const [uploadedModelSize, setUploadedModelSize] = useState(cachedUploadMeta?.size || null);

  // Server connection test
  // Remove message.info from testServerConnection to avoid auto notice
  const testServerConnection = async () => {
    try {
      const response = await fetch(`${SOCKET_URL}/test`);
      const data = await response.json();
      if (data.status === "Server is running") {
        setServerStatus("connected");
      } else {
        setServerStatus("error");
      }
    } catch {
      setServerStatus("error");
    }
  };

  // Enhanced server status checking function
  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${SOCKET_URL}/test`, {
        method: 'GET',
        timeout: 5000
      });
      const data = await response.json();
      if (data.status === "Server is running") {
        setServerStatus("connected");
        setSocketConnected(true);
        return true;
      } else {
        setServerStatus("error");
        setSocketConnected(false);
        return false;
      }
    } catch (error) {
      console.log("Server status check failed:", error);
      setServerStatus("error");
      setSocketConnected(false);
      return false;
    }
  };

  useEffect(() => {
    // Immediately check server status on mount
    checkServerStatus();

    // Set up periodic server status checks
    const statusCheckInterval = setInterval(checkServerStatus, 10000); // Check every 10 seconds

    socket.on("connect", () => {
      console.log("Socket connected successfully");
      setSocketConnected(true);
      setServerStatus("connected");
      setTrainingError(null); // Clear any previous errors
    });

    // Enhanced connection monitoring
    socket.on("server_ready", (data) => {
      console.log("Server ready:", data);
      setServerStatus("connected");
      setSocketConnected(true);
    });

    socket.on("connection_acknowledged", (data) => {
      console.log("Connection acknowledged by server:", data);
      setServerStatus("connected");
      setSocketConnected(true);
    });

    socket.on("health_status", (data) => {
      console.log("Server health status:", data);
      setServerStatus("connected");
      setSocketConnected(true);
    });

    socket.on("server_error", (data) => {
      console.error("Server error:", data);
      message.error({ content: `Server error: ${data.error}`, duration: 15 });
    });
    
    socket.on("connect_error", (error) => {
      console.log("Socket connection error:", error);
      setSocketConnected(false);
      // Don't immediately set server status to error - check HTTP first
      checkServerStatus();
    });
    
    socket.on("disconnect", (reason) => {
      console.log("Socket disconnected:", reason);
      setSocketConnected(false);
      // Check if server is still reachable via HTTP
      if (!training) {
        checkServerStatus();
      }
    });
    
    socket.on("reconnect", (attemptNumber) => {
      console.log("Socket reconnected after", attemptNumber, "attempts");
      setSocketConnected(true);
      setServerStatus("connected");
      setTrainingError(null);
    });
    
    socket.on("reconnect_failed", () => {
      console.log("Socket reconnection failed");
      setSocketConnected(false);
      // Check if server is still reachable via HTTP
      checkServerStatus();
      if (training) {
        message.error({ content: "Lost socket connection during training. Training may continue in background.", duration: 15 });
      } else {
        message.error({ content: "Socket reconnection failed. Checking server status...", duration: 15 });
      }
    });
    
    // Add heartbeat to keep connection alive during training
    const heartbeat = setInterval(() => {
      if (socket.connected && training) {
        socket.emit('ping');
      }
    }, 30000); // Ping every 30 seconds during training
    const phaseOrder = ["model_loading", "knowledge_distillation", "pruning", "evaluation", "completed"];

socket.on("training_progress", (data) => {
  if (!data) return;

  // Make sure progress only moves forward
  if (typeof data.progress === "number") {
    setProgress(prev => Math.max(prev, data.progress));
  }

  // Make sure phase only moves forward
  if (data.phase) {
    setTrainingPhase(prevPhase => {
      const prevIdx = phaseOrder.indexOf(prevPhase);
      const newIdx = phaseOrder.indexOf(data.phase);

      // allow forward or same, block backward
      if (newIdx === -1) return prevPhase;
      if (prevIdx === -1 || newIdx >= prevIdx) {
        return data.phase;
      }
      return prevPhase;
    });
  }

  // Update loss if present
  if (data.loss !== undefined) {
    setCurrentLoss(data.loss.toFixed(4));
  }

  // Always take latest message if provided
  if (data.message) {
    setTrainingMessage(data.message);
  }

  // Mark training complete only when backend says so
  if (data.status === "completed" || data.phase === "completed" || data.progress === 100) {
    setProgress(100);
    setTrainingComplete(true);
    setTraining(false);
    console.log("Training marked as complete");
  }
});

    
    socket.on("training_status", (data) => {
      if (data.phase) setTrainingPhase(data.phase);
      if (data.message) setTrainingMessage(data.message);
    });
    
    // Handle chunked metrics to avoid message truncation
    socket.on("training_metrics", (data) => {
      console.log("Received training_metrics:", data);
      setMetrics(prevMetrics => {
        // If no previous metrics, just use incoming
        if (!prevMetrics) return data;

        // Start with previous
        const merged = { ...prevMetrics };

        // Helper: deep merge comparisons
        const mergeComparison = (prevComp, newComp) => {
          if (!prevComp) return newComp || {};
          if (!newComp) return prevComp;
          const out = { ...prevComp };
          Object.keys(newComp).forEach((metricKey) => {
            out[metricKey] = { ...(prevComp[metricKey] || {}), ...(newComp[metricKey] || {}) };
          });
          return out;
        };

        // Merge each top-level section
        Object.keys(data).forEach((key) => {
          // Pass-through error/basic
          if (key === "error" || key === "basic_metrics") {
            merged[key] = data[key];
            return;
          }

          if (key === "model_performance") {
            // Ensure metrics are merged, not overwritten
            const prevMp = merged.model_performance || {};
            const newMp = data.model_performance || {};
            merged.model_performance = {
              ...prevMp,
              ...newMp,
              metrics: {
                ...(prevMp.metrics || {}),
                ...(newMp.metrics || {}),
              }
            };
            return;
          }

          if (key === "teacher_vs_student") {
            const prevTvs = merged.teacher_vs_student || {};
            const newTvs = data.teacher_vs_student || {};
            merged.teacher_vs_student = {
              ...prevTvs,
              ...newTvs,
              comparison: mergeComparison(prevTvs.comparison, newTvs.comparison)
            };
            return;
          }

          // Default shallow merge for other sections
          merged[key] = { ...(merged[key] || {}), ...(data[key] || {}) };
        });

        // Persist after each merge if we have core data
        if (merged.model_performance) {
          setEvaluationResults(merged);
          localStorage.setItem('kd_pruning_evaluation_results', JSON.stringify(merged));
        }
        return merged;
      });
    });

    // Listen for evaluation metrics (4 categories)
    socket.on("evaluation_metrics", (data) => {
      console.log("Received evaluation metrics:", data);
      setEvaluationResults(prevResults => {
        const merged = { ...prevResults };
        merged.evaluation_metrics = data;
        localStorage.setItem('kd_pruning_evaluation_results', JSON.stringify(merged));
        return merged;
      });
    });

    // Listen for training computation details (side-by-side comparison)
    socket.on("training_computation_details", (data) => {
      console.log("Received training_computation_details:", data);
      setComputationDetails(data);
    });

    socket.on("training_error", (data) => {
      setTraining(false);
      setProgress(0);
      setTrainingError(data.error || "Training Failed");
      message.error({ content: `Training Failed: ${data.error}`, key: "training", duration: 15 });
      
      // Auto-clear error after 15 seconds
      setTimeout(() => {
        setTrainingError(null);
      }, 15000);
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
      clearInterval(heartbeat);
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
      socket.off("reconnect");
      socket.off("reconnect_failed");
      socket.off("training_progress");
      socket.off("training_status");
      socket.off("training_metrics");
      socket.off("evaluation_metrics");
      socket.off("training_computation_details");
      socket.off("training_error");
      socket.off("training_cancelled");
      // Do not disconnect the shared socket here to allow free navigation
    };
    // eslint-disable-next-line
  }, []);

  // Enhanced reconnection with health check
  const reconnectSocket = async () => {
    if (retryLoading) return; // Prevent multiple clicks
    setRetryLoading(true);
    
    // Show notice to user ONLY when user clicks Retry
    if (!reconnectSocket._noticeShown) {
      message.info("Attempting to reconnect to the server...");
      reconnectSocket._noticeShown = true;
      setTimeout(() => { reconnectSocket._noticeShown = false; }, 3000);
    }
    
    try {
      // Check current connection health
      const health = checkConnectionHealth();
      console.log("Connection health before reconnect:", health);
      
      // Force reconnection
      forceReconnect();
      setRetryCount((prev) => prev + 1);
      
      // Wait a moment for connection to establish
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Test server connection
      await testServerConnection();
      
      // Check health again
      const newHealth = checkConnectionHealth();
      console.log("Connection health after reconnect:", newHealth);
      
      if (newHealth.isConnected) {
        message.success("Successfully reconnected to server!");
      } else {
        message.warning("Reconnection attempted. Please check server status.");
      }
    } catch (err) {
      console.error("Reconnection error:", err);
      message.error({ content: "Failed to reconnect. Please check your server and network.", duration: 15 });
    } finally {
      setRetryLoading(false);
    }
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

  const startTraining = async (options = {}) => {
    const {
      modelPath: overrideModelPath,
      modelName: overrideModelName,
      skipUploadValidation = false
    } = options;

    if (!selectedModel) {
      message.error({ content: "Please select a model first.", duration: 15 });
      return;
    }
    const effectiveModelPath = overrideModelPath || uploadedModelPath;
    const effectiveModelName = overrideModelName || uploadedModelName;

    if (!effectiveModelPath) {
      message.error({ content: "Upload a valid custom model (.pt/.pth/.bin, max 500MB) before training.", duration: 15 });
      return;
    }
    if (!skipUploadValidation && uploadStatus !== "success") {
      message.error({ content: "Upload a valid custom model (.pt/.pth/.bin, max 500MB) before training.", duration: 15 });
      return;
    }
    if (uploadingFile) {
      message.warning("Please wait for the upload to finish.");
      return;
    }
    
    if (training) {
      message.warning("Training is already in progress.");
      return;
    }
    
    // Reset training state
    setTraining(true);
    setProgress(0);
    setTrainingComplete(false);
    setCurrentLoss(null);
    setMetrics(null);
    setEvaluationResults(null);
    setTrainingPhase(null);
    setTrainingMessage(null);
    setTrainingError(null);
    // Clear previous results when starting new training
    localStorage.removeItem('kd_pruning_evaluation_results');
    
    try {
      // Test server connection first
      console.log("Testing server connection...");
      const serverResponse = await fetch(`${SOCKET_URL}/test`, {
        method: "GET",
        timeout: 10000
      });
      
      if (!serverResponse.ok) {
        throw new Error("Server is not responding");
      }
      
      const serverData = await serverResponse.json();
      if (serverData.status !== "Server is running") {
        throw new Error("Server is not ready");
      }
      
      console.log("Server connection verified");
      
      // Test model loading
      console.log(`Testing model loading for ${selectedModel}...`);
      const testResponse = await fetch(`${SOCKET_URL}/test_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ model_name: selectedModel }),
        timeout: 15000
      });
      
      const testData = await testResponse.json();
      console.log("Model test response:", testData);
      
      if (!testResponse.ok || !testData.success) {
        throw new Error(testData.error || "Failed to load model");
      }
      
      console.log("Model loading test passed");
      
      // Start training
      console.log("Starting training...");
      const trainResponse = await fetch(`${SOCKET_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ 
          model_name: selectedModel,
          uploaded_model_path: effectiveModelPath,
          uploaded_model_name: effectiveModelName
        }),
        timeout: 5000
      });
      
      if (!trainResponse.ok) {
        throw new Error("Failed to start training");
      }
      
      console.log("Training started successfully");
      message.success("Training started successfully!");
      
    } catch (error) {
      console.error("Training start error:", error);
      setTraining(false);
      setProgress(0);
      setTrainingError(error.message);
      message.error({ 
        content: `Failed to start training: ${error.message}`, 
        key: "training", 
        duration: 15 
      });
    }
  };

  // Stop Training must terminate backend process too
  const cancelTraining = async () => {
    try {
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
        message.error({ content: "Failed to cancel training. Please try again.", duration: 15 });
      }
    } catch (error) {
      console.error("Error cancelling training:", error);
      message.error({ content: "Error cancelling training. Please try again.", duration: 15 });
    }
  };

  const proceedToVisualization = () => {
    if (progress < 100) {
      message.error({ content: "Training must be completed before proceeding!", duration: 15 });
      return;
    }
    navigate("/visualization", { 
      state: { 
        selectedModel, 
        trainingComplete: true, 
        metrics,
        uploadedModelMeta: uploadedModelPath ? {
          name: uploadedModelName,
          size: uploadedModelSize,
          path: uploadedModelPath
        } : null
      } 
    });
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

  // Enhanced server status indicator with health check
  const renderServerStatus = () => {
    const handleHealthCheck = () => {
      const health = checkConnectionHealth();
      console.log("Connection health check:", health);
      message.info(`Connection Status: ${health.isConnected ? 'Connected' : 'Disconnected'}\nServer: ${health.serverUrl}\nLast Ping: ${health.lastPing || 'Never'}`);
    };

    return (
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <span style={{ color: serverStatus === "connected" ? "green" : serverStatus === "error" ? "red" : "orange", marginRight: "10px" }}>
          ● {serverStatus === "connected" ? "Server Connected" : serverStatus === "error" ? "Server Disconnected" : "Checking Connection..."}
        </span>
        <Button 
          type="default" 
          size="small" 
          onClick={handleHealthCheck}
          style={{ marginLeft: "10px" }}
          title="Check connection health"
        >
          Health Check
        </Button>
        {serverStatus === "error" && (
          <>
            <Button type="primary" size="small" onClick={reconnectSocket} loading={retryLoading} disabled={retryLoading} style={{ marginLeft: "10px" }}>
              Retry Connection
            </Button>
            <span style={{ color: "#faad14", marginLeft: 10 }}>
              If the server was restarted, click "Retry Connection" to reconnect.
            </span>
          </>
        )}
      </div>
    );
  };

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
    // Only load persisted result if it exists, training is not in progress, AND user has previously started training in this session
    // We'll use a session flag to track if user has started training at least once
    const saved = localStorage.getItem('kd_training_persisted_result');
    const trainingStarted = sessionStorage.getItem('kd_training_started');
    if (saved && trainingStarted === 'true') {
      try {
        const parsed = JSON.parse(saved);
        if (!parsed.training) {
          return parsed;
        }
      } catch (e) { return null; }
    }
    return null;
  });

  // Track if user has started training in this session
  const [hasStartedTraining, setHasStartedTraining] = useState(() => {
    return sessionStorage.getItem('kd_training_started') === 'true';
  });

  const formatBytes = (bytes) => {
    if (!bytes && bytes !== 0) return null;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const isDefaultModelFile = (filename) => {
    if (!filename) return false;
    const normalized = filename.toLowerCase().replace(/[\s_-]/g, "");
    return DEFAULT_MODEL_NAMES.some((name) =>
      normalized.includes(name.replace(/[\s_-]/g, ""))
    );
  };

  const handleCustomModelUpload = async (event) => {
    const inputEl = event.target;
    const file = inputEl.files && inputEl.files[0];
    if (!file) return;

    const extension = `.${file.name.split(".").pop().toLowerCase()}`;
    if (!ALLOWED_UPLOAD_EXTENSIONS.includes(extension)) {
      const errorMsg = "Invalid file type. Allowed: .pt, .pth, .bin (primary); optionally .json, .config, .ckpt.";
      setUploadError(errorMsg);
      setUploadStatus("error");
      setUploadedModelPath(null);
      message.error({ content: errorMsg, duration: 15 });
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      const errorMsg = "File exceeds the 500MB limit.";
      setUploadError(errorMsg);
      setUploadStatus("error");
      setUploadedModelPath(null);
      message.error({ content: errorMsg, duration: 15 });
      return;
    }

    if (isDefaultModelFile(file.name)) {
      const errorMsg = "This model type is already available in the system. Please upload a different model.";
      setUploadError(errorMsg);
      setUploadStatus("error");
      setUploadedModelPath(null);
      message.warning(errorMsg);
      return;
    }

    setUploadingFile(true);
    setUploadError(null);
    setUploadStatus("uploading");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${SOCKET_URL}/upload`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Upload failed");
      }

      setUploadedModelPath(data.file_path);
      setUploadedModelName(file.name);
      setUploadedModelSize(file.size);
      setUploadStatus("success");
      const meta = {
        name: file.name,
        size: file.size,
        path: data.file_path,
        uploadedAt: new Date().toISOString()
      };
      localStorage.setItem("kd_uploaded_model_meta", JSON.stringify(meta));
      message.success(`Uploaded ${file.name} successfully.`);
    } catch (error) {
      console.error("Upload error:", error);
      const errorMsg = error.message || "Upload failed. Please try again.";
      setUploadError(errorMsg);
      setUploadStatus("error");
      setUploadedModelPath(null);
      message.error({ content: errorMsg, duration: 15 });
    } finally {
      setUploadingFile(false);
      if (inputEl) {
        inputEl.value = null;
      }
    }
  };

  const clearUploadedModel = () => {
    setUploadedModelPath(null);
    setUploadedModelName(null);
    setUploadedModelSize(null);
    setUploadStatus("idle");
    setUploadError(null);
    localStorage.removeItem("kd_uploaded_model_meta");
  };

  const resetForNewTraining = () => {
    // Clear uploaded model
    clearUploadedModel();
    // Reset training state
    setTraining(false);
    setTrainingComplete(false);
    setTrainingError(null);
    setProgress(0);
    setCurrentLoss(null);
    setTrainingPhase(null);
    setTrainingMessage(null);
    setMetrics(null);
    setComputationDetails(null);
    // Clear persisted results
    clearPersistedResult();
    // Scroll to top to show upload section
    window.scrollTo({ top: 0, behavior: 'smooth' });
    message.success("Ready to train a new model. Please upload your model file.");
  };

  

  // On mount, restore all state from persistedResult if available and not currently training, and only if user has started training before
  useEffect(() => {
    const restoreState = async () => {
      // First try to restore from localStorage
      if (persistedResult && !training && hasStartedTraining) {
        setProgress(persistedResult.progress || 0);
        setTraining(false);
        setTrainingComplete(!!persistedResult.trainingComplete);
        setCurrentLoss(persistedResult.currentLoss || null);
        setTrainingPhase(persistedResult.trainingPhase || null);
        setTrainingMessage(persistedResult.trainingMessage || null);
        setSelectedModel(persistedResult.selectedModel || null);
        setMetrics(persistedResult.metrics || null);
        setEvaluationResults(persistedResult.evaluationResults || null);
        console.log("Restored state from localStorage");
      } else if (!training && !hasStartedTraining) {
        // Previous results loading is disabled for real training
        console.log("Starting fresh training session - no cached results");
      }
    };
    
    restoreState();
    // eslint-disable-next-line
  }, []);

  // Debug effect to track metrics changes
  useEffect(() => {
    console.log("Metrics updated:", metrics);
    if (metrics) {
      console.log("✅ Metrics received - results should show now");
      console.log("Training complete:", trainingComplete);
      console.log("Training active:", training);
      console.log("Progress:", progress);
    }
  }, [metrics, trainingComplete, training, progress]);

  // When training completes, persist the result and mark that user has started training
  useEffect(() => {
    if (trainingComplete && metrics && evaluationResults) {
      const result = {
        progress,
        training: false,
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
      setHasStartedTraining(true);
      sessionStorage.setItem('kd_training_started', 'true');
      // Unlock visualization
      setVisualizationUnlocked(true);
      localStorage.setItem('visualization_unlocked', 'true');
    }
  }, [trainingComplete, metrics, evaluationResults, progress, currentLoss, trainingPhase, trainingMessage, selectedModel]);

  const clearPersistedResult = () => {
    localStorage.removeItem('kd_training_persisted_result');
    setPersistedResult(null);
    setHasStartedTraining(false);
    sessionStorage.removeItem('kd_training_started');
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
    // Hide results section immediately
    setHasStartedTraining(false);
    sessionStorage.removeItem('kd_training_started');
  };

  // Results display logic
  // Show results if we have metrics (primary condition)
  // Also show if training is complete and we have metrics
  // Also show persisted results if available
  const showResults = metrics || // Show as soon as metrics are available
  (trainingComplete && metrics) || 
  (!training && hasStartedTraining && persistedResult && persistedResult.metrics) ||
  (!training && metrics && progress >= 90); // Fallback: show if we have metrics and high progress
  
  // Debug logging
  console.log("Training state:", { trainingComplete, hasMetrics: !!metrics, showResults, training, hasStartedTraining, progress });

  // --- Results Container Pages ---
  const renderResultsPage = () => {
    if (trainingError) {
      return (
        <Alert
          message="Training Error"
          description={trainingError}
          type="error"
          showIcon
        />
      );
    }
    // Show results if we have metrics, even if training completion is delayed
    if (!metrics) {
      return (
        <div style={{ textAlign: "center", padding: 32 }}>
          <Text type="secondary">
            {progress >= 90 ? "Processing final results..." : "Results will appear here after training completes."}
          </Text>
        </div>
      );
    }
    switch (resultsPage) {
      case 0:
        return (
          <div>
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
            <Alert
              type="info"
              showIcon
              message="Comparison Context"
              description={
                <div>
                  <div><strong>Baseline:</strong> {getSelectedModelLabel(selectedModel)}</div>
                  <div><strong>Uploaded Model:</strong> {uploadedModelName || "Custom model"}</div>
                  {uploadedModelSize && (
                    <div><strong>Uploaded Size:</strong> {formatBytes(uploadedModelSize)}</div>
                  )}
                </div>
              }
              style={{ marginBottom: 16 }}
            />
            <Title level={4} style={{ marginTop: 16, marginBottom: 16 }}>Training Results Summary</Title>
            
            {/* Display Evaluation Metrics (4 Categories) */}
            {evaluationResults?.evaluation_metrics && (
              <Card style={{ marginBottom: 16, background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)' }}>
                <Title level={5} style={{ color: '#52c41a', marginBottom: 16 }}>
                  📊 Evaluation Metrics (4 Categories)
                </Title>
                <Row gutter={[16, 16]}>
                  {Object.entries(evaluationResults.evaluation_metrics).map(([category, metrics]) => (
                    <Col span={6} key={category}>
                      <Card size="small" style={{ height: '100%' }}>
                        <Title level={5} style={{ color: '#1890ff', marginBottom: 12 }}>
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </Title>
                        {metrics.map((metric, index) => (
                          <div key={index} style={{ marginBottom: 8, fontSize: '12px' }}>
                            <div style={{ fontWeight: 'bold', color: '#666' }}>{metric.metric}</div>
                            <div style={{ color: '#52c41a' }}>
                              {metric.before} → {metric.after}
                            </div>
                          </div>
                        ))}
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            )}
            
            {metrics.model_performance && (
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
            )}
          </div>
        );
      case 1:
        // Student Model Performance
        return (
          <div>
            {metrics.model_performance ? (
              <Card bordered={false}>
                <Title level={4}>{metrics.model_performance.title || "Student Model Performance"}</Title>
                <Paragraph style={{ color: "#666" }}>{metrics.model_performance.description}</Paragraph>
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
            ) : <Text type="secondary">No student model performance data.</Text>}
          </div>
        );
      case 2:
        // Teacher vs Student Comparison
        return (
          <div>
            {metrics.teacher_vs_student ? (
              <Card bordered={false}>
                <Title level={4}>{metrics.teacher_vs_student.title || "Teacher vs Student"}</Title>
                <Paragraph style={{ color: "#666" }}>{metrics.teacher_vs_student.description}</Paragraph>
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
            ) : <Text type="secondary">No teacher vs student comparison data.</Text>}
          </div>
        );
      case 3:
        // Knowledge Distillation + Pruning Analysis
        return (
          <div>
            {metrics.knowledge_distillation_analysis && (
              <Card bordered={false} style={{ marginBottom: 16 }}>
                <Title level={4}>{metrics.knowledge_distillation_analysis.title || "Knowledge Distillation"}</Title>
                <Paragraph style={{ color: "#666" }}>{metrics.knowledge_distillation_analysis.description}</Paragraph>
                <div style={{ marginBottom: 16 }}>
                  <Text strong>Process Details:</Text>
                  <ul style={{ marginTop: 8 }}>
                    <li>
                      Temperature:{" "}
                      {metrics.knowledge_distillation_analysis.process?.temperature_used ?? "N/A"}
                    </li>
                    <li>
                      Final Loss:{" "}
                      {metrics.knowledge_distillation_analysis.process?.distillation_loss ?? "N/A"}
                    </li>
                    <li>
                      Training Steps:{" "}
                      {metrics.knowledge_distillation_analysis.process?.training_steps ?? "N/A"}
                    </li>
                    <li>
                      Status:{" "}
                      {metrics.knowledge_distillation_analysis.process?.convergence ?? "N/A"}
                    </li>
                  </ul>
                </div>
                <div style={{ marginBottom: 16 }}>
                  <Text strong>Effects:</Text>
                  <ul style={{ marginTop: 8 }}>
                    <li>
                      {metrics.knowledge_distillation_analysis.effects?.knowledge_transfer ?? "N/A"}
                    </li>
                    <li>
                      {metrics.knowledge_distillation_analysis.effects?.regularization ?? "N/A"}
                    </li>
                    <li>
                      {metrics.knowledge_distillation_analysis.effects?.efficiency_gain ?? "N/A"}
                    </li>
                  </ul>
                </div>
                <Alert
                  message="Educational Insight"
                  description={metrics.knowledge_distillation_analysis?.educational_insight ?? ""
                  }
                  type="info"
                  showIcon
                />
              </Card>
            )}
            {metrics.pruning_analysis && (
              <Card bordered={false}>
                <Title level={4}>{metrics.pruning_analysis.title || "Pruning Analysis"}</Title>
                <Paragraph style={{ color: "#666" }}>{metrics.pruning_analysis.description}</Paragraph>
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col span={12}>
                    <Text strong>Pruning Details:</Text>
                    <ul style={{ marginTop: 8 }}>
                      <li>
                        Method:{" "}
                        {metrics.pruning_analysis.pruning_details?.pruning_method ?? "N/A"}
                      </li>
                      <li>
                        Ratio:{" "}
                        {metrics.pruning_analysis.pruning_details?.pruning_ratio ?? "N/A"}
                      </li>
                      <li>
                        Layers:{" "}
                        {metrics.pruning_analysis.pruning_details?.layers_affected ?? "N/A"}
                      </li>
                      <li>
                        Sparsity:{" "}
                        {metrics.pruning_analysis.pruning_details?.sparsity_introduced ?? "N/A"}
                      </li>
                    </ul>
                  </Col>
                  <Col span={12}>
                    <Text strong>Impact Analysis:</Text>
                    <ul style={{ marginTop: 8 }}>
                      <li>
                        Parameter Reduction:{" "}
                        {metrics.pruning_analysis.impact_analysis?.parameter_reduction ?? "N/A"}
                      </li>
                      <li>
                        Memory Savings:{" "}
                        {metrics.pruning_analysis.impact_analysis?.memory_savings ?? "N/A"}
                      </li>
                      <li>
                        Speed Improvement:{" "}
                        {metrics.pruning_analysis.impact_analysis?.speed_improvement ?? "N/A"}
                      </li>
                      <li>
                        Accuracy Trade-off:{" "}
                        {metrics.pruning_analysis.impact_analysis?.accuracy_tradeoff ?? "N/A"}
                      </li>
                    </ul>
                  </Col>
                </Row>
                <Alert
                  message="Educational Insight"
                  description={metrics.pruning_analysis?.educational_insight ?? ""}
                  type="info"
                  showIcon
                />
              </Card>
            )}
          </div>
        );
      default:
        return null;
    }
  };

  // --- Learning Center Pages ---
  const renderLearningPage = (pageIndex) => {
    switch (pageIndex) {
      case 0:
        return (
          <div>
            <Card
              size="small"
              style={{
                marginBottom: 16,
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
            <Card
              size="small"
              style={{
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
          </div>
        );
      case 1:
        return (
          <div>
            <Card
              size="small"
              style={{
                marginBottom: 16,
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
            <Card
              size="small"
              style={{
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
          </div>
        );
      default:
        return null;
    }
  };

  // --- Responsive Layout ---
  const isMobile = window.innerWidth < 992;

  // --- Results Navigation ---
  const resultsPagesCount = 4;

  // --- Main Render ---
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
              <Nav.Link as={Link} to="/visualization" disabled={!visualizationUnlocked} style={{ pointerEvents: visualizationUnlocked ? 'auto' : 'none', opacity: visualizationUnlocked ? 1 : 0.5 }}>Visualization</Nav.Link>
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

          {/* Main Layout: Organized Flow */}
          <Row gutter={[24, 24]} justify="center">
            {/* Main Content: Step-by-step flow */}
            <Col xs={24} lg={16}>
              <Row gutter={[16, 8]}>
                {/* Step 1: Select Baseline Model - FIRST */}
                <Col xs={24}>
                  <Card
                    className="mb-4"
                    style={{
                      borderRadius: '16px',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                      textAlign: 'center'
                    }}
                  >
                    <Title level={3} style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#1890ff', marginBottom: '0.75rem', textAlign: 'center' }}>
                      Step 1: Select Baseline Model
                    </Title>
                    <DropdownButton
                      id="dropdown-item-button"
                      title={
                        selectedModel
                          ? `Selected: ${modelOptions.find(opt => opt.value === selectedModel)?.label || selectedModel}`
                          : "Select a baseline model"
                      }
                      variant="dark"
                      disabled={training || trainingComplete}
                      style={{ marginBottom: 16 }}
                    >
                      {modelOptions.map(option => (
                        <Dropdown.Item
                          as="button"
                          key={option.value}
                          onClick={() => setSelectedModel(option.value)}
                          disabled={training || trainingComplete}
                        >
                          {option.label}
                        </Dropdown.Item>
                      ))}
                    </DropdownButton>
                    {selectedModel && (
                      <Paragraph style={{ marginTop: 8, color: "#666" }}>
                        {modelData[selectedModel]?.description}
                      </Paragraph>
                    )}
                    {!selectedModel && (
                      <Alert
                        message="No Model Selected"
                        description="Please select a baseline model from the dropdown above."
                        type="info"
                        showIcon
                        style={{ marginTop: 16 }}
                      />
                    )}
                  </Card>
                </Col>

                {/* Step 2 & 3: Upload Model & Start Training in Same Container */}
                <Col xs={24}>
                  <Card
                    className="mb-4"
                    style={{
                      borderRadius: '16px',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                      textAlign: 'left',
                      opacity: selectedModel ? 1 : 0.6
                    }}
                  >
                    <Title level={3} style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#1890ff', marginBottom: '1rem', textAlign: 'center' }}>
                      Step 2 & 3: Upload Model & Start Training
                    </Title>

                    {/* Upload Section */}
                    <div style={{ marginBottom: 24 }}>
                      <Title level={4} style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#1890ff', marginBottom: 12 }}>
                        Upload Custom Model
                      </Title>
                      <Paragraph style={{ marginBottom: 8, color: '#666' }}>
                        Upload your custom model file. Accepts <code>.pt</code>, <code>.pth</code>, <code>.bin</code> (primary). Optionally accepts <code>.json</code>, <code>.config</code>, <code>.ckpt</code> for metadata/checkpoints. Max 500&nbsp;MB.
                      </Paragraph>
                      <input
                        type="file"
                        accept=".pt,.pth,.bin,.json,.config,.ckpt"
                        onChange={handleCustomModelUpload}
                        disabled={uploadingFile || training || !selectedModel || (uploadedModelPath !== null)}
                        style={{ 
                          width: '100%', 
                          margin: '8px 0', 
                          padding: '8px', 
                          border: '1px solid #d9d9d9', 
                          borderRadius: '4px',
                          opacity: (uploadedModelPath !== null) ? 0.6 : 1,
                          cursor: (uploadedModelPath !== null) ? 'not-allowed' : 'pointer'
                        }}
                      />
                      {(uploadedModelPath !== null) && (
                        <Alert
                          style={{ marginTop: 12 }}
                          type="info"
                          showIcon
                          message="Upload disabled"
                          description="A model has already been uploaded. Click 'Remove Uploaded Model' or 'Train Another Model' to upload a different model."
                        />
                      )}
                      <small style={{ color: '#999' }}>Maximum upload size: 500MB</small>
                      {!selectedModel && (
                        <Alert
                          style={{ marginTop: 12 }}
                          type="warning"
                          showIcon
                          message="Select a baseline model first"
                          description="Please select a baseline model from Step 1 before uploading."
                        />
                      )}
                      <Alert
                        style={{ marginTop: 12 }}
                        type="warning"
                        showIcon
                        message="Strict rule"
                        description="Do not upload DistilBERT, ResNet-18, MobileNetV2, or T5 Small. The system blocks built-in models."
                      />
                      {uploadStatus === "uploading" && (
                        <Alert
                          style={{ marginTop: 12 }}
                          type="info"
                          showIcon
                          message="Uploading..."
                          description="Validating file type, size, and uniqueness."
                        />
                      )}
                      {uploadStatus === "success" && uploadedModelName && (
                        <Alert
                          style={{ marginTop: 12 }}
                          type="success"
                          showIcon
                          message={`Ready: ${uploadedModelName}`}
                          description={`Stored at ${formatBytes(uploadedModelSize)}. You can now start training.`}
                        />
                      )}
                      {uploadError && (
                        <Alert
                          style={{ marginTop: 12 }}
                          type="error"
                          showIcon
                          message="Upload blocked"
                          description={uploadError}
                        />
                      )}
                      {uploadedModelPath && (
                        <div style={{ marginTop: 12 }}>
                          <Button size="small" danger onClick={clearUploadedModel} disabled={training}>
                            Remove Uploaded Model
                          </Button>
                        </div>
                      )}
                      {uploadStatus === "success" && uploadedModelPath && (
                        <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
                          <Button
                            type="primary"
                            size="large"
                            onClick={() => startTraining()}
                            disabled={training || !selectedModel || !uploadedModelPath}
                            style={{ flex: 1 }}
                          >
                            {training ? "Training in Progress..." : "Start Training"}
                          </Button>
                        </div>
                      )}
                    </div>

                    <Divider />

                    {/* Training Progress Section */}
                    <div style={{ marginBottom: 24 }}>
                      <Title level={4} style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#1890ff', marginBottom: 12 }}>
                        Training Progress
                      </Title>
                      <Progress
                        percent={progress}
                        status={training ? "active" : progress === 100 ? "success" : "normal"}
                        style={{ marginBottom: 16 }}
                        strokeColor={training ? "#1890ff" : progress === 100 ? "#52c41a" : "#d9d9d9"}
                      />
                      {currentLoss && (
                        <div style={{ marginBottom: 12, padding: '8px', background: '#f0f8ff', borderRadius: '6px' }}>
                          <Text strong style={{ color: '#1890ff' }}>Current Loss: </Text>
                          <Text>{currentLoss}</Text>
                        </div>
                      )}
                      {training && trainingPhase && (
                        <div style={{ marginBottom: 16, padding: "16px", background: "#f0f8ff", borderRadius: "8px", border: "1px solid #d6e4ff" }}>
                          <div style={{ display: "flex", alignItems: "center", marginBottom: "8px" }}>
                            <div style={{
                              width: "12px",
                              height: "12px",
                              borderRadius: "50%",
                              backgroundColor: "#1890ff",
                              marginRight: "8px",
                              animation: "pulse 1.5s infinite"
                            }}></div>
                            <Text strong style={{ color: "#1890ff", textTransform: "capitalize", fontSize: '16px' }}>
                              {trainingPhase.replace(/_/g, " ")}
                            </Text>
                          </div>
                          {trainingMessage && (
                            <Text style={{ color: "#666", fontSize: "14px", display: 'block', marginTop: 8 }}>
                              {trainingMessage}
                            </Text>
                          )}
                        </div>
                      )}
                      {!training && progress === 0 && (
                        <Alert
                          type="info"
                          showIcon
                          message="Ready to Train"
                          description="Select a baseline reference, upload your custom model, and the KD → pruning pipeline will start automatically."
                        />
                      )}
                    </div>

                    <Divider />

                    {/* Manual training info / cancel */}
                    <div style={{ textAlign: 'center' }}>
                      <Alert
                        type="info"
                        showIcon
                        message="Manual Training"
                        description="Click the 'Start Training' button above to begin Knowledge Distillation followed by pruning. Monitor progress and metrics here."
                        style={{ marginBottom: 16 }}
                      />
                      {training && (
                        <Button
                          type="default"
                          size="large"
                          onClick={cancelTraining}
                          danger
                          style={{ minWidth: 180 }}
                        >
                          Cancel Training
                        </Button>
                      )}
                    </div>
                  </Card>
                </Col>

                {/* Comparison Metrics - only show after training is complete */}
                {(trainingComplete && metrics && !trainingError) && (
                  <>
                    {/* TrainingComparison Component - displays socket-emitted computation details */}
                    {computationDetails && (
                      <Col xs={24}>
                        <TrainingComparison 
                          computationDetails={computationDetails} 
                          socket={socket}
                        />
                      </Col>
                    )}

                    <Col xs={24}>
                      <Card
                        style={{
                          borderRadius: '16px',
                          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                        }}
                        title={
                          <Title level={3} style={{ margin: 0 }}>
                            Comparison Metrics: {getSelectedModelLabel(selectedModel)} vs {uploadedModelName || "Uploaded Model"}
                          </Title>
                        }
                      >
                        {(() => {
                          const baselineMetricsData = baselineReferenceMetrics[selectedModel] || null;
                          const studentPerformance = metrics?.model_performance?.metrics || {};
                          const baselineLabel = baselineMetricsData?.label || getSelectedModelLabel(selectedModel);
                          const studentAccuracy = studentPerformance?.accuracy || metrics?.evaluation_metrics?.effectiveness?.[0]?.after || 'N/A';
                          const studentF1 = studentPerformance?.f1_score || metrics?.evaluation_metrics?.effectiveness?.[3]?.after || 'N/A';
                          const studentLatency = studentPerformance?.latency_ms || metrics?.evaluation_metrics?.efficiency?.[0]?.after || 'N/A';
                          const studentComplexity = studentPerformance?.num_params ? `${studentPerformance.num_params} params` : 'N/A';
                          const studentSizeLabel = studentPerformance?.size_mb || 'N/A';
                          const studentSizeValue = studentPerformance?.size_mb ? parseFloat(studentPerformance.size_mb) : null;
                          const baselineSizeValue = baselineMetricsData?.sizeMB;
                          const computedSizeReduction = (baselineSizeValue && studentSizeValue)
                            ? ((baselineSizeValue - studentSizeValue) / baselineSizeValue) * 100
                            : null;

                          const comparisonRows = [
                            {
                              key: "accuracy",
                              label: "Accuracy",
                              baseline: baselineMetricsData ? formatPercentValue(baselineMetricsData.accuracy) : "N/A",
                              uploaded: studentAccuracy,
                              formula: "Accuracy = Correct Predictions / Total Samples"
                            },
                            {
                              key: "f1_score",
                              label: "F1-Score",
                              baseline: baselineMetricsData ? formatPercentValue(baselineMetricsData.f1) : "N/A",
                              uploaded: studentF1,
                              formula: "F1 = 2 × (Precision · Recall) / (Precision + Recall)"
                            },
                            {
                              key: "size_reduction",
                              label: "Model Size Reduction",
                              baseline: baselineMetricsData ? `${baselineMetricsData.sizeMB.toFixed(1)} MB` : "N/A",
                              uploaded: studentSizeLabel === 'N/A'
                                ? "N/A"
                                : `${studentSizeLabel}${computedSizeReduction !== null ? ` (${computedSizeReduction.toFixed(2)}% ↓)` : ""}`,
                              formula: "SizeReduction(%) = (Size_teacher − Size_student) / Size_teacher × 100"
                            },
                            {
                              key: "latency",
                              label: "Inference Latency",
                              baseline: baselineMetricsData ? `${baselineMetricsData.latencyMs} ms` : "N/A",
                              uploaded: studentLatency,
                              formula: "Latency = Total Inference Time / Number of Samples"
                            },
                            {
                              key: "complexity",
                              label: "Model Complexity",
                              baseline: baselineMetricsData ? formatParamValue(baselineMetricsData.params) : "N/A",
                              uploaded: studentComplexity,
                              formula: "Complexity = Count of trainable parameters"
                            }
                          ];

                          return (
                            <>
                              <Paragraph style={{ color: '#000000' }}>
                                Baseline metrics for {baselineLabel} come from fixed KD + pruning runs. Your uploaded model is distilled, pruned,
                                and then compared against that reference using the required evaluation formulas.
                              </Paragraph>
                              <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', color: '#000000' }}>
                                  <thead>
                                    <tr>
                                      <th style={{ textAlign: 'left', padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.2)', color: '#000000', fontWeight: 'bold' }}>Metric</th>
                                      <th style={{ textAlign: 'left', padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.2)', color: '#000000', fontWeight: 'bold' }}>{baselineLabel} (Reference)</th>
                                      <th style={{ textAlign: 'left', padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.2)', color: '#000000', fontWeight: 'bold' }}>{uploadedModelName || "Uploaded Model (Training Results)"}</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {comparisonRows.map((row) => (
                                      <tr key={row.key}>
                                        <td style={{ padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.1)', color: '#000000' }}>
                                          <strong style={{ color: '#000000' }}>{row.label}</strong>
                                          <div style={{ fontSize: '12px', color: '#333333', marginTop: 4 }}>{row.formula}</div>
                                        </td>
                                        <td style={{ padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.1)', color: '#000000' }}>{row.baseline}</td>
                                        <td style={{ padding: '12px', borderBottom: '1px solid rgba(0, 0, 0, 0.1)', color: '#000000' }}>{row.uploaded}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </>
                          );
                        })()}
                      </Card>
                    </Col>

                    {/* Computation Explanation */}
                    <Col xs={24}>
                      <Card
                        style={{
                          borderRadius: '16px',
                          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                          marginTop: 16
                        }}
                        title={<Title level={3} style={{ margin: 0 }}>How the Computation Works</Title>}
                      >
                        <Row gutter={[16, 16]}>
                          <Col xs={24} md={12}>
                            <Title level={4} style={{ color: '#1890ff' }}>Knowledge Distillation Process</Title>
                            <Paragraph>
                              <strong>Step 1:</strong> Your uploaded checkpoint acts as the teacher. A lightweight student is generated automatically and receives softened targets with temperature scaling (T=2.0).
                            </Paragraph>
                            <Paragraph>
                              <strong>Step 2:</strong> The student minimizes a blended loss (α·CE + (1−α)·KD) across {metrics?.knowledge_distillation_analysis?.process?.training_steps || 50} epochs, matching the teacher's logits while respecting ground-truth labels.
                            </Paragraph>
                            <Paragraph>
                              <strong>Step 3:</strong> This transfer captures the teacher's decision boundaries, enabling a compact model with comparable accuracy.
                            </Paragraph>
                            <Paragraph>
                              <strong>Result:</strong> The distilled student retains most of the teacher's performance while being smaller and faster; we then compare it against your chosen baseline reference.
                            </Paragraph>
                          </Col>
                          <Col xs={24} md={12}>
                            <Title level={4} style={{ color: '#fa8c16' }}>Model Pruning Process</Title>
                            <Paragraph>
                              <strong>Step 1:</strong> After KD, L1 unstructured pruning is applied to remove {metrics?.pruning_analysis?.pruning_details?.pruning_ratio || '30%'} of weights with the smallest magnitudes.
                            </Paragraph>
                            <Paragraph>
                              <strong>Step 2:</strong> Pruning targets Linear and Convolutional layers, removing connections that contribute least to model performance.
                            </Paragraph>
                            <Paragraph>
                              <strong>Step 3:</strong> The pruned model is fine-tuned to adapt to the new structure and recover any lost performance.
                            </Paragraph>
                            <Paragraph>
                              <strong>Result:</strong> Model size is reduced by {metrics?.efficiency_improvements?.improvements?.storage?.reduction || 'N/A'}, inference speed improves by {metrics?.efficiency_improvements?.improvements?.speed?.improvement || 'N/A'}, while maintaining {metrics?.model_performance?.metrics?.accuracy || 'N/A'} accuracy.
                            </Paragraph>
                          </Col>
                        </Row>
                        <Alert
                          type="info"
                          message="Training Process"
                          description={
                            <div>
                              <p><strong>Backend Training Log:</strong></p>
                              <ul style={{ marginTop: 8, paddingLeft: 20 }}>
                                <li>Knowledge Distillation: {metrics?.knowledge_distillation_analysis?.process?.training_steps || 50} epochs with temperature T=2.0</li>
                                <li>Pruning: {metrics?.pruning_analysis?.pruning_details?.pruning_ratio || '30%'} L1 unstructured pruning</li>
                                <li>Fine-tuning: 20 additional epochs to stabilize pruned structure</li>
                                <li>Evaluation: Real metrics computed from actual model performance</li>
                              </ul>
                              <p style={{ marginTop: 12, marginBottom: 0 }}>
                                All metrics shown above are calculated from real training runs, not simulated values.
                              </p>
                            </div>
                          }
                          showIcon
                          style={{ marginTop: 16 }}
                        />
                      </Card>
                    </Col>
                  </>
                )}

                {/* Training Results - only show after training is complete */}
                {(trainingComplete && metrics && !trainingError) && (
                  <Col xs={24}>
                    <Card
                      style={{
                        borderRadius: '16px',
                        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                        minHeight: 200,
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between'
                      }}
                      title="Training Results"
                      bodyStyle={{ minHeight: 120, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
                    >
                      <div style={{ flex: 1 }}>{renderResultsPage()}</div>
                      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
                        <Button
                          onClick={() => setResultsPage((p) => Math.max(0, p - 1))}
                          disabled={resultsPage === 0}
                          size="small"
                        >
                          Previous
                        </Button>
                        <span style={{ alignSelf: "center", fontSize: '12px' }}>
                          Page {resultsPage + 1} / {resultsPagesCount}
                        </span>
                        <Button
                          onClick={() => setResultsPage((p) => Math.min(resultsPagesCount - 1, p + 1))}
                          disabled={resultsPage === resultsPagesCount - 1}
                          size="small"
                        >
                          Next
                        </Button>
                      </div>
                    </Card>
                  </Col>
                )}
              </Row>
            </Col>
          </Row>

          {/* Training Computation Comparison */}
          {computationDetails && (
            <Row justify="center">
              <Col xs={24} lg={16}>
                <TrainingComparison computationDetails={computationDetails} socket={socket} />
              </Col>
            </Row>
          )}

          {/* Error message if training fails */}
          {trainingError && (
            <Row justify="center">
              <Col xs={24} sm={20} md={16} lg={12} xl={10}>
                <Alert
                  message="Training Error"
                  description={trainingError}
                  type="error"
                  showIcon
                  style={{ marginTop: 24 }}
                />
              </Col>
            </Row>
          )}

          {/* Action Buttons at Bottom of Page - only show after training is complete */}
          {(trainingComplete && metrics && !trainingError) && (
            <Row justify="center" style={{ marginTop: 40, marginBottom: 40 }}>
              <Col xs={24} sm={20} md={16} lg={12} xl={10}>
                <Card
                  style={{
                    borderRadius: '16px',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                    textAlign: 'center'
                  }}
                >
                  <Title level={4} style={{ marginBottom: 20 }}>
                    Next Steps
                  </Title>
                  <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
                    <Button
                      type="default"
                      size="large"
                      onClick={resetForNewTraining}
                      style={{
                        borderRadius: 999,
                        padding: '0 32px',
                        fontWeight: 500,
                        height: 45
                      }}
                    >
                      Train Another Model
                    </Button>
                    <Button
                      type="primary"
                      size="large"
                      icon={<ArrowRightOutlined />}
                      onClick={() => navigate("/visualization", {
                        state: {
                          fromTraining: true,
                          selectedModel,
                          uploadedModelName,
                          latestMetrics: metrics
                        }
                      })}
                      style={{
                        borderRadius: 999,
                        padding: '0 32px',
                        fontWeight: 500,
                        height: 45
                      }}
                    >
                      Proceed to Visualization
                    </Button>
                  </div>
                </Card>
              </Col>
            </Row>
          )}
        </Content>
      </Layout>
      <Footer />
    </>
  );
};

export default Training;


