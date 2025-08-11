import React, { useState, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography } from "antd";
import { PlayCircleOutlined } from "@ant-design/icons";
import { useNavigate, Link } from "react-router-dom";
import { io } from "socket.io-client";
import { Navbar, Nav, Container, DropdownButton, Dropdown } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Training.css";

const { Title } = Typography;
const { Content } = Layout;

// Initialize socket with error handling
const SOCKET_URL = "http://127.0.0.1:5001";
const socket = io(SOCKET_URL, { 
  transports: ["websocket", "polling"],
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  timeout: 20000,
  forceNew: true,
  autoConnect: true
});

const Training = () => {
  const navigate = useNavigate();
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [selectedModel, setSelectedModel] = useState("distillBert");
  const [socketConnected, setSocketConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");
  const [retryCount, setRetryCount] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const modelData = {
    distillBert: {
      description: "DistilBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks. It's 40% smaller than BERT while retaining 97% of its language understanding capabilities.",
      metrics: {
        f1Score: "92%",
        accuracy: "90%",
        sizeReduction: "40%",
        latency: "50ms",
        complexity: "Medium",
      },
    },
    "T5-small": {
      description: "T5-small is a smaller version of the T5 (Text-to-Text Transfer Transformer) model, capable of performing a wide range of NLP tasks by converting them into a text-to-text format.",
      metrics: {
        f1Score: "88%",
        accuracy: "85%",
        sizeReduction: "35%",
        latency: "70ms",
        complexity: "High",
      },
    },
    MobileNetV2: {
      description: "MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and object detection on mobile and embedded devices.",
      metrics: {
        f1Score: "85%",
        accuracy: "83%",
        sizeReduction: "50%",
        latency: "30ms",
        complexity: "Low",
      },
    },
    "ResNet-18": {
      description: "ResNet-18 is a deep residual network with 18 layers, known for its ability to train very deep networks by using skip connections to avoid vanishing gradients.",
      metrics: {
        f1Score: "90%",
        accuracy: "88%",
        sizeReduction: "25%",
        latency: "60ms",
        complexity: "High",
      },
    },
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    message.info(`Selected model: ${model}`);
  };

  const startTraining = async () => {
    if (!socketConnected) {
      message.error("⚠️ Not connected to training server. Please try again.");
      return;
    }

    if (training) {
      message.warning("⚠️ Training is already in progress.");
      return;
    }

    setTraining(true);
    setProgress(0);
    setTrainingComplete(false);
    setCurrentLoss(null);
    setMetrics(null);

    try {
      console.log("Starting training with model:", selectedModel);
      message.loading({ 
        content: "Initializing models...", 
        key: "training",
        duration: 0
      });
      
      // First test if the model can be loaded
      const testResponse = await fetch(`${SOCKET_URL}/test_model`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ 
          model_name: selectedModel 
        }),
      });

      const testData = await testResponse.json();
      console.log("Model test response:", testData);

      if (!testResponse.ok || !testData.success) {
        throw new Error(testData.error || "Failed to load model");
      }

      message.loading({ 
        content: "Starting training...", 
        key: "training",
        duration: 0
      });

      // If model loading test passes, start training
      const response = await fetch(`${SOCKET_URL}/train`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ 
          model_name: selectedModel 
        }),
      });

      const data = await response.json();
      console.log("Training response:", data);

      if (response.ok && data.success) {
        message.success({ 
          content: "✅ Training has started in the background!", 
          key: "training",
          duration: 3
        });
      } else {
        throw new Error(data.error || "Failed to start training");
      }
    } catch (error) {
      console.error("Training Error:", error);
      message.error({ 
        content: `🚨 Error: ${error.message}`, 
        key: "training",
        duration: 5
      });
      setTraining(false);
      setProgress(0);
    }
  };

  // Function to manually reconnect
  const reconnectSocket = () => {
    console.log("Attempting to reconnect...");
    socket.connect();
    setRetryCount(prev => prev + 1);
  };

  // Test server connection
  const testServerConnection = async () => {
    try {
      const response = await fetch(`${SOCKET_URL}/test`);
      const data = await response.json();
      if (data.status === "Server is running") {
        setServerStatus("connected");
        message.success("Server is running");
        // Try to reconnect socket if it's disconnected
        if (!socketConnected) {
          reconnectSocket();
        }
      } else {
        setServerStatus("error");
        message.error("Server is not responding correctly");
      }
    } catch (error) {
      console.error("Server test error:", error);
      setServerStatus("error");
      message.error("Cannot connect to server. Please make sure the server is running.");
    }
  };

  useEffect(() => {
    // Test server connection on component mount
    testServerConnection();

    // Socket connection handlers
    socket.on("connect", () => {
      console.log("Socket connected successfully");
      setSocketConnected(true);
      setServerStatus("connected");
      message.success("Connected to training server");
    });

    socket.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
      setSocketConnected(false);
      setServerStatus("error");
      message.error("Failed to connect to training server. Please make sure the server is running.");
    });

    socket.on("disconnect", (reason) => {
      console.log("Socket disconnected:", reason);
      setSocketConnected(false);
      setServerStatus("error");
      message.warning("Disconnected from training server. Please refresh the page.");
    });

    socket.on("training_progress", (data) => {
      console.log("Training progress data received:", data);
      if (data.progress !== undefined) {
        setProgress(data.progress);
      }
      if (data.loss !== undefined) {
        setCurrentLoss(data.loss.toFixed(4));
      }
      if (data.metrics) {
        console.log("Received metrics:", data.metrics);
        setMetrics(data.metrics);
      }
      if (data.progress === 100) {
        setTrainingComplete(true);
        setTraining(false);
        message.success("✅ Training completed successfully!");
      }
    });

    socket.on("training_error", (data) => {
      console.error("Training error from server:", data.error);
      setTraining(false);
      setProgress(0);
      message.error({
        content: `🚨 Training Failed: ${data.error}`,
        key: "training",
        duration: 0,
      });
    });

    // Cleanup function
    return () => {
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
      socket.off("training_progress");
      socket.off("training_error");
    };
  }, []);

  // Add server status indicator to the UI with retry button
  const renderServerStatus = () => {
    return (
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <span style={{ 
          color: serverStatus === "connected" ? "green" : 
                 serverStatus === "error" ? "red" : "orange",
          marginRight: "10px"
        }}>
          ● {serverStatus === "connected" ? "Server Connected" : 
             serverStatus === "error" ? "Server Disconnected" : 
             "Checking Connection..."}
        </span>
        {serverStatus === "error" && (
          <Button 
            type="primary" 
            size="small" 
            onClick={reconnectSocket}
            style={{ marginLeft: "10px" }}
          >
            Retry Connection
          </Button>
        )}
      </div>
    );
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

      <Title style={{ marginTop: "30px" }}>Select A Model To Train</Title>
      {renderServerStatus()}

      <Layout>
        <Content style={{ display: "flex", flexDirection: "column", alignItems: "center", minHeight: "80vh", padding: "20px" }}>
          <Card
            title="Select Model"
            bordered={false}
            style={{ maxWidth: 400, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)", marginBottom: "20px" }}
          >
            <DropdownButton
              id="dropdown-item-button"
              title={`Selected Model: ${selectedModel}`}
              variant="dark"
            >
              <Dropdown.Item as="button" onClick={() => handleModelSelect("distillBert")}>DistilBERT</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("T5-small")}>T5-small</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("MobileNetV2")}>MobileNetV2</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("ResNet-18")}>ResNet-18</Dropdown.Item>
            </DropdownButton>
          </Card>

          <Card
            title="Model Description"
            bordered={false}
            style={{ maxWidth: 400, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)", marginBottom: "20px" }}
          >
            {modelData[selectedModel] ? (
              <>
                <p>{modelData[selectedModel].description}</p>
                <table style={{ width: "100%", marginTop: "10px" }}>
                  <tbody>
                    <tr>
                      <td><strong>F1-Score:</strong></td>
                      <td>{modelData[selectedModel].metrics.f1Score}</td>
                    </tr>
                    <tr>
                      <td><strong>Accuracy:</strong></td>
                      <td>{modelData[selectedModel].metrics.accuracy}</td>
                    </tr>
                    <tr>
                      <td><strong>Model Size Reduction:</strong></td>
                      <td>{modelData[selectedModel].metrics.sizeReduction}</td>
                    </tr>
                    <tr>
                      <td><strong>Inference Latency:</strong></td>
                      <td>{modelData[selectedModel].metrics.latency}</td>
                    </tr>
                    <tr>
                      <td><strong>Model Complexity:</strong></td>
                      <td>{modelData[selectedModel].metrics.complexity}</td>
                    </tr>
                  </tbody>
                </table>
              </>
            ) : (
              <p>No model selected or data unavailable.</p>
            )}
          </Card>

          <Card
            bordered={false}
            style={{ maxWidth: 400, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
          >
            <Progress
              percent={progress}
              status={training ? "active" : progress === 100 ? "success" : "normal"}
              style={{ marginBottom: "20px" }}
            />
            {currentLoss && (
              <p style={{ marginBottom: "10px" }}>
                Current Loss: {currentLoss}
              </p>
            )}
            {metrics && (
              <div style={{ marginBottom: "20px", textAlign: "left" }}>
                <h4>Model Performance</h4>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    <tr><td>Accuracy:</td><td>{metrics.model_performance.accuracy}</td></tr>
                    <tr><td>Precision:</td><td>{metrics.model_performance.precision}</td></tr>
                    <tr><td>Recall:</td><td>{metrics.model_performance.recall}</td></tr>
                    <tr><td>F1-Score:</td><td>{metrics.model_performance.f1_score}</td></tr>
                  </tbody>
                </table>

                <h4 style={{marginTop: "20px"}}>Distillation Metrics</h4>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    <tr><td>Teacher Accuracy:</td><td>{metrics.distillation_metrics.teacher_accuracy}</td></tr>
                    <tr><td>Student Accuracy:</td><td>{metrics.distillation_metrics.student_accuracy}</td></tr>
                    <tr><td>Size Reduction:</td><td>{metrics.distillation_metrics.size_reduction}</td></tr>
                    <tr><td>Memory Saved:</td><td>{metrics.distillation_metrics.memory_saved}</td></tr>
                  </tbody>
                </table>

                <h4 style={{marginTop: "20px"}}>Pruning Results</h4>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    <tr><td>Original Size:</td><td>{metrics.pruning_results.original_size}</td></tr>
                    <tr><td>Pruned Size:</td><td>{metrics.pruning_results.pruned_size}</td></tr>
                    <tr><td>Size Reduction:</td><td>{metrics.pruning_results.size_reduction}</td></tr>
                    <tr><td>Accuracy Impact:</td><td>{metrics.pruning_results.accuracy_impact}</td></tr>
                  </tbody>
                </table>

                <h4 style={{marginTop: "20px"}}>Efficiency Improvements</h4>
                 <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Metric</th>
                      <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Before</th>
                      <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>After</th>
                      <th style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Improvement</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Size</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.size.before}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.size.after}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd", color: "green" }}>
                        {metrics.efficiency_improvements.size.reduction}
                      </td>
                    </tr>
                    <tr>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Latency</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.latency.before}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.latency.after}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd", color: "green" }}>
                        {metrics.efficiency_improvements.latency.improvement}
                      </td>
                    </tr>
                    <tr>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>Parameters</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.parameters.before}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>{metrics.efficiency_improvements.parameters.after}</td>
                      <td style={{ padding: "8px", borderBottom: "1px solid #ddd", color: "green" }}>
                        {metrics.efficiency_improvements.parameters.reduction}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={startTraining}
              disabled={!socketConnected || training}
              style={{ 
                width: "100%", 
                backgroundColor: (!socketConnected || training) ? "#ccc" : "black",
                borderColor: (!socketConnected || training) ? "#ccc" : "black",
                cursor: (!socketConnected || training) ? "not-allowed" : "pointer"
              }}
            >
              {training ? "Training in Progress..." : "Start Training"}
            </Button>
            {!socketConnected && (
              <p style={{ color: "red", marginTop: "10px" }}>
                Not connected to training server
              </p>
            )}
          </Card>
        </Content>
      </Layout>
    </>
  );
};

export default Training;


