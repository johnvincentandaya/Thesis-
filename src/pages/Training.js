import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography } from "antd";
import { PlayCircleOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate, Link, useLocation } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";
import { io } from "socket.io-client";
import { Navbar, Nav, Container, DropdownButton, Dropdown } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Training.css";

const { Title, Paragraph } = Typography;
const { Header, Content, Footer } = Layout;

const socket = io("http://localhost:5000", { transports: ["websocket"] });

const Training = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { uploadedFile } = useContext(UploadContext);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [selectedModel, setSelectedModel] = useState(location.state?.selectedModel || "distillBert");

  const modelData = {
    distillBert: {
      description: "DistillBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks like text classification and question answering.",
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

  useEffect(() => {
    socket.on("connect", () => {
      console.log("Socket connected");
    });

    socket.on("training_progress", (data) => {
      console.log("Training progress:", data.progress);
      setProgress(data.progress);
      if (data.progress === 100) {
        setTrainingComplete(true);
        message.success("✅ Training completed successfully!");
      }
    });

    return () => {
      socket.off("training_progress");
      socket.off("connect");
    };
  }, []);

  const startTraining = async () => {
    if (!uploadedFile) {
      message.error("⚠️ Please upload a dataset before starting training.");
      return;
    }

    setTraining(true);
    setProgress(0);
    setTrainingComplete(false);

    try {
      const response = await fetch("http://localhost:5000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });

      const data = await response.json();
      if (response.ok && data.success) {
        message.success("✅ Training started successfully!");
      } else {
        message.error(data.error || "❌ Training failed. Please try again.");
        setTraining(false);
        setProgress(0);
      }
    } catch (error) {
      console.error("Training Error:", error);
      message.error("🚨 Error connecting to server.");
      setTraining(false);
      setProgress(0);
    }
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
              <Dropdown.Item as="button" onClick={() => handleModelSelect("distillBert")}>distillBert</Dropdown.Item>
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
            {modelData[selectedModel] ? ( // Ensure selectedModel exists in modelData
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
              <p>No model selected or data unavailable.</p> // Fallback message
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
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={startTraining}
              disabled={!selectedModel || training}
              style={{ width: "100%", backgroundColor: "black", borderColor: "black" }}
            >
              {training ? "Training in Progress..." : "Start Training"}
            </Button>
          </Card>
        </Content>
      </Layout>
    </>
  );
};

export default Training;


