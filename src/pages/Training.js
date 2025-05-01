import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography } from "antd";
import { PlayCircleOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate, Link } from "react-router-dom";
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
  const { uploadedFile } = useContext(UploadContext);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [selectedModel, setSelectedModel] = useState("distillBert");

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
              <Nav.Link as={Link} to="/upload">Models</Nav.Link>
              <Nav.Link as={Link} to="/training">Training</Nav.Link>
              <Nav.Link as={Link} to="/evaluation">Evaluation</Nav.Link>
              <Nav.Link as={Link} to="/visualization">Visualization</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

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
              variant="dark" // Set the dropdown button to black
            >
              <Dropdown.Item as="button" onClick={() => handleModelSelect("distillBert")}>distillBert</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("T5-small")}>T5-small</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("MobileNetV2")}>MobileNetV2</Dropdown.Item>
              <Dropdown.Item as="button" onClick={() => handleModelSelect("ResNet-18")}>ResNet-18</Dropdown.Item>
            </DropdownButton>
          </Card>

          
        </Content>

        
      </Layout>
    </>
  );
};

export default Training;


