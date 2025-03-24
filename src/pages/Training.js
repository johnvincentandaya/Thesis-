import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography, Tooltip } from "antd";
import { PlayCircleOutlined, ArrowLeftOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";
import { io } from "socket.io-client"; // ✅ Import Socket.IO client

const { Title, Paragraph } = Typography;
const { Header, Content, Footer } = Layout;

const socket = io("http://localhost:5000", { transports: ["websocket"] }); // Ensure WebSocket transport is used

const Training = () => {
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // ✅ Listen for real-time training progress updates
    socket.on("training_progress", (data) => {
      setProgress(data.progress);
      if (data.progress === 100) {
        message.success("✅ Model training completed successfully!");
        setTimeout(() => navigate("/evaluation"), 1000); // ✅ Navigate to evaluation after training
      }
    });

    return () => {
      socket.off("training_progress"); // ✅ Cleanup event listener on unmount
    };
  }, [navigate]);

  const startTraining = async () => {
    if (!uploadedFile) {
      message.error("⚠️ Please upload a dataset before starting training.");
      return;
    }

    setTraining(true);
    setProgress(0); // Reset progress before starting

    try {
      const response = await fetch("http://localhost:5000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });

      if (!response.ok) {
        message.error("❌ Training failed. Please try again.");
        setProgress(0);
        setTraining(false);
      }
    } catch (error) {
      message.error("🚨 Error connecting to server. Check your backend.");
      setProgress(0);
      setTraining(false);
    }
  };

  return (
    <Layout>
      {/* Navigation Bar */}
      <Header style={{ background: "#001529", display: "flex", alignItems: "center", padding: "0 20px" }}>
        <Title level={3} style={{ color: "white", margin: "0", flex: 1 }}>KD-Pruning Simulator</Title>
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} style={{ color: "white" }}>
          Back
        </Button>
      </Header>

      {/* Main Content */}
      <Content style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "80vh", padding: "20px" }}>
        <Card
          title="🚀 Train AI Model"
          bordered={false}
          style={{ maxWidth: 600, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
        >
          <Paragraph>
            Train the AI model using the latest dataset and configurations.
          </Paragraph>

          <Tooltip title="Ensure you've uploaded a dataset before starting training.">
            <InfoCircleOutlined style={{ fontSize: 18, color: "#1890ff", marginBottom: 10 }} />
          </Tooltip>

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={startTraining}
            disabled={training}
            style={{ width: "100%", marginTop: 10, transition: "0.3s" }}
          >
            {training ? "Training in Progress..." : "Start Training"}
          </Button>

          <Progress percent={progress} status={training ? "active" : "normal"} style={{ marginTop: 20 }} />
        </Card>
      </Content>

      {/* Footer */}
      <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px" }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
};

export default Training;
