import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, Progress, message, Typography } from "antd";
import { PlayCircleOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";
import { io } from "socket.io-client";

const { Title, Paragraph } = Typography;
const { Header, Content, Footer } = Layout;

const socket = io("http://localhost:5000", { transports: ["websocket"] });

const Training = () => {
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);

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
    <Layout>
      <Header style={{ background: "#001529", display: "flex", alignItems: "center", padding: "0 20px" }}>
        <Title level={3} style={{ color: "white", margin: "0", flex: 1 }}>KD-Pruning Simulator</Title>
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} style={{ color: "white" }}>
          Back
        </Button>
      </Header>

      <Content style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "80vh", padding: "20px" }}>
        <Card
          title="🚀 Train AI Model"
          bordered={false}
          style={{ maxWidth: 600, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
        >
          <Paragraph>
            Train the AI model using the latest dataset and configurations.
          </Paragraph>

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={startTraining}
            disabled={training || trainingComplete}
            style={{ width: "100%", marginTop: 10 }}
          >
            {training ? "Training in Progress..." : trainingComplete ? "Training Complete" : "Start Training"}
          </Button>

          <Progress percent={progress} status={training ? "active" : "normal"} style={{ marginTop: 20 }} />

          {trainingComplete && (
            <Button
              type="primary"
              onClick={() => navigate("/evaluation")}
              style={{ width: "100%", marginTop: 20 }}
            >
              Next Step: Evaluate Model ➡️
            </Button>
          )}
        </Card>
      </Content>

      <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px" }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
};

export default Training;


