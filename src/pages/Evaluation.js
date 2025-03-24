import React, { useState, useContext } from "react";
import { Layout, Card, Button, Table, message, Spin, Typography, Tooltip } from "antd";
import { BarChartOutlined, ArrowLeftOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Title, Paragraph } = Typography;
const { Header, Content, Footer } = Layout;

const Evaluation = () => {
  const navigate = useNavigate();
  const { uploadedFile, setEvaluationResults } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  const evaluateModel = async () => {
    if (!uploadedFile) {
      message.error("🚨 Please upload a dataset before evaluating.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });
      const data = await response.json();

      if (response.ok && data.success) {
        message.success("✅ Evaluation completed successfully!");
        setResults(data.results);
        setEvaluationResults(data.results);
      } else {
        message.error(data.error || "❌ Evaluation failed. Try again.");
      }
    } catch (error) {
      message.error("🚨 Error connecting to server.");
    } finally {
      setLoading(false);
    }
  };

  const fetchDistillationMetrics = async () => {
    if (!uploadedFile) {
      message.error("🚨 Please upload a dataset before fetching metrics.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/distill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });
      const data = await response.json();

      if (response.ok && data.success) {
        message.success("✅ Distillation metrics fetched successfully!");
        setResults(Object.entries(data.metrics).map(([key, value]) => ({ metric: key, value }))); // Map metrics to table format
      } else {
        message.error(data.error || "❌ Failed to fetch distillation metrics.");
      }
    } catch (error) {
      message.error("🚨 Error connecting to server.");
    } finally {
      setLoading(false);
    }
  };

  const fetchPruningMetrics = async () => {
    if (!uploadedFile) {
      message.error("🚨 Please upload a dataset before fetching metrics.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/prune", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });
      const data = await response.json();

      if (response.ok && data.success) {
        message.success("✅ Pruning metrics fetched successfully!");
        setResults(Object.entries(data.metrics).map(([key, value]) => ({ metric: key, value }))); // Map metrics to table format
      } else {
        message.error(data.error || "❌ Failed to fetch pruning metrics.");
      }
    } catch (error) {
      message.error("🚨 Error connecting to server.");
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    { title: "📌 Metric", dataIndex: "metric", key: "metric", width: "50%", align: "center" },
    { title: "📊 Value", dataIndex: "value", key: "value", width: "50%", align: "center" },
  ];

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
          title="📊 Model Evaluation"
          bordered={false}
          style={{ maxWidth: 600, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
        >
          <Paragraph>
            Evaluate the performance of the trained model using key metrics:
          </Paragraph>
          <ul style={{ textAlign: "left", marginBottom: "20px" }}>
            <li><b>Accuracy:</b> The percentage of correct predictions made by the model.</li>
            <li><b>Precision:</b> The proportion of true positive predictions out of all positive predictions.</li>
            <li><b>Recall:</b> The proportion of true positive predictions out of all actual positives.</li>
          </ul>

          <Tooltip title="Ensure training is complete before running evaluation.">
            <InfoCircleOutlined style={{ fontSize: 18, color: "#1890ff", marginBottom: 10 }} />
          </Tooltip>

          <Button type="primary" icon={<BarChartOutlined />} onClick={evaluateModel} disabled={loading} style={{ width: "100%" }}>
            {loading ? "Evaluating..." : "Run Evaluation"}
          </Button>

          <Button type="primary" icon={<BarChartOutlined />} onClick={fetchDistillationMetrics} disabled={loading} style={{ width: "100%", marginTop: 20 }}>
            Fetch Distillation Metrics
          </Button>

          <Button type="primary" icon={<BarChartOutlined />} onClick={fetchPruningMetrics} disabled={loading} style={{ width: "100%", marginTop: 20 }}>
            Fetch Pruning Metrics
          </Button>

          {loading && <Spin size="large" style={{ marginTop: 20 }} />}

          {results.length > 0 && (
            <>
              <Table dataSource={results} columns={columns} pagination={false} style={{ marginTop: 20 }} bordered />
              <Button type="primary" style={{ marginTop: 20, width: "100%" }} onClick={() => navigate("/visualization")}>
                Proceed to Visualization ➡️
              </Button>
            </>
          )}
        </Card>
      </Content>

      <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px" }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
};

export default Evaluation;