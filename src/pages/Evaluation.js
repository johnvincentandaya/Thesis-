import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, message, Spin, Typography, Table } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Title } = Typography;
const { Header, Content, Footer } = Layout;

const defaultEvaluationData = {
  effectiveness: [
    { metric: "Accuracy", before: "91.2%", after: "89.0%" },
    { metric: "Precision (Macro Avg)", before: "91.1%", after: "88.8%" },
    { metric: "Recall (Macro Avg)", before: "91.0%", after: "88.5%" },
    { metric: "F1-Score (Macro Avg)", before: "91.0%", after: "88.6%" },
  ],
  efficiency: [
    { metric: "Latency (ms)", before: "14.5 ms", after: "6.1 ms" },
    { metric: "RAM Usage (MB)", before: "228.7 MB", after: "124.2 MB" },
    { metric: "Model Size (MB)", before: "2.4 MB", after: "1.1 MB" },
  ],
  compression: [
    { metric: "Parameters Count", before: "72,000", after: "28,000" },
    { metric: "Layers Count", before: "3", after: "3" },
    { metric: "Compression Ratio", before: "Not Applicable", after: "2.6×" },
    { metric: "Accuracy Drop (%)", before: "Not Applicable", after: "2.2%" },
    { metric: "Size Reduction (%)", before: "Not Applicable", after: "54.2%" },
  ],
  complexity: [
    { metric: "Time Complexity", before: "Not Applicable", after: "O(n²)" },
    { metric: "Space Complexity", before: "Not Applicable", after: "O(n)" },
  ],
};

const Evaluation = () => {
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetchAllMetrics();
  }, [uploadedFile]);

  const fetchAllMetrics = async () => {
    if (!uploadedFile) {
      message.warning("⚠️ No dataset uploaded. Using default evaluation values.");
      setResults([
        { title: "Effectiveness Metrics", results: defaultEvaluationData.effectiveness },
        { title: "Efficiency Metrics", results: defaultEvaluationData.efficiency },
        { title: "Compression Metrics", results: defaultEvaluationData.compression },
        { title: "Complexity Metrics", results: defaultEvaluationData.complexity },
      ]);
      return;
    }

    setLoading(true);
    try {
      const evalResponse = await fetch("http://localhost:5000/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadedFile }),
      });

      const evalData = await evalResponse.json();
      setResults([
        { title: "Effectiveness Metrics", results: evalData.effectiveness },
        { title: "Efficiency Metrics", results: evalData.efficiency },
        { title: "Compression Metrics", results: evalData.compression },
        { title: "Complexity Metrics", results: evalData.complexity },
      ]);
      message.success("✅ Model evaluation completed!");
    } catch (error) {
      console.error("Evaluation Error:", error);
      message.error("🚨 Error fetching evaluation metrics. Using default values.");
      setResults([
        { title: "Effectiveness Metrics", results: defaultEvaluationData.effectiveness },
        { title: "Efficiency Metrics", results: defaultEvaluationData.efficiency },
        { title: "Compression Metrics", results: defaultEvaluationData.compression },
        { title: "Complexity Metrics", results: defaultEvaluationData.complexity },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const renderTable = (title, data) => {
    const columns = [
      {
        title: "Metric",
        dataIndex: "metric",
        key: "metric",
        width: "40%",
      },
      ...(title === "Complexity Metrics"
        ? [] // No "Before" column for Complexity Metrics
        : [
            {
              title: "Before",
              dataIndex: "before",
              key: "before",
              align: "center",
            },
          ]),
      {
        title: "After",
        dataIndex: "after",
        key: "after",
        align: "center",
      },
    ];

    return (
      <div style={{ marginBottom: 24 }}>
        <Title level={4}>{title}</Title>
        <Table
          columns={columns}
          dataSource={data}
          pagination={false}
          bordered
          rowKey={(record) => record.metric}
        />
      </div>
    );
  };

  return (
    <Layout>
      <Header style={{ background: "#001529", display: "flex", alignItems: "center", padding: "0 20px" }}>
        <Title level={3} style={{ color: "white", margin: "0", flex: 1 }}>KD-Pruning Simulator</Title>
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} style={{ color: "white" }}>
          Back
        </Button>
      </Header>

      <Content style={{ padding: "20px" }}>
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          <Card title="📊 Overall Model Evaluation" bordered={false}>
            {loading ? (
              <div style={{ textAlign: "center", padding: "20px" }}>
                <Spin size="large" />
                <p>Calculating metrics...</p>
              </div>
            ) : (
              results.map((section, idx) => renderTable(section.title, section.results))
            )}

            {results.length > 0 && (
              <Button
                type="primary"
                style={{ marginTop: 20, width: "100%" }}
                onClick={() => navigate("/visualization")}
              >
                View Network Visualization ➡️
              </Button>
            )}
          </Card>
        </div>
      </Content>

      <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px" }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
};

export default Evaluation;