import React, { useState, useContext, useEffect } from "react";
import { Layout, Card, Button, Table, message, Spin, Typography } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Title } = Typography;
const { Header, Content, Footer } = Layout;

const Evaluation = () => {
  const navigate = useNavigate();
  const { uploadedFile, setEvaluationResults } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetchAllMetrics();
  }, [uploadedFile]);

  const fetchAllMetrics = async () => {
    if (!uploadedFile) {
      message.error("🚨 Please upload a dataset first.");
      return;
    }

    setLoading(true);
    try {
      // Fetch all metrics in parallel
      const [evalResponse, distillResponse, pruneResponse] = await Promise.all([
        fetch("http://localhost:5000/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_path: uploadedFile }),
        }),
        fetch("http://localhost:5000/distill", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_path: uploadedFile }),
        }),
        fetch("http://localhost:5000/prune", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_path: uploadedFile }),
        })
      ]);

      const [evalData, distillData, pruneData] = await Promise.all([
        evalResponse.json(),
        distillResponse.json(),
        pruneResponse.json()
      ]);

      const combinedResults = [
        { title: "Model Performance", results: evalData.results },
        { title: "Distillation Metrics", results: distillData.results },
        { title: "Pruning Results", results: pruneData.results }
      ];

      setResults(combinedResults);
      setEvaluationResults(combinedResults);
      message.success("✅ Model evaluation completed!");
    } catch (error) {
      console.error("Evaluation Error:", error);
      message.error("🚨 Error fetching evaluation metrics");
    } finally {
      setLoading(false);
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

      <Content style={{ padding: "20px" }}>
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          <Card title="📊 Overall Model Evaluation" bordered={false}>
            {loading ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <Spin size="large" />
                <p>Calculating metrics...</p>
              </div>
            ) : (
              results.map((section, idx) => (
                <div key={idx} style={{ marginBottom: 24 }}>
                  <Title level={4}>{section.title}</Title>
                  <Table 
                    dataSource={section.results}
                    columns={[
                      { title: "Metric", dataIndex: "metric", key: "metric" },
                      { title: "Value", dataIndex: "value", key: "value" }
                    ]}
                    pagination={false}
                    bordered
                  />
                </div>
              ))
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