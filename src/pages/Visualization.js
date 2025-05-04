import React, { useEffect, useRef, useState, useContext } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Button, Layout, Typography, Spin, message, Card, Table } from "antd";
import { ArrowLeftOutlined, DownloadOutlined } from "@ant-design/icons";
import { useNavigate, Link } from "react-router-dom";
import { Navbar, Nav, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import { UploadContext } from "../context/UploadContext";

const { Header, Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

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
    { metric: "Layers Count", before: "6", after: "6" },
    { metric: "Compression Ratio", before: "Not Applicable", after: "2.6×" },
    { metric: "Accuracy Drop (%)", before: "Not Applicable", after: "2.2%" },
    { metric: "Size Reduction (%)", before: "Not Applicable", after: "54.2%" },
  ],
  complexity: [
    { metric: "Time Complexity", before: "Not Applicable", after: "O(n²)" },
    { metric: "Space Complexity", before: "Not Applicable", after: "O(n)" },
  ],
};

const Visualization = () => {
  const mountRef = useRef(null);
  const kdMountRef = useRef(null);
  const pruningMountRef = useRef(null);
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);

  const defaultVisualizationData = {
    nodes: [
      // Input Layer (4 green nodes)
      { id: "input_1", x: 0, y: 1.5, z: 0, size: 0.5, color: "green" },
      { id: "input_2", x: 0, y: 0.5, z: 0, size: 0.5, color: "green" },
      { id: "input_3", x: 0, y: -0.5, z: 0, size: 0.5, color: "green" },
      { id: "input_4", x: 0, y: -1.5, z: 0, size: 0.5, color: "green" },

      // Hidden Layer 1 (16 yellow nodes)
      ...Array.from({ length: 16 }, (_, i) => ({
        id: `hidden1_${i + 1}`,
        x: 2,
        y: 7.5 - i,
        z: 0,
        size: 0.4,
        color: "yellow",
      })),

      // Hidden Layer 2 (12 yellow nodes)
      ...Array.from({ length: 12 }, (_, i) => ({
        id: `hidden2_${i + 1}`,
        x: 4,
        y: 5.5 - i,
        z: 0,
        size: 0.4,
        color: "yellow",
      })),

      // Hidden Layer 3 (8 red nodes, pruned)
      ...Array.from({ length: 8 }, (_, i) => ({
        id: `hidden3_${i + 1}`,
        x: 6,
        y: 3.5 - i,
        z: 0,
        size: i % 2 === 0 ? 0.3 : 0.2,
        color: "red",
        opacity: i % 2 === 0 ? 1 : 0.5,
      })),

      // Output Layer (3 blue nodes)
      { id: "output_1", x: 8, y: 1, z: 0, size: 0.5, color: "blue" },
      { id: "output_2", x: 8, y: 0, z: 0, size: 0.5, color: "blue" },
      { id: "output_3", x: 8, y: -1, z: 0, size: 0.5, color: "blue" },
    ],
    connections: [
      // Connections from Input Layer to Hidden Layer 1
      ...Array.from({ length: 4 }, (_, i) =>
        Array.from({ length: 16 }, (_, j) => ({
          source: { x: 0, y: 1.5 - i, z: 0 },
          target: { x: 2, y: 7.5 - j, z: 0 },
          color: "gray",
        }))
      ).flat(),

      // Connections from Hidden Layer 1 to Hidden Layer 2
      ...Array.from({ length: 16 }, (_, i) =>
        Array.from({ length: 12 }, (_, j) => ({
          source: { x: 2, y: 7.5 - i, z: 0 },
          target: { x: 4, y: 5.5 - j, z: 0 },
          color: "gray",
        }))
      ).flat(),

      // Connections from Hidden Layer 2 to Hidden Layer 3
      ...Array.from({ length: 12 }, (_, i) =>
        Array.from({ length: 8 }, (_, j) => ({
          source: { x: 4, y: 5.5 - i, z: 0 },
          target: { x: 6, y: 3.5 - j, z: 0 },
          color: "gray",
        }))
      ).flat(),

      // Connections from Hidden Layer 3 to Output Layer
      ...Array.from({ length: 8 }, (_, i) =>
        Array.from({ length: 3 }, (_, j) => ({
          source: { x: 6, y: 3.5 - i, z: 0 },
          target: { x: 8, y: 1 - j, z: 0 },
          color: "gray",
        }))
      ).flat(),
    ],
  };

  useEffect(() => {
    renderVisualization(defaultVisualizationData);
  }, []);

  const renderVisualization = (visualizationData) => {
    if (!mountRef.current) return;

    // Clean up any existing child nodes in mountRef
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 10);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth * 0.8, window.innerHeight * 0.8);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Render nodes
    visualizationData.nodes.forEach((node) => {
      const geometry = new THREE.SphereGeometry(node.size, 32, 32);
      const material = new THREE.MeshPhongMaterial({
        color: node.color,
        shininess: 100,
        specular: 0x444444,
        transparent: true,
        opacity: node.opacity || 1,
      });

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x, node.y, node.z);
      scene.add(sphere);
    });

    // Render connections
    visualizationData.connections.forEach((connection) => {
      const material = new THREE.LineBasicMaterial({
        color: connection.color,
        linewidth: 1,
      });
      const points = [
        new THREE.Vector3(connection.source.x, connection.source.y, connection.source.z),
        new THREE.Vector3(connection.target.x, connection.target.y, connection.target.z),
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, material);
      scene.add(line);
    });

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    animate();
  };

  const handleDownload = async () => {
    try {
      const response = await fetch("http://localhost:5000/download", {
        method: "GET",
      });

      if (!response.ok) {
        const errorData = await response.json();
        message.error(errorData.error || "Failed to download the file.");
        return;
      }

      // Create a blob and trigger the download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "compressed_model_and_results.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      message.success("✅ Download started!");
    } catch (error) {
      console.error("Download Error:", error);
      message.error("🚨 Failed to download the file. Please try again.");
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

      <Layout>
        <Content style={{ padding: "20px", textAlign: "center", background: "#f0f2f5", minHeight: "calc(100vh - 64px - 70px)" }}>
          <Title level={3}>📊 Knowledge Distillation & Pruning Visualization</Title>
          <Paragraph>
            This 3D visualization shows the <b>compressed student model</b> architecture after applying
            <b> knowledge distillation</b> and <b> pruning</b>. Each sphere is a neuron, and lines represent
            the connections between layers.
          </Paragraph>

          {loading && <Spin size="large" style={{ margin: "30px 0" }} />}
          <div ref={mountRef} style={{ display: "flex", justifyContent: "center", alignItems: "center", width: "100%", height: "80vh", background: "#fff", borderRadius: 10, marginBottom: "40px" }} />

          {/* Legend Section */}
          <Card
              title="Legend"
              bordered={false}
              style={{ maxWidth: 600, margin: "20px auto", textAlign: "left", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
          >
            <ul style={{ lineHeight: "1.8" }}>
                <li>
                    <b>🟢 Green Nodes:</b> Represent the <b>input layer</b>, where the model receives the features from the dataset.
                </li>
                <li>
                    <b>🟡 Yellow Nodes:</b> Represent the <b>hidden layers</b>, where the model processes the data and learns patterns. These nodes are fully active.
                </li>
                <li>
                    <b>🔴 Red Nodes:</b> Represent the <b>pruned hidden layer</b>, where some connections are removed to make the model smaller and faster. Smaller or transparent nodes indicate pruned neurons.
                </li>
                <li>
                    <b>🔵 Blue Nodes:</b> Represent the <b>output layer</b>, where the model predicts the final classes.
                </li>
                <li>
                    <b>Gray Lines:</b> Represent the <b>connections</b> (weights) between neurons in different layers.
                </li>
            </ul>
          </Card>

          {/* Evaluation Tables */}
          <div style={{ maxWidth: 800, margin: "40px auto" }}>
            <Card title="📊 Evaluation Metrics" bordered={false}>
              {renderTable("Effectiveness Metrics", defaultEvaluationData.effectiveness)}
              {renderTable("Efficiency Metrics", defaultEvaluationData.efficiency)}
              {renderTable("Compression Metrics", defaultEvaluationData.compression)}
              {renderTable("Complexity Metrics", defaultEvaluationData.complexity)}
            </Card>
          </div>

          <Button
              type="primary"
              icon={<DownloadOutlined />}
              style={{ marginTop: 30, backgroundColor: "black", borderColor: "black" }} // Set button background and border to black
              onClick={handleDownload}
          >
            📦 Download Compressed Model & Results
          </Button>
        </Content>

        <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px", marginTop: "40px" }}>
          © 2025 KD-Pruning Simulator. All rights reserved.
        </Footer>
      </Layout>
    </>
  );
};

export default Visualization;
