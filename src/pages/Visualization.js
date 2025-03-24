import React, { useEffect, useRef, useState, useContext } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Button, Layout, Typography, Spin, message } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Content } = Layout;
const { Text, Title, Paragraph } = Typography;

const Visualization = () => {
  const mountRef = useRef(null);
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    if (!uploadedFile) {
        message.error("🚨 No dataset uploaded. Redirecting to upload page.");
        navigate("/upload");
        return;
    }

    const fetchVisualizationData = async () => {
        setLoading(true);
        try {
            const response = await fetch("http://localhost:5000/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file_path: uploadedFile }),
            });
            const data = await response.json();

            if (response.ok && data.success) {
                renderVisualization(data.data); // Render visualization with backend data
                setMetrics(data.metrics); // Set metrics data
            } else {
                message.error(data.error || "❌ Failed to generate visualization.");
            }
        } catch (error) {
            message.error("🚨 Error connecting to server.");
        } finally {
            setLoading(false);
        }
    };

    const renderVisualization = (visualizationData) => {
        if (!mountRef.current) return;

        // 🔹 THREE.js Scene Setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 8);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.enableRotate = true; // Allow rotation
        controls.enableZoom = true;  // Allow zoom

        // 🔹 Add Nodes
        const sphereGeometry = new THREE.SphereGeometry(0.2, 32, 32);
        const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
        visualizationData.nodes.forEach((node) => {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(node.x, node.y, node.z);
            scene.add(sphere);
        });

        // 🔹 Add Connections
        const connections = new THREE.Group();
        visualizationData.connections.forEach((connection) => {
            const material = new THREE.LineBasicMaterial({ color: 0xaaaaaa });
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(
                    visualizationData.nodes[connection.source].x,
                    visualizationData.nodes[connection.source].y,
                    visualizationData.nodes[connection.source].z
                ),
                new THREE.Vector3(
                    visualizationData.nodes[connection.target].x,
                    visualizationData.nodes[connection.target].y,
                    visualizationData.nodes[connection.target].z
                ),
            ]);
            const line = new THREE.Line(geometry, material);
            connections.add(line);
        });
        scene.add(connections);

        // 🔹 Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();
    };

    fetchVisualizationData();
}, [uploadedFile]);

const renderMetrics = (metrics) => {
    if (!metrics) return null; // Ensure metrics exist
    return (
        <div style={{ marginTop: "20px", textAlign: "left" }}>
            <Title level={4}>Performance Metrics:</Title>
            <ul>
                {Object.entries(metrics).map(([key, value]) => (
                    <li key={key}>
                        <b>{key.replace(/_/g, " ")}:</b> {value}
                    </li>
                ))}
            </ul>
        </div>
    );
};

  return (
    <Layout style={{ background: "#f0f2f5", minHeight: "100vh", display: "flex" }}>
      {/* Sidebar with Back Button */}
      <div style={{ width: "80px", padding: "20px", display: "flex", alignItems: "center" }}>
        <Button type="default" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)}>
          Back
        </Button>
      </div>

      {/* Main Content */}
      <Content style={{ flex: 1, textAlign: "center", padding: "20px" }}>
        <Title level={3}>📊 Knowledge Distillation & Pruning Visualization</Title>
        {uploadedFile ? (
          <>
            {loading && <Spin size="large" style={{ marginBottom: 20 }} />}
            <div ref={mountRef} style={{ width: "100%", height: "80vh", background: "#fff", borderRadius: 10, overflow: "hidden" }} />
            <div style={{ marginTop: "20px", textAlign: "left" }}>
                <Title level={4}>Legend:</Title>
                <ul>
                    <li><b>Green Spheres:</b> Represent nodes in the neural network.</li>
                    <li><b>Gray Lines:</b> Represent connections (edges) between nodes.</li>
                </ul>
                <Paragraph>
                    This visualization shows the structure of the neural network after applying knowledge distillation and pruning.
                    The nodes represent neurons, and the connections represent relationships between them.
                </Paragraph>
            </div>
            {metrics && renderMetrics(metrics)}
          </>
        ) : (
          <Text type="secondary" style={{ fontSize: "16px" }}>🚀 Upload a dataset to generate the simulation.</Text>
        )}
      </Content>
    </Layout>
  );
};

export default Visualization;
