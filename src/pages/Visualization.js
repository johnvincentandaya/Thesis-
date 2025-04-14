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

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 15); // Changed initial camera position

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth * 0.8, window.innerHeight * 0.8);
        renderer.setPixelRatio(window.devicePixelRatio);
        mountRef.current.appendChild(renderer.domElement);

        // Enhanced lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        scene.add(directionalLight);

        // Add nodes with hover effect for all colors
        visualizationData.nodes.forEach((node) => {
            const geometry = new THREE.SphereGeometry(node.size, 32, 32);
            const material = new THREE.MeshPhongMaterial({ 
                color: node.color,
                shininess: 100,
                specular: 0x444444,
                transparent: true,
                opacity: 0.8,
                emissive: node.color,  // Add emissive for better visibility
                emissiveIntensity: 0.2
            });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(node.x, node.y, node.z);
            sphere.userData = {
                id: node.id,
                type: node.color === "#00ff00" ? "input" : 
                      node.color === "#0000ff" ? "output" : "hidden",
                originalColor: node.color,
                originalScale: 1
            };
            scene.add(sphere);
        });

        // Enhanced controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = true;
        controls.minDistance = 2;
        controls.maxDistance = 50;
        controls.enableRotate = true;
        controls.rotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.zoomSpeed = 1.0;
        controls.panSpeed = 0.8;

        // Raycaster for node interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let hoveredNode = null;

        // Mouse move handler
        const onMouseMove = (event) => {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(scene.children);

            // Reset previous hovered node
            if (hoveredNode) {
                hoveredNode.material.opacity = 0.8;
                hoveredNode.material.emissiveIntensity = 0.2;
                hoveredNode.scale.set(1, 1, 1);
            }

            // Handle new hover
            if (intersects.length > 0) {
                const object = intersects[0].object;
                if (object.userData.id) {
                    hoveredNode = object;
                    object.material.opacity = 1;
                    object.material.emissiveIntensity = 0.5;
                    object.scale.set(1.2, 1.2, 1.2);

                    // Optional: Show node info
                    const nodeType = object.userData.type;
                    const nodeId = object.userData.id;
                    console.log(`Hovering over ${nodeType} node: ${nodeId}`);
                }
            } else {
                hoveredNode = null;
            }
        };

        renderer.domElement.addEventListener('mousemove', onMouseMove);

        // Animation
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Enhanced resize handler
        const handleResize = () => {
            const width = window.innerWidth * 0.8;
            const height = window.innerHeight * 0.8;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            renderer.domElement.removeEventListener('mousemove', onMouseMove);
            mountRef.current?.removeChild(renderer.domElement);
        };
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
                    <li><b>Green Spheres:</b> Input layer nodes (where data enters the network)</li>
                    <li><b>Yellow Spheres:</b> Hidden layer nodes (where intermediate processing occurs)</li>
                    <li><b>Blue Spheres:</b> Output layer nodes (where final predictions are made)</li>
                    <li><b>Gray Lines:</b> Connections (weights) between nodes</li>
                </ul>
                <Paragraph>
                    This visualization shows the structure of the neural network after applying knowledge distillation and pruning.
                    The yellow nodes in the middle layers represent the hidden neurons where the model performs its intermediate computations,
                    transforming input features into more complex representations before making final predictions.
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
