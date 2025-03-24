import React, { useEffect, useRef, useState, useContext } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Button, Layout, Typography, Spin, message } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Content } = Layout;
const { Text, Title } = Typography;

const Visualization = () => {
  const mountRef = useRef(null);
  const navigate = useNavigate();
  const { uploadedFile } = useContext(UploadContext);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!uploadedFile) {
        message.error("🚨 No dataset uploaded. Redirecting to upload page.");
        navigate("/upload");
        return;
    }

    if (!mountRef.current) return;

    setLoading(true);

    // 🔹 THREE.js Scene Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 8);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    // 🔹 Neural Network Simulation
    const sphereGeometry = new THREE.SphereGeometry(0.2, 32, 32);
    const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });

    let neuralNodes = [];
    for (let i = 0; i < 15; i++) {
      const node = new THREE.Mesh(sphereGeometry, sphereMaterial);
      node.position.set(Math.random() * 4 - 2, Math.random() * 4 - 2, Math.random() * 4 - 2);
      scene.add(node);
      neuralNodes.push(node);
    }

    const connections = new THREE.Group();
    neuralNodes.forEach((nodeA, indexA) => {
      neuralNodes.forEach((nodeB, indexB) => {
        if (indexA < indexB) {
          const material = new THREE.LineBasicMaterial({ color: 0xaaaaaa });
          const geometry = new THREE.BufferGeometry().setFromPoints([nodeA.position, nodeB.position]);
          const line = new THREE.Line(geometry, material);
          connections.add(line);
        }
      });
    });
    scene.add(connections);

    setLoading(false);

    // 🔹 Handle Resizing
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    // 🔹 Animation Loop
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      scene.clear();

      if (mountRef.current) {
        mountRef.current.innerHTML = "";
      }
    };
  }, [uploadedFile]);

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
          </>
        ) : (
          <Text type="secondary" style={{ fontSize: "16px" }}>🚀 Upload a dataset to generate the simulation.</Text>
        )}
      </Content>
    </Layout>
  );
};

export default Visualization;
