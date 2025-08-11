import React, { useState, useRef, useEffect } from "react";
import { Layout, Card, Button, Typography, Row, Col, Progress, Alert, Space, Divider } from "antd";
import { Navbar, Nav, Container } from "react-bootstrap";
import { Link, useLocation, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Cylinder, Html } from '@react-three/drei';
import * as THREE from 'three';
import './Visualization.css';

const { Title, Paragraph, Text: AntText } = Typography;
const { Content } = Layout;

// 3D Neural Network Components
function NeuralNode({ position, color = "#4fc3f7", size = 0.3, isActive = false, isPruned = false, opacity = 1 }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (isActive && meshRef.current) {
      meshRef.current.scale.x = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      meshRef.current.scale.y = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      meshRef.current.scale.z = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
    }
  });

  return (
    <Sphere ref={meshRef} args={[size, 16, 16]} position={position}>
      <meshStandardMaterial 
        color={isPruned ? "#666" : color} 
        opacity={isPruned ? 0.3 : opacity}
        transparent
        emissive={isActive ? color : "#000"}
        emissiveIntensity={isActive ? 0.3 : 0}
      />
    </Sphere>
  );
}

function Connection({ start, end, isActive = false, isPruned = false, strength = 1 }) {
  const lineRef = useRef();
  
  useFrame((state) => {
    if (isActive && lineRef.current) {
      lineRef.current.material.opacity = 0.5 + Math.sin(state.clock.elapsedTime * 2) * 0.3;
    }
  });

  const points = [start, end];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <line ref={lineRef}>
      <bufferGeometry attach="geometry" {...geometry} />
      <lineBasicMaterial 
        attach="material" 
        color={isPruned ? "#666" : "#888"} 
        opacity={isPruned ? 0.2 : strength}
        transparent
        linewidth={isPruned ? 1 : 2}
      />
    </line>
  );
}

function DataFlow({ step, isActive }) {
  const particlesRef = useRef();
  const [particles] = useState(() => {
    const temp = [];
    for (let i = 0; i < 50; i++) {
      temp.push({
        position: new THREE.Vector3(
          Math.random() * 10 - 5,
          Math.random() * 10 - 5,
          Math.random() * 10 - 5
        ),
        velocity: new THREE.Vector3(
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05,
          Math.random() * 0.1 - 0.05
        ),
        color: new THREE.Color().setHSL(Math.random(), 0.7, 0.5)
      });
    }
    return temp;
  });

  useFrame((state) => {
    if (!isActive || !particlesRef.current) return;
    
    particles.forEach((particle, i) => {
      particle.position.add(particle.velocity);
      
      // Bounce off boundaries
      if (Math.abs(particle.position.x) > 5) particle.velocity.x *= -1;
      if (Math.abs(particle.position.y) > 5) particle.velocity.y *= -1;
      if (Math.abs(particle.position.z) > 5) particle.velocity.z *= -1;
      
      // Update particle position in geometry
      const positions = particlesRef.current.geometry.attributes.position.array;
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;
    });
    
    particlesRef.current.geometry.attributes.position.needsUpdate = true;
  });

  if (!isActive) return null;

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.length}
          array={new Float32Array(particles.length * 3)}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} vertexColors />
    </points>
  );
}

function NeuralNetwork({ step, selectedModel }) {
  const { camera, gl } = useThree();
  const networkRef = useRef();
  
  // Handle responsive resizing for the container
  useEffect(() => {
    const handleResize = () => {
      // Get the actual container dimensions
      const container = document.querySelector('.visualization-container');
      if (container && gl) {
        const rect = container.getBoundingClientRect();
        gl.setSize(rect.width, rect.height);
        gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      }
    };
    
    // Initial resize
    handleResize();
    
    // Use ResizeObserver for more accurate container size detection
    const container = document.querySelector('.visualization-container');
    let resizeObserver;
    
    if (container && window.ResizeObserver) {
      resizeObserver = new ResizeObserver(() => {
        handleResize();
      });
      resizeObserver.observe(container);
    }
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [gl]);
  
  // Define network architecture based on model
  const getNetworkConfig = () => {
    switch(selectedModel) {
      case "distillBert":
        return {
          layers: [12, 8, 6, 4], // Transformer layers
          colors: ["#4fc3f7", "#29b6f6", "#0288d1", "#01579b"],
          spacing: 2.5
        };
      case "T5-small":
        return {
          layers: [10, 8, 6, 4],
          colors: ["#ff7043", "#ff5722", "#e64a19", "#bf360c"],
          spacing: 2.2
        };
      case "MobileNetV2":
        return {
          layers: [8, 6, 4, 3],
          colors: ["#66bb6a", "#4caf50", "#388e3c", "#2e7d32"],
          spacing: 2.0
        };
      case "ResNet-18":
        return {
          layers: [6, 5, 4, 3],
          colors: ["#ab47bc", "#8e24aa", "#7b1fa2", "#6a1b9a"],
          spacing: 2.3
        };
      default:
        return {
          layers: [8, 6, 4, 3],
          colors: ["#4fc3f7", "#29b6f6", "#0288d1", "#01579b"],
          spacing: 2.0
        };
    }
  };

  const config = getNetworkConfig();
  const nodes = [];
  const connections = [];
  let nodeId = 0;

  // Generate nodes for each layer
  config.layers.forEach((layerSize, layerIndex) => {
    const x = layerIndex * config.spacing;
    const isPruned = step >= 4; // Pruning starts at step 4
    const isActive = step >= layerIndex + 1;
    
    for (let i = 0; i < layerSize; i++) {
      const y = (layerSize - 1) / 2 - i;
      const z = Math.sin(i * 0.5) * 0.5;
      
      nodes.push({
        id: nodeId++,
        position: [x, y, z],
        color: config.colors[layerIndex],
        isActive,
        isPruned: isPruned && i >= layerSize * 0.7, // Prune 30% of nodes
        size: 0.3
      });
    }
  });

  // Generate connections
  for (let layerIndex = 0; layerIndex < config.layers.length - 1; layerIndex++) {
    const currentLayerStart = config.layers.slice(0, layerIndex).reduce((sum, size) => sum + size, 0);
    const nextLayerStart = config.layers.slice(0, layerIndex + 1).reduce((sum, size) => sum + size, 0);
    
    for (let i = 0; i < config.layers[layerIndex]; i++) {
      for (let j = 0; j < config.layers[layerIndex + 1]; j++) {
        const startNode = nodes[currentLayerStart + i];
        const endNode = nodes[nextLayerStart + j];
        
        if (startNode && endNode) {
          connections.push({
            start: new THREE.Vector3(...startNode.position),
            end: new THREE.Vector3(...endNode.position),
            isActive: step >= layerIndex + 2,
            isPruned: step >= 4 && (startNode.isPruned || endNode.isPruned),
            strength: Math.random() * 0.5 + 0.5
          });
        }
      }
    }
  }

  // Camera fit and animation
  useEffect(() => {
    if (networkRef.current) {
      // Fit camera to model with container-aware positioning
      const box = new THREE.Box3().setFromObject(networkRef.current);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      
      // Calculate optimal camera distance based on container size
      const container = document.querySelector('.visualization-container');
      let cameraZ = 8; // Default distance
      
      if (container) {
        const rect = container.getBoundingClientRect();
        const aspectRatio = rect.width / rect.height;
        cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * (aspectRatio > 1 ? 1.2 : 1.5);
      }
      
      camera.position.set(center.x + cameraZ * 0.5, center.y + cameraZ * 0.3, center.z + cameraZ * 0.5);
      camera.lookAt(center);
      camera.updateMatrixWorld();
    }
  }, [step, selectedModel, camera]);

  // Smooth camera animation
  useFrame((state) => {
    if (networkRef.current) {
      const time = state.clock.elapsedTime;
      const targetX = Math.sin(time * 0.2) * 8;
      const targetY = Math.cos(time * 0.3) * 3 + 2;
      const targetZ = Math.cos(time * 0.2) * 8;
      
      camera.position.x += (targetX - camera.position.x) * 0.02;
      camera.position.y += (targetY - camera.position.y) * 0.02;
      camera.position.z += (targetZ - camera.position.z) * 0.02;
      camera.lookAt(0, 0, 0);
    }
  });

  return (
    <group ref={networkRef}>
      {/* Connections */}
      {connections.map((conn, index) => (
        <Connection key={`conn-${index}`} {...conn} />
      ))}
      
      {/* Nodes */}
      {nodes.map((node) => (
        <NeuralNode key={node.id} {...node} />
      ))}
      
      {/* Data flow particles */}
      <DataFlow step={step} isActive={step >= 1 && step <= 3} />
    </group>
  );
}

// Step information with detailed explanations
const getStepInfo = (step, selectedModel) => {
  const steps = [
    {
      title: "Model Architecture Initialization",
      subtitle: `Loading ${selectedModel} Neural Network`,
      description: `Initializing the ${selectedModel} architecture with pre-trained weights. The model is structured with multiple layers that process information hierarchically.`,
      technicalDetails: [
        "Loading pre-trained weights from HuggingFace",
        "Initializing transformer/CNN architecture",
        "Setting up attention mechanisms",
        "Configuring layer normalization"
      ],
      visualHint: "Watch the network structure materialize layer by layer"
    },
    {
      title: "Input Data Processing",
      subtitle: "Tokenization & Embedding",
      description: `Preparing input data for ${selectedModel}. For language models, this involves tokenization and embedding. For vision models, this includes image preprocessing and normalization.`,
      technicalDetails: [
        "Tokenizing input text into subword units",
        "Converting tokens to numerical embeddings",
        "Applying positional encoding",
        "Normalizing input data"
      ],
      visualHint: "Observe data particles flowing through the network"
    },
    {
      title: "Forward Propagation",
      subtitle: "Information Flow Through Layers",
      description: `Data flows through the ${selectedModel} network. Each layer processes the input and passes it to the next, building increasingly complex representations.`,
      technicalDetails: [
        "Computing attention scores between tokens",
        "Applying multi-head attention mechanisms",
        "Processing through feed-forward networks",
        "Updating hidden representations"
      ],
      visualHint: "See how information propagates through each layer"
    },
    {
      title: "Knowledge Distillation",
      subtitle: "Teacher-Student Learning",
      description: `The student model learns from the teacher's soft predictions. This transfers not just the correct answers, but also the teacher's confidence levels and decision-making patterns.`,
      technicalDetails: [
        "Computing teacher model predictions",
        "Applying temperature scaling (T=2.0)",
        "Calculating distillation loss",
        "Updating student model weights"
      ],
      visualHint: "Notice how the student model adapts to teacher patterns"
    },
    {
      title: "Model Pruning",
      subtitle: "Removing Redundant Weights",
      description: `Removing redundant weights and connections from ${selectedModel}. This reduces model size while preserving the most important learned features.`,
      technicalDetails: [
        "Identifying low-importance weights",
        "Applying L1 unstructured pruning",
        "Removing 30% of connections",
        "Maintaining model sparsity"
      ],
      visualHint: "Watch as connections fade and nodes become inactive"
    },
    {
      title: "Fine-tuning",
      subtitle: "Recovery & Optimization",
      description: `Adjusting the pruned ${selectedModel} to recover any lost accuracy. The model adapts to the new, sparser architecture.`,
      technicalDetails: [
        "Re-training on compressed architecture",
        "Optimizing remaining connections",
        "Recovering lost performance",
        "Finalizing model weights"
      ],
      visualHint: "See the network stabilize with optimized connections"
    },
    {
      title: "Performance Evaluation",
      subtitle: "Compression Results Analysis",
      description: `Comprehensive evaluation of the compressed ${selectedModel}. Comparing accuracy, speed, and size against the original model to measure compression effectiveness.`,
      technicalDetails: [
        "Measuring inference latency",
        "Calculating model size reduction",
        "Evaluating accuracy preservation",
        "Computing efficiency metrics"
      ],
      visualHint: "Review the final compressed network structure"
    }
  ];
  
  return steps[step] || steps[0];
};

const Visualization = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { trainingComplete, selectedModel, metrics } = location.state || {};
  const [started, setStarted] = useState(false);
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  // Handle window resize for responsive design
  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Auto-play functionality
  useEffect(() => {
    if (autoPlay && started) {
      const timer = setTimeout(() => {
        if (step < 6) {
          setStep(step + 1);
        } else {
          setAutoPlay(false);
        }
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [autoPlay, step, started]);

  // If not trained, redirect or show warning
  if (!trainingComplete) {
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
        <div style={{ padding: 40, textAlign: 'center' }}>
          <Alert
            message="Training Required"
            description={<>
              You must complete training before running the simulation.<br />
              <Button type="primary" style={{ marginTop: 16 }} onClick={() => navigate('/training')}>
                Go to Training
              </Button>
            </>}
            type="warning"
            showIcon
          />
        </div>
      </>
    );
  }

  const stepInfo = getStepInfo(step, selectedModel);

  const startSimulation = () => {
    setStarted(true);
    setStep(0);
  };

  const nextStep = () => {
    if (step < 6) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > 0) setStep(step - 1);
  };

  const resetSimulation = () => {
    setStep(0);
    setAutoPlay(false);
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
      
      <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
        <Content style={{ padding: "20px" }}>
          <div style={{ maxWidth: 1400, margin: '0 auto' }}>
            <Row gutter={[24, 24]}>
              {/* 3D Visualization Panel */}
              <Col xs={24} lg={16}>
                <Card 
                  className="visualization-container"
                  style={{ 
                    height: windowSize.width < 992 ? '50vh' : '70vh', 
                    background: '#1a1a1a',
                    border: 'none',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    padding: 0,
                    position: 'relative'
                  }}
                >
                  {!started ? (
                    <div style={{ 
                      height: '100%', 
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'center', 
                      alignItems: 'center',
                      color: 'white',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '48px', marginBottom: '20px' }}>🧠</div>
                      <Title level={2} style={{ color: 'white', marginBottom: '16px' }}>
                        3D Neural Network Simulation
                      </Title>
                      <Paragraph style={{ color: '#ccc', fontSize: '16px', marginBottom: '32px' }}>
                        Experience the complete training process of {selectedModel} in an interactive 3D environment
                      </Paragraph>
                      <Button 
                        type="primary" 
                        size="large" 
                        onClick={startSimulation}
                        style={{ 
                          height: '48px', 
                          fontSize: '16px',
                          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          border: 'none'
                        }}
                      >
                        🚀 Start 3D Simulation
                      </Button>
                    </div>
                  ) : (
                    <div style={{ 
                      width: '100%', 
                      height: '100%', 
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0
                    }}>
                      <Canvas
                        camera={{ position: [8, 4, 8], fov: 60 }}
                        style={{ 
                          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
                          width: '100%',
                          height: '100%',
                          display: 'block'
                        }}
                        gl={{ 
                          antialias: true, 
                          alpha: false,
                          powerPreference: "high-performance"
                        }}
                        dpr={Math.min(window.devicePixelRatio, 2)}
                      >
                        <ambientLight intensity={0.4} />
                        <pointLight position={[10, 10, 10]} intensity={1} />
                        <pointLight position={[-10, -10, -10]} intensity={0.5} />
                        
                        <NeuralNetwork step={step} selectedModel={selectedModel} />
                        
                        <OrbitControls 
                          enablePan={true} 
                          enableZoom={true} 
                          enableRotate={true}
                          maxDistance={25}
                          minDistance={2}
                          dampingFactor={0.05}
                          enableDamping={true}
                          zoomSpeed={0.8}
                          panSpeed={0.8}
                          rotateSpeed={0.8}
                        />
                        
                        {/* Step indicator */}
                        <Html position={[0, 5, 0]} center>
                          <div style={{
                            background: 'rgba(0,0,0,0.8)',
                            color: 'white',
                            padding: '8px 16px',
                            borderRadius: '20px',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            whiteSpace: 'nowrap'
                          }}>
                            Step {step + 1}/7: {stepInfo.title}
                          </div>
                        </Html>
                      </Canvas>
                    </div>
                  )}
                </Card>
              </Col>

              {/* Control Panel */}
              <Col xs={24} lg={8}>
                <div style={{ height: windowSize.width < 992 ? '50vh' : '70vh', overflowY: 'auto' }}>
                  {started ? (
                    <>
                      {/* Step Information */}
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Title level={3} style={{ marginBottom: 8, color: '#1890ff' }}>
                          {stepInfo.title}
                        </Title>
                        <Paragraph style={{ color: '#666', marginBottom: 16 }}>
                          {stepInfo.subtitle}
                        </Paragraph>
                        <Paragraph style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          {stepInfo.description}
                        </Paragraph>
                        
                        <Divider style={{ margin: '16px 0' }} />
                        
                        <div style={{ marginBottom: 16 }}>
                          <AntText strong style={{ color: '#52c41a' }}>💡 Visual Hint:</AntText>
                          <Paragraph style={{ fontSize: '13px', color: '#666', marginTop: 4 }}>
                            {stepInfo.visualHint}
                          </Paragraph>
                        </div>

                        {showTechnicalDetails && (
                          <div style={{ marginTop: 16 }}>
                            <AntText strong style={{ color: '#722ed1' }}>🔧 Technical Details:</AntText>
                            <ul style={{ fontSize: '13px', color: '#666', marginTop: 8 }}>
                              {stepInfo.technicalDetails.map((detail, index) => (
                                <li key={index}>{detail}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </Card>

                      {/* Controls */}
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <Button 
                              onClick={prevStep} 
                              disabled={step === 0}
                              style={{ flex: 1 }}
                            >
                              ⬅️ Previous
                            </Button>
                            <Button 
                              onClick={nextStep} 
                              disabled={step === 6}
                              type="primary"
                              style={{ flex: 1 }}
                            >
                              Next ➡️
                            </Button>
                          </div>
                          
                          <Button 
                            onClick={() => setAutoPlay(!autoPlay)}
                            type={autoPlay ? "default" : "primary"}
                            style={{ width: '100%' }}
                          >
                            {autoPlay ? '⏸️ Pause Auto-play' : '▶️ Start Auto-play'}
                          </Button>
                          
                          <Button 
                            onClick={resetSimulation}
                            style={{ width: '100%' }}
                          >
                            🔄 Reset Simulation
                          </Button>
                          
                          <Button 
                            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                            type="dashed"
                            style={{ width: '100%' }}
                          >
                            {showTechnicalDetails ? '🔽 Hide' : '🔼 Show'} Technical Details
                          </Button>
                        </Space>
                      </Card>

                      {/* Progress */}
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Title level={5} style={{ marginBottom: 12 }}>Simulation Progress</Title>
                        <Progress 
                          percent={((step + 1) / 7) * 100} 
                          status="active"
                          strokeColor={{
                            '0%': '#108ee9',
                            '100%': '#87d068',
                          }}
                        />
                        <div style={{ textAlign: 'center', marginTop: 8, fontSize: '12px', color: '#666' }}>
                          Step {step + 1} of 7
                        </div>
                      </Card>

                      {/* Final Results */}
                      {step === 6 && metrics && (
                        <Card style={{ borderRadius: '12px', background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)' }}>
                          <Title level={4} style={{ color: '#52c41a', marginBottom: 16 }}>
                            🎯 Compression Results
                          </Title>
                          
                          {metrics.model_performance && (
                            <div style={{ marginBottom: 16 }}>
                              <Row gutter={8}>
                                <Col span={12}>
                                  <div style={{ textAlign: 'center', padding: '12px', background: '#f6ffed', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#52c41a' }}>
                                      {metrics.model_performance.metrics?.accuracy || 'N/A'}
                                    </div>
                                    <div style={{ fontSize: '12px', color: '#666' }}>Accuracy</div>
                                  </div>
                                </Col>
                                <Col span={12}>
                                  <div style={{ textAlign: 'center', padding: '12px', background: '#fff7e6', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#fa8c16' }}>
                                      {metrics.model_performance.metrics?.size_mb || 'N/A'}
                                    </div>
                                    <div style={{ fontSize: '12px', color: '#666' }}>Size (MB)</div>
                                  </div>
                                </Col>
                              </Row>
                            </div>
                          )}
                          
                          <Alert
                            message="Simulation Complete!"
                            description="You've successfully witnessed the complete Knowledge Distillation and Pruning process in 3D."
                            type="success"
                            showIcon
                          />
                        </Card>
                      )}
                    </>
                  ) : (
                    <Card style={{ borderRadius: '12px' }}>
                      <Title level={4}>Interactive 3D Simulation</Title>
                      <Paragraph>
                        This advanced 3D visualization will show you:
                      </Paragraph>
                      <ul style={{ fontSize: '14px', color: '#666' }}>
                        <li>🎯 Real-time neural network architecture</li>
                        <li>⚡ Dynamic data flow visualization</li>
                        <li>✂️ Interactive pruning process</li>
                        <li>📊 Performance metrics display</li>
                        <li>🎓 Educational step-by-step guidance</li>
                      </ul>
                      <Divider />
                      <Paragraph style={{ fontSize: '12px', color: '#999' }}>
                        <strong>Controls:</strong><br />
                        • Mouse: Rotate, zoom, and pan the 3D view<br />
                        • Auto-play: Watch the complete process automatically<br />
                        • Manual: Step through each phase manually
                      </Paragraph>
                    </Card>
                  )}
                </div>
              </Col>
            </Row>
          </div>
        </Content>
      </Layout>
    </>
  );
};

export default Visualization;