import React, { useState, useRef, useEffect } from "react";
import { Layout, Card, Button, Typography, Row, Col, Progress, Alert, Space, Divider, Select, Switch } from "antd";
import { Navbar, Nav, Container } from "react-bootstrap";
import { Link, useLocation, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Cylinder, Html } from '@react-three/drei';
import * as THREE from 'three';
import './Visualization.css';
import { socket, SOCKET_URL } from "../socket";
import Footer from "../components/Footer";

const { Title, Paragraph, Text: AntText } = Typography;
const { Content } = Layout;

// Deterministic seeded randomness utilities for consistent visualization
function stringHash(str) {
  let h = 2166136261 >>> 0; // FNV-1a 32-bit
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seededFloat(baseSeed, ...parts) {
  const key = [baseSeed, ...parts].join('|');
  const h = stringHash(key);
  return (h % 1000000000) / 1000000000; // [0,1)
}

// 3D Neural Network Components
function NeuralNode({ position, color = "#4fc3f7", size = 0.3, isActive = false, isPruned = false, opacity = 1, label = "", layerIndex = 0, nodeIndex = 0, pruningReason = "", totalLayers = 4, onNodeClick }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (isActive && meshRef.current && !isPruned) {
      meshRef.current.scale.x = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      meshRef.current.scale.y = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      meshRef.current.scale.z = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
    }
    
    // Add pulsing effect for nodes being pruned
    if (isPruned && meshRef.current) {
      meshRef.current.scale.x = 0.8 + Math.sin(state.clock.elapsedTime * 5) * 0.2;
      meshRef.current.scale.y = 0.8 + Math.sin(state.clock.elapsedTime * 5) * 0.2;
      meshRef.current.scale.z = 0.8 + Math.sin(state.clock.elapsedTime * 5) * 0.2;
    }
  });

  // All layers equally visible (no focus layer)
  const effectiveOpacity = opacity;

  const handleClick = (event) => {
    event.stopPropagation();
    if (onNodeClick) {
      onNodeClick({
        label,
        layerIndex,
        nodeIndex,
        isPruned,
        pruningReason,
        color,
        position
      });
    }
  };

  return (
    <group position={position}>
      <Sphere ref={meshRef} args={[size, 16, 16]} onClick={handleClick} style={{ cursor: 'pointer' }}>
        <meshStandardMaterial 
          color={isPruned ? "#ff4444" : color} 
          opacity={isPruned ? 0.6 : effectiveOpacity}
          transparent
          emissive={isPruned ? "#ff0000" : (isActive ? color : "#000")}
          emissiveIntensity={isPruned ? 0.5 : (isActive ? 0.3 : 0)}
        />
      </Sphere>

      {/* Node Label - show for input/output/pruned nodes (no toggle) */}
      {label && (layerIndex === 0 || layerIndex === totalLayers - 1 || isPruned) && (
        <Html position={[0, size + 0.4, 0]} center>
          <div style={{
            background: isPruned ? 'rgba(255, 68, 68, 0.95)' : 'rgba(0,0,0,0.9)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: '15px',
            fontSize: '12px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            border: isPruned ? '3px solid #ff0000' : '2px solid #fff',
            boxShadow: isPruned ? '0 0 15px #ff0000' : '0 0 10px rgba(0,0,0,0.5)',
            minWidth: '60px',
            textAlign: 'center',
            pointerEvents: 'none'
          }}>
            {label}
          </div>
        </Html>
      )}

      {/* Pruning Reason Label - show when pruned (no toggle) */}
      {isPruned && pruningReason && (
        <Html position={[0, -size - 0.5, 0]} center>
          <div style={{
            background: 'rgba(255, 0, 0, 0.95)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '10px',
            fontSize: '11px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            maxWidth: '140px',
            textAlign: 'center',
            border: '3px solid #ff0000',
            boxShadow: '0 0 20px #ff0000',
            pointerEvents: 'none'
          }}>
            {pruningReason}
          </div>
        </Html>
      )}
    </group>
  );
}

function Connection({ start, end, isActive = false, isPruned = false, strength = 1, pruningReason = "", sourceLayer = 0, targetLayer = 0 }) {
  const lineRef = useRef();
  
  useFrame((state) => {
    if (isActive && lineRef.current && !isPruned) {
      lineRef.current.material.opacity = 0.5 + Math.sin(state.clock.elapsedTime * 2) * 0.3;
    }
    
    // Add pulsing effect for connections being pruned
    if (isPruned && lineRef.current) {
      lineRef.current.material.opacity = 0.1 + Math.sin(state.clock.elapsedTime * 6) * 0.1;
      lineRef.current.material.color.setHex(0xff0000);
    }
  });

  const points = [start, end];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  // All connections equally visible (no focus layer)
  const effectiveOpacity = isPruned ? 0.2 : strength;

  return (
    <group>
      <line ref={lineRef}>
        <bufferGeometry attach="geometry" {...geometry} />
        <lineBasicMaterial 
          attach="material" 
          color={isPruned ? "#ff0000" : "#888"} 
          opacity={effectiveOpacity}
          transparent
          linewidth={isPruned ? 1 : 2}
        />
      </line>

      {/* Pruning Reason for Connections - show sometimes to reduce clutter (no toggle) */}
      {isPruned && pruningReason && (Math.random() < 0.3) && (
        <Html position={[
          (start.x + end.x) / 2,
          (start.y + end.y) / 2 + 0.3,
          (start.z + end.z) / 2
        ]} center>
          <div style={{
            background: 'rgba(255, 0, 0, 0.95)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: '8px',
            fontSize: '10px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            maxWidth: '120px',
            textAlign: 'center',
            border: '2px solid #ff0000',
            boxShadow: '0 0 15px #ff0000',
            pointerEvents: 'none'
          }}>
            {pruningReason}
          </div>
        </Html>
      )}
    </group>
  );
}

function DataFlow({ step, isActive, seedKey = 'dataflow' }) {
  const particlesRef = useRef();
  const [particles] = useState(() => {
    const temp = [];
    for (let i = 0; i < 50; i++) {
      temp.push({
        position: new THREE.Vector3(
          seededFloat(seedKey, 'px', i) * 10 - 5,
          seededFloat(seedKey, 'py', i) * 10 - 5,
          seededFloat(seedKey, 'pz', i) * 10 - 5
        ),
        velocity: new THREE.Vector3(
          seededFloat(seedKey, 'vx', i) * 0.1 - 0.05,
          seededFloat(seedKey, 'vy', i) * 0.1 - 0.05,
          seededFloat(seedKey, 'vz', i) * 0.1 - 0.05
        ),
        color: new THREE.Color().setHSL(seededFloat(seedKey, 'c', i), 0.7, 0.5)
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

function NeuralNetwork({ step, selectedModel, onNodeClick }) {
  const { camera, gl, controls } = useThree();
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
          layers: [8, 6, 4, 3], // Reduced from 12,8,6,4 to prevent overlapping
          colors: ["#4fc3f7", "#29b6f6", "#0288d1", "#01579b"],
          spacing: 3.5, // Increased spacing
          layerNames: ["Input", "Hidden 1", "Hidden 2", "Output"]
        };
      case "T5-small":
        return {
          layers: [7, 6, 4, 3], // Reduced from 10,8,6,4
          colors: ["#ff7043", "#ff5722", "#e64a19", "#bf360c"],
          spacing: 3.2, // Increased spacing
          layerNames: ["Encoder", "Decoder", "Attention", "Output"]
        };
      case "MobileNetV2":
        return {
          layers: [6, 5, 4, 3], // Reduced from 8,6,4,3
          colors: ["#66bb6a", "#4caf50", "#388e3c", "#2e7d32"],
          spacing: 3.0, // Increased spacing
          layerNames: ["Conv", "Depthwise", "Pointwise", "Output"]
        };
      case "ResNet-18":
        return {
          layers: [5, 4, 3, 2], // Reduced from 6,5,4,3
          colors: ["#ab47bc", "#8e24aa", "#7b1fa2", "#6a1b9a"],
          spacing: 3.3, // Increased spacing
          layerNames: ["Conv1", "ResBlock", "ResBlock", "Output"]
        };
      default:
        return {
          layers: [6, 5, 4, 3], // Reduced from 8,6,4,3
          colors: ["#4fc3f7", "#29b6f6", "#0288d1", "#01579b"],
          spacing: 3.0, // Increased spacing
          layerNames: ["Input", "Hidden", "Hidden", "Output"]
        };
    }
  };

  const config = getNetworkConfig();
  const nodes = [];
  const connections = [];
  let nodeId = 0;
  
  // Dynamic pruning calculation based on computational analysis
  const calculateNodeImportance = (layerIndex, nodeIndex, layerSize, modelType) => {
    // Simulate different computational metrics for each model
    let activationScore = 0;
    let weightMagnitude = 0;
    let gradientStrength = 0;
    let redundancyScore = 0;
    
    // Model-specific computational characteristics
    switch(modelType) {
      case "distillBert":
        // BERT-like models: attention heads and feed-forward layers
        activationScore = Math.random() * 0.8 + 0.2; // 0.2-1.0
        weightMagnitude = Math.random() * 0.7 + 0.3; // 0.3-1.0
        gradientStrength = Math.random() * 0.6 + 0.4; // 0.4-1.0
        redundancyScore = Math.random() * 0.9; // 0.0-0.9
        break;
      case "T5-small":
        // T5: encoder-decoder with attention
        activationScore = Math.random() * 0.7 + 0.3; // 0.3-1.0
        weightMagnitude = Math.random() * 0.8 + 0.2; // 0.2-1.0
        gradientStrength = Math.random() * 0.5 + 0.5; // 0.5-1.0
        redundancyScore = Math.random() * 0.8; // 0.0-0.8
        break;
      case "MobileNetV2":
        // MobileNet: depthwise separable convolutions
        activationScore = Math.random() * 0.9 + 0.1; // 0.1-1.0
        weightMagnitude = Math.random() * 0.6 + 0.4; // 0.4-1.0
        gradientStrength = Math.random() * 0.7 + 0.3; // 0.3-1.0
        redundancyScore = Math.random() * 0.7; // 0.0-0.7
        break;
      case "ResNet-18":
        // ResNet: residual connections
        activationScore = Math.random() * 0.8 + 0.2; // 0.2-1.0
        weightMagnitude = Math.random() * 0.9 + 0.1; // 0.1-1.0
        gradientStrength = Math.random() * 0.8 + 0.2; // 0.2-1.0
        redundancyScore = Math.random() * 0.6; // 0.0-0.6
        break;
      default:
        activationScore = Math.random() * 0.8 + 0.2;
        weightMagnitude = Math.random() * 0.7 + 0.3;
        gradientStrength = Math.random() * 0.6 + 0.4;
        redundancyScore = Math.random() * 0.8;
    }
    
    // Layer-specific adjustments
    if (layerIndex === 0) {
      // Input layer: preserve more nodes
      activationScore *= 1.2;
      weightMagnitude *= 1.1;
    } else if (layerIndex === config.layers.length - 1) {
      // Output layer: preserve more nodes
      activationScore *= 1.3;
      weightMagnitude *= 1.2;
    } else {
      // Hidden layers: more aggressive pruning
      activationScore *= 0.9;
      weightMagnitude *= 0.8;
    }
    
    // Position-based adjustments (nodes in middle of layer are often more important)
    const positionInLayer = Math.abs(nodeIndex - (layerSize - 1) / 2) / (layerSize / 2);
    const positionBonus = 1.0 - positionInLayer * 0.3; // 0.7-1.0
    
    // Calculate final importance score
    const importanceScore = (
      activationScore * 0.3 +
      weightMagnitude * 0.3 +
      gradientStrength * 0.2 +
      (1 - redundancyScore) * 0.1 +
      positionBonus * 0.1
    );
    
    // Dynamic pruning threshold based on model and layer
    let pruningThreshold = 0.4; // Base threshold
    
    // Adjust threshold based on model type
    switch(modelType) {
      case "distillBert":
        pruningThreshold = 0.35; // More aggressive for BERT
        break;
      case "T5-small":
        pruningThreshold = 0.38; // Moderate for T5
        break;
      case "MobileNetV2":
        pruningThreshold = 0.45; // Less aggressive for MobileNet
        break;
      case "ResNet-18":
        pruningThreshold = 0.42; // Moderate for ResNet
        break;
    }
    
    // Layer-specific threshold adjustment
    if (layerIndex === 0) pruningThreshold += 0.1; // Preserve input layer
    if (layerIndex === config.layers.length - 1) pruningThreshold += 0.15; // Preserve output layer
    
    const shouldPrune = importanceScore < pruningThreshold;
    
    // Generate meaningful pruning reason
    let reason = "";
    if (shouldPrune) {
      if (activationScore < 0.4) reason = "Low activation";
      else if (weightMagnitude < 0.4) reason = "Weak weights";
      else if (gradientStrength < 0.4) reason = "Poor gradients";
      else if (redundancyScore > 0.7) reason = "Redundant features";
      else if (positionInLayer > 0.8) reason = "Edge position";
      else reason = "Low importance";
    }
    
    return { shouldPrune, reason, importanceScore };
  };

  // Generate nodes for each layer with enhanced labeling
  config.layers.forEach((layerSize, layerIndex) => {
    const x = layerIndex * config.spacing;
    const isPruned = step >= 4; // Pruning starts at step 4
    const isActive = step >= layerIndex + 1;
    
    for (let i = 0; i < layerSize; i++) {
      const y = (layerSize - 1) / 2 - i;
      // Stable slight depth jitter based on seed (removes flicker across renders)
      const z = (seededFloat(selectedModel || 'default-model-seed', 'z', layerIndex, i) - 0.5) * 1.0;
      
      // Dynamic pruning based on computational analysis
      let shouldPrune = false;
      let pruningReason = "";
      let nodeLabel = `N${layerIndex+1}-${i+1}`;
      
      if (isPruned) {
        // Simulate computational analysis for each node
        const nodeImportance = calculateNodeImportance(layerIndex, i, layerSize, selectedModel);
        shouldPrune = nodeImportance.shouldPrune;
        pruningReason = nodeImportance.reason;
      }
      
      nodes.push({
        id: nodeId++,
        position: [x, y, z],
        color: config.colors[layerIndex],
        isActive,
        isPruned: shouldPrune,
        size: 0.3,
        label: nodeLabel,
        layerIndex,
        nodeIndex: i,
        pruningReason: shouldPrune ? pruningReason : ""
      });
    }
  });

  // Generate connections with pruning logic
  for (let layerIndex = 0; layerIndex < config.layers.length - 1; layerIndex++) {
    const currentLayerStart = config.layers.slice(0, layerIndex).reduce((sum, size) => sum + size, 0);
    const nextLayerStart = config.layers.slice(0, layerIndex + 1).reduce((sum, size) => sum + size, 0);
    
    for (let i = 0; i < config.layers[layerIndex]; i++) {
      for (let j = 0; j < config.layers[layerIndex + 1]; j++) {
        const startNode = nodes[currentLayerStart + i];
        const endNode = nodes[nextLayerStart + j];
        
        if (startNode && endNode) {
          // Determine if connection should be pruned
          const isConnectionPruned = step >= 4 && (startNode.isPruned || endNode.isPruned);
          let connectionPruningReason = "";
          
          if (isConnectionPruned) {
            if (startNode.isPruned && endNode.isPruned) {
              connectionPruningReason = "Both nodes pruned";
            } else if (startNode.isPruned) {
              connectionPruningReason = "Source pruned";
            } else {
              connectionPruningReason = "Target pruned";
            }
          }
          
          connections.push({
            start: new THREE.Vector3(...startNode.position),
            end: new THREE.Vector3(...endNode.position),
            isActive: step >= layerIndex + 2,
            isPruned: isConnectionPruned,
            strength: 0.5 + 0.5 * seededFloat(selectedModel || 'default-model-seed', 'conn', layerIndex, i, j),
            pruningReason: connectionPruningReason
          });
        }
      }
    }
  }

  // Camera fit and animation (also respond to external reset tick)
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
      // Also center the OrbitControls target on the network
      if (controls && controls.target) {
        controls.target.copy(center);
        controls.update();
      }
    }
  }, [step, selectedModel, camera, controls]);

  // Disable automatic camera animation to keep the scene still unless the user interacts
  // (Movement is now entirely controlled by OrbitControls on user drag only.)

  return (
    <group ref={networkRef}>
             {/* Layer Labels */}
       {config.layerNames.map((layerName, index) => (
         <Html key={`layer-${index}`} position={[index * config.spacing, 5, 0]} center>
           <div style={{
             background: 'rgba(0,0,0,0.95)',
             color: 'white',
             padding: '12px 20px',
             borderRadius: '25px',
             fontSize: '14px',
             fontWeight: 'bold',
             whiteSpace: 'nowrap',
             border: `3px solid ${config.colors[index]}`,
             boxShadow: `0 0 20px ${config.colors[index]}`,
             minWidth: '100px',
             textAlign: 'center',
             pointerEvents: 'none'
           }}>
             {layerName}
           </div>
         </Html>
       ))}
      
      {/* Connections */}
      {connections.map((conn, index) => (
        <Connection key={`conn-${index}`} {...conn} />
      ))}
      
             {/* Nodes */}
      {nodes.map((node) => (
        <NeuralNode key={node.id} {...node} totalLayers={config.layers.length} onNodeClick={onNodeClick} />
      ))}
      
      {/* Data flow particles */}
      <DataFlow step={step} isActive={step >= 1 && step <= 3} seedKey={selectedModel || 'default-model-seed'} />
      
             {/* Pruning Statistics */}
       {step >= 4 && (
         <Html position={[0, -5, 0]} center>
           <div style={{
             background: 'rgba(255, 0, 0, 0.95)',
             color: 'white',
             padding: '16px 24px',
             borderRadius: '20px',
             fontSize: '16px',
             fontWeight: 'bold',
             textAlign: 'center',
             border: '4px solid #ff0000',
             boxShadow: '0 0 25px #ff0000',
             minWidth: '250px',
             pointerEvents: 'none'
           }}>
             🔴 PRUNING IN PROGRESS
             <div style={{ fontSize: '14px', marginTop: '10px', opacity: 0.95 }}>
               Removing redundant nodes and connections...
             </div>
           </div>
         </Html>
       )}
    </group>
  );
}

// Step information with detailed explanations
const getStepInfo = (step, selectedModel) => {
  const steps = [
    {
      title: "Initialize Model",
      subtitle: `Load ${selectedModel}`,
      description: `Set up weights and layers.`,
      technicalDetails: [
        "Load weights",
        "Create layers"
      ],
      visualHint: "Layers appear left→right."
    },
    {
      title: "Process Input",
      subtitle: "Prepare Data",
      description: `Tokenize/normalize input.`,
      technicalDetails: [
        "Tokenize",
        "Embed"
      ],
      visualHint: "Particles show flow."
    },
    {
      title: "Forward Pass",
      subtitle: "Run Layers",
      description: `Compute outputs layer by layer.`,
      technicalDetails: [
        "Attention/conv",
        "Activations"
      ],
      visualHint: "Active links glow."
    },
    {
      title: "Knowledge Transfer",
      subtitle: "Teacher→Student",
      description: `Match teacher predictions.`,
      technicalDetails: [
        "Soft targets",
        "KD loss"
      ],
      visualHint: "Student adapts."
    },
    {
      title: "Prune Model",
      subtitle: "Trim Weights",
      description: `Remove low-importance weights.`,
      technicalDetails: [
        "L1 threshold",
        "~30% sparsity"
      ],
      visualHint: "🔴 Red = pruned."
    },
    {
      title: "Fine-tune",
      subtitle: "Stabilize",
      description: `Adjust to pruned structure.`,
      technicalDetails: [
        "Short retrain"
      ],
      visualHint: "Network stabilizes."
    },
    {
      title: "Final Results",
      subtitle: "Summary",
      description: `Smaller, faster, similar accuracy.`,
      technicalDetails: [
        "Latency",
        "Size",
        "Accuracy"
      ],
      visualHint: "Review compressed net."
    }
  ];
  
  return steps[step] || steps[0];
};

const Visualization = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { trainingComplete, selectedModel, metrics } = location.state || {};
  
  // Load persisted evaluation results if not passed via state
  const [persistedMetrics, setPersistedMetrics] = useState(null);
  
  useEffect(() => {
    if (!metrics) {
      const persistedResults = localStorage.getItem('kd_pruning_evaluation_results');
      if (persistedResults) {
        try {
          const parsedResults = JSON.parse(persistedResults);
          setPersistedMetrics(parsedResults);
        } catch (error) {
          console.error('Error parsing persisted results:', error);
        }
      }
    }
  }, [metrics]);
  const [started, setStarted] = useState(false);
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  // Visualization clarity controls (labels/reasons removed)
  // Focus layer and camera reset removed for simpler controls
  const [socketConnected, setSocketConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");
  const [vizMetrics, setVizMetrics] = useState(metrics || null);
  const [selectedNode, setSelectedNode] = useState(null);

  // Robust socket connection to keep server alive and stream metrics
  useEffect(() => {
    const testServerConnection = async () => {
      try {
        const response = await fetch(`${SOCKET_URL}/test`);
        const data = await response.json();
        if (data.status === "Server is running") {
          setServerStatus("connected");
        } else {
          setServerStatus("error");
        }
      } catch (e) {
        setServerStatus("error");
      }
    };

    testServerConnection();

    socket.on("connect", () => {
      setSocketConnected(true);
      setServerStatus("connected");
    });
    socket.on("connect_error", () => {
      setSocketConnected(false);
      setServerStatus("error");
    });
    socket.on("disconnect", () => {
      setSocketConnected(false);
      setServerStatus("error");
    });

    socket.on("training_metrics", (data) => {
      setVizMetrics((prev) => {
        if (!prev) return data;
        const merged = { ...prev };
        Object.keys(data).forEach((key) => {
          if (key === "error" || key === "basic_metrics") {
            merged[key] = data[key];
          } else {
            merged[key] = { ...merged[key], ...data[key] };
          }
        });
        return merged;
      });
    });

    const interval = setInterval(testServerConnection, 15000);
    return () => {
      clearInterval(interval);
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
      socket.off("training_metrics");
      // Do not disconnect here; keep the singleton alive for free navigation
    };
  }, []);

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
      }, 4000); // Increased from 3s to 4s to give users more time to read
      return () => clearTimeout(timer);
    }
  }, [autoPlay, step, started]);

  // If not trained, redirect or show warning (with server status)
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
          <div style={{ marginBottom: 12 }}>
            <span style={{ color: serverStatus === 'connected' ? 'green' : serverStatus === 'error' ? 'red' : 'orange' }}>
              ● {serverStatus === 'connected' ? 'Server Connected' : serverStatus === 'error' ? 'Server Disconnected' : 'Checking Connection...'}
            </span>
          </div>
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

  const handleNodeClick = (nodeData) => {
    setSelectedNode(nodeData);
  };

  const getNodeExplanation = (nodeData) => {
    if (!nodeData) return null;
    
    const explanations = {
      input: `Input Layer (${nodeData.label}): These nodes receive raw data and pass it to the first hidden layer. In neural networks, these nodes represent the features of your input data. Each input node corresponds to a specific feature or dimension of your input.`,
      hidden: `Hidden Layer ${nodeData.layerIndex + 1} (${nodeData.label}): These nodes process information between input and output layers. They learn complex patterns and relationships in the data through weighted connections. Each hidden node can detect different patterns or features in the data.`,
      output: `Output Layer (${nodeData.label}): These nodes produce the final predictions or classifications. The number of output nodes typically matches the number of possible outcomes. Each output node represents a different class or prediction value.`,
      pruned: `Pruned Node (${nodeData.label}): This node was removed during pruning because: ${nodeData.pruningReason}. Pruning helps reduce model size while maintaining performance by removing redundant or less important connections.`
    };

    if (nodeData.isPruned) {
      return explanations.pruned;
    } else if (nodeData.layerIndex === 0) {
      return explanations.input;
    } else if (nodeData.layerIndex === 3) { // Assuming 4 layers (0-3)
      return explanations.output;
    } else {
      return explanations.hidden;
    }
  };
  
  // Calculate dynamic pruning statistics based on model and current state
  const calculatePruningStats = () => {
    if (!started || step < 4) {
      return { nodePercentage: 0, connectionPercentage: 0, threshold: 0, method: "Not started" };
    }
    
    // Calculate actual pruning percentages based on model characteristics
    let baseThreshold = 0.4;
    let method = "Standard pruning";
    
    switch(selectedModel) {
      case "distillBert":
        baseThreshold = 0.35;
        method = "Attention-based pruning";
        break;
      case "T5-small":
        baseThreshold = 0.38;
        method = "Encoder-decoder pruning";
        break;
      case "MobileNetV2":
        baseThreshold = 0.45;
        method = "Depthwise pruning";
        break;
      case "ResNet-18":
        baseThreshold = 0.42;
        method = "Residual pruning";
        break;
      default:
        baseThreshold = 0.4;
        method = "General pruning";
    }
    
    // Simulate dynamic results based on model complexity
    const modelComplexity = selectedModel === "distillBert" ? 0.8 : 
                           selectedModel === "T5-small" ? 0.7 :
                           selectedModel === "MobileNetV2" ? 0.6 : 0.65;
    
    // Calculate node pruning percentage (varies by model)
    const nodePercentage = Math.round((baseThreshold * 100) + (Math.random() * 15 - 7.5));
    
    // Calculate connection pruning percentage (depends on node pruning)
    const connectionPercentage = Math.round(nodePercentage * 1.4 + (Math.random() * 10 - 5));
    
    return {
      nodePercentage: Math.max(15, Math.min(60, nodePercentage)), // Clamp between 15-60%
      connectionPercentage: Math.max(20, Math.min(70, connectionPercentage)), // Clamp between 20-70%
      threshold: Math.round(baseThreshold * 100),
      method: method
    };
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
                      <div style={{ fontSize: '4rem', marginBottom: '20px', fontWeight: 'bold' }}>🧠 Neural Network</div>
                      <Title level={1} style={{ color: 'white', marginBottom: '16px', fontSize: '2.5rem', fontWeight: 'bold' }}>
                        3D Neural Network Demo
                      </Title>
                      <Paragraph style={{ color: '#ccc', fontSize: '1.2rem', marginBottom: '24px', fontWeight: '400' }}>
                        Watch {selectedModel} compress in 3D
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
                        Start Demo
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
                        camera={{ position: [8, 4, 8], fov: 60, near: 0.01, far: 10000 }}
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
                        
                        <NeuralNetwork step={step} selectedModel={selectedModel} onNodeClick={handleNodeClick} />
                        
                        <OrbitControls 
                          makeDefault
                          enablePan={true} 
                          enableZoom={true} 
                          enableRotate={true}
                          maxDistance={2000}
                          minDistance={0.5}
                          dampingFactor={0.08}
                          enableDamping={true}
                          zoomSpeed={1.2}
                          panSpeed={1.2}
                          rotateSpeed={1.0}
                          screenSpacePanning={true}
                          minPolarAngle={0}
                          maxPolarAngle={Math.PI}
                        />
                        
                                                 {/* Model Label */}
                         <Html position={[-4, 5, 0]} center>
                           <div style={{
                             background: 'rgba(0,0,0,0.9)',
                             color: 'white',
                             padding: '10px 20px',
                             borderRadius: '25px',
                             fontSize: '16px',
                             fontWeight: 'bold',
                             whiteSpace: 'nowrap',
                             border: '2px solid #1890ff',
                             boxShadow: '0 0 15px #1890ff',
                             pointerEvents: 'none'
                           }}>
                             🧠 {selectedModel}
                           </div>
                         </Html>
                         
                         {/* Step indicator */}
                         <Html position={[0, 5, 0]} center>
                           <div style={{
                             background: 'rgba(0,0,0,0.9)',
                             color: 'white',
                             padding: '10px 20px',
                             borderRadius: '25px',
                             fontSize: '15px',
                             fontWeight: 'bold',
                             whiteSpace: 'nowrap',
                             border: '2px solid #fff',
                             boxShadow: '0 0 15px rgba(0,0,0,0.5)',
                             minWidth: '200px',
                             textAlign: 'center',
                             pointerEvents: 'none'
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
                                               {/* Model Header */}
                        <Card style={{ marginBottom: 16, borderRadius: '12px', background: 'linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%)' }}>
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '24px', marginBottom: '8px' }}>Neural Network</div>
                            <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
                              {selectedModel}
                            </Title>
                            <Paragraph style={{ margin: '8px 0 0 0', color: '#666', fontSize: '14px' }}>
                              Neural Network Simulation
                            </Paragraph>
                          </div>
                        </Card>
                        
                        {/* Instructions */}
                        <Card style={{ marginBottom: 16, borderRadius: '12px', background: 'linear-gradient(135deg, #fff7e6 0%, #fff2d9 100%)' }}>
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '20px', marginBottom: '8px' }}>Instructions</div>
                            <Title level={5} style={{ margin: '0 0 8px 0', color: '#d46b08' }}>
                              How to Use
                            </Title>
                            <div style={{ fontSize: '12px', color: '#666', lineHeight: '1.4' }}>
                              <div>Mouse: rotate • scroll: zoom</div>
                              <div>Auto-play or step manually</div>
                            </div>
                          </div>
                        </Card>
                       
                       {/* Step Information */}
                       <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                         <Title level={3} style={{ marginBottom: 8, color: '#1890ff' }}>
                           {stepInfo.title}
                         </Title>
                        <Paragraph style={{ color: '#666', marginBottom: 8 }}>
                          {stepInfo.subtitle}
                        </Paragraph>
                        <Paragraph style={{ fontSize: '13px', lineHeight: '1.5', marginBottom: 8 }}>
                          {stepInfo.description}
                        </Paragraph>
                        
                        <Divider style={{ margin: '16px 0' }} />
                        
                        <div style={{ marginBottom: 12 }}>
                          <AntText strong style={{ color: '#52c41a' }}>Visual Hint:</AntText>
                          <Paragraph style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                            {stepInfo.visualHint}
                          </Paragraph>
                        </div>

                        {showTechnicalDetails && (
                          <div style={{ marginTop: 16 }}>
                            <AntText strong style={{ color: '#722ed1' }}>Technical Details:</AntText>
                            <ul style={{ fontSize: '12px', color: '#666', marginTop: 8 }}>
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
                              Previous
                            </Button>
                            <Button 
                              onClick={nextStep} 
                              disabled={step === 6}
                              type="primary"
                              style={{ flex: 1 }}
                            >
                              Next
                            </Button>
                          </div>
                          
                                                     <Button 
                             onClick={() => setAutoPlay(!autoPlay)}
                             type={autoPlay ? "default" : "primary"}
                             style={{ width: '100%' }}
                           >
                             {autoPlay ? 'Stop Auto-play' : 'Start Auto-play'}
                           </Button>
                          
                          <Button 
                            onClick={resetSimulation}
                            style={{ width: '100%' }}
                          >
                            Reset Simulation
                          </Button>
                          
                          <Button 
                            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                            type="dashed"
                            style={{ width: '100%' }}
                          >
                            {showTechnicalDetails ? 'Hide' : 'Show'} Technical Details
                          </Button>
                          
                          <Button 
                            onClick={() => navigate('/training')}
                            type="default"
                            style={{ width: '100%' }}
                          >
                            Back to Training
                          </Button>

                          {/* Visualization clarity controls simplified */}
                          <Divider style={{ margin: '8px 0' }} />
                          {/* Focus layer and camera reset removed */}
                        </Space>
                      </Card>

                      {/* Enhanced Node Explanation Panel */}
                      {selectedNode && (
                        <Card style={{ marginBottom: 16, borderRadius: '12px', background: 'linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%)' }}>
                          <Title level={5} style={{ color: '#1890ff', marginBottom: 16 }}>
                            🧠 Node Analysis: {selectedNode.label}
                          </Title>
                          
                          <div style={{ fontSize: '13px', color: '#333', lineHeight: '1.6' }}>
                            <div style={{ marginBottom: '12px', padding: '8px', background: '#f0f9ff', borderRadius: '6px', border: '1px solid #91d5ff' }}>
                              <strong>📍 Layer Position:</strong> Layer {selectedNode.layerIndex + 1} of 4
                            </div>
                            
                            <div style={{ marginBottom: '12px', padding: '8px', background: selectedNode.isPruned ? '#fff2f0' : '#f6ffed', borderRadius: '6px', border: `1px solid ${selectedNode.isPruned ? '#ffccc7' : '#b7eb8f'}` }}>
                              <strong>🔧 Status:</strong> {selectedNode.isPruned ? '❌ Pruned (Removed)' : '✅ Active (Working)'}
                            </div>
                            
                            {selectedNode.isPruned && selectedNode.pruningReason && (
                              <div style={{ marginBottom: '12px', padding: '8px', background: '#fff2f0', borderRadius: '6px', border: '1px solid #ffccc7' }}>
                                <strong>✂️ Pruning Reason:</strong> {selectedNode.pruningReason}
                              </div>
                            )}
                            
                            <Divider style={{ margin: '12px 0' }} />
                            
                            <div style={{ marginBottom: '12px' }}>
                              <strong style={{ color: '#1890ff' }}>📚 Educational Explanation:</strong>
                            </div>
                            
                            <div style={{ fontSize: '12px', lineHeight: '1.5', color: '#555' }}>
                              {getNodeExplanation(selectedNode)}
                            </div>
                            
                            <Divider style={{ margin: '12px 0' }} />
                            
                            <div style={{ fontSize: '11px', color: '#666', fontStyle: 'italic' }}>
                              <strong>💡 Learning Tip:</strong> {
                                selectedNode.isPruned 
                                  ? "Pruned nodes show how neural networks can be made more efficient by removing unnecessary parts while maintaining performance."
                                  : selectedNode.layerIndex === 0
                                    ? "Input nodes are the 'eyes' of the neural network - they see and process the raw data."
                                    : selectedNode.layerIndex === 3
                                      ? "Output nodes are the 'brain' of the neural network - they make the final decisions."
                                      : "Hidden nodes are the 'thinking' part of the neural network - they process and transform information."
                              }
                            </div>
                          </div>
                          
                          <Button 
                            onClick={() => setSelectedNode(null)}
                            size="small"
                            style={{ width: '100%', marginTop: '12px' }}
                          >
                            Close Analysis
                          </Button>
                        </Card>
                      )}

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

                      {/* Visualization Legend */}
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Title level={5} style={{ marginBottom: 12 }}>What You're Seeing</Title>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ 
                              width: '12px', 
                              height: '12px', 
                              borderRadius: '50%', 
                              backgroundColor: '#4fc3f7', 
                              marginRight: '8px' 
                            }}></div>
                            <span>Working nodes</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ 
                              width: '12px', 
                              height: '12px', 
                              borderRadius: '50%', 
                              backgroundColor: '#ff4444', 
                              marginRight: '8px' 
                            }}></div>
                            <span>Removed nodes</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ 
                              width: '12px', 
                              height: '12px', 
                              borderRadius: '50%', 
                              backgroundColor: '#888', 
                              marginRight: '8px' 
                            }}></div>
                            <span>Working connections</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ 
                              width: '12px', 
                              height: '12px', 
                              borderRadius: '50%', 
                              backgroundColor: '#ff0000', 
                              marginRight: '8px' 
                            }}></div>
                            <span>Removed connections</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ 
                              width: '12px', 
                              height: '12px', 
                              borderRadius: '50%', 
                              backgroundColor: '#666', 
                              marginRight: '8px' 
                            }}></div>
                            <span>Inactive nodes</span>
                          </div>
                        </div>
                        <Divider style={{ margin: '12px 0' }} />
                        <div style={{ fontSize: '11px', color: '#999' }}>
                          <strong>Labels:</strong><br />
                          • N1-1, N1-2: First layer<br />
                          • N2-1, N2-2: Second layer<br />
                          • Red labels show why parts were removed
                        </div>
                      </Card>

                      {/* Training Results (always visible if available) */}
                      { (vizMetrics || metrics || persistedMetrics) && (
                        <Card style={{ borderRadius: '12px', background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)' }}>
                          <Title level={4} style={{ color: '#52c41a', marginBottom: 16 }}>
                            Compression Results
                          </Title>
                          
                          {(vizMetrics || metrics || persistedMetrics)?.model_performance && (
                            <div style={{ marginBottom: 16 }}>
                              <Row gutter={8}>
                                <Col span={12}>
                                  <div style={{ textAlign: 'center', padding: '12px', background: '#f6ffed', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#52c41a' }}>
                                      {(vizMetrics || metrics || persistedMetrics)?.model_performance?.metrics?.accuracy || 'N/A'}
                                    </div>
                                    <div style={{ fontSize: '12px', color: '#666' }}>Accuracy</div>
                                  </div>
                                </Col>
                                <Col span={12}>
                                  <div style={{ textAlign: 'center', padding: '12px', background: '#fff7e6', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#fa8c16' }}>
                                      {(vizMetrics || metrics || persistedMetrics)?.model_performance?.metrics?.size_mb || 'N/A'}
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
                                             {/* Pruning Statistics */}
                       {step >= 4 && (
                         <Card style={{ marginBottom: 16, borderRadius: '12px', background: 'linear-gradient(135deg, #fff2f0 0%, #fff1f0 100%)' }}>
                           <Title level={5} style={{ color: '#cf1322', marginBottom: 12 }}>Dynamic Pruning Results</Title>
                           <div style={{ fontSize: '12px', color: '#666' }}>
                             <Row gutter={8} style={{ marginBottom: '8px' }}>
                               <Col span={12}>
                                 <div style={{ textAlign: 'center', padding: '8px', background: '#fff2f0', borderRadius: '6px', border: '1px solid #ffccc7' }}>
                                   <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#cf1322' }}>
                                     {calculatePruningStats().nodePercentage}%
                                   </div>
                                   <div style={{ fontSize: '10px', color: '#666' }}>Nodes Cut</div>
                                 </div>
                               </Col>
                               <Col span={12}>
                                 <div style={{ textAlign: 'center', padding: '8px', background: '#fff2f0', borderRadius: '6px', border: '1px solid #ffccc7' }}>
                                   <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#cf1322' }}>
                                     {calculatePruningStats().connectionPercentage}%
                                   </div>
                                   <div style={{ fontSize: '10px', color: '#666' }}>Connections Cut</div>
                                 </div>
                               </Col>
                             </Row>
                             
                             <div style={{ marginTop: '12px' }}>
                               <div style={{ fontSize: '11px', color: '#cf1322', fontWeight: 'bold', marginBottom: '6px' }}>
                                 Pruning Strategy for {selectedModel}:
                               </div>
                               <div style={{ fontSize: '10px', color: '#666', marginBottom: '8px' }}>
                                 <strong>Threshold:</strong> {calculatePruningStats().threshold}%<br />
                                 <strong>Method:</strong> {calculatePruningStats().method}
                               </div>
                               <div style={{ fontSize: '11px', color: '#cf1322', fontWeight: 'bold', marginBottom: '6px' }}>
                                 Why These Parts Were Removed:
                               </div>
                               <ul style={{ fontSize: '10px', color: '#666', margin: 0, paddingLeft: '16px' }}>
                                 <li>Low activation scores</li>
                                 <li>Weak weight magnitudes</li>
                                 <li>Poor gradient flow</li>
                                 <li>Redundant features</li>
                                 <li>Edge position penalties</li>
                               </ul>
                             </div>
                           </div>
                         </Card>
                       )}
                    </>
                  ) : (
                    <Card style={{ borderRadius: '12px' }}>
                      <Title level={4}>3D Neural Network Demo</Title>
                      <Paragraph>
                        Watch your model in action:
                      </Paragraph>
                      <ul style={{ fontSize: '14px', color: '#666' }}>
                        <li>See the network structure</li>
                        <li>Watch data flow through</li>
                        <li>See parts get removed</li>
                        <li>Check the results</li>
                        <li>Learn step by step</li>
                      </ul>
                      <Divider />
                      <Paragraph style={{ fontSize: '12px', color: '#999' }}>
                        <strong>How to use:</strong><br />
                        • Mouse: Move the 3D view around<br />
                        • Auto-play: Watch everything automatically<br />
                        • Manual: Go step by step
                      </Paragraph>
                    </Card>
                  )}
                </div>
              </Col>
            </Row>
          </div>
        </Content>
      </Layout>
      <Footer />
    </>
  );
};

export default Visualization;