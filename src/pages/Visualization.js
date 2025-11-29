import React, { useState, useRef, useEffect } from "react";
import { Layout, Card, Button, Typography, Row, Col, Progress, Space, Divider, Collapse, Steps } from "antd";
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

const baselineModelInfo = {
  distillBert: {
    label: "DistilBERT",
    architecture: "6-layer Transformer (12 attention heads)",
    layerStructure: "Tokenizer → Embedding → 6 Transformer blocks → classification head",
    nodeSizes: "768 hidden units per block, attention heads highlight token interactions",
    parameters: "≈67 million parameters",
    effects: "KD preserves ~97% of BERT accuracy before pruning."
  },
  "T5-small": {
    label: "T5 Small",
    architecture: "Encoder-decoder Transformer (6 encoder + 6 decoder layers)",
    layerStructure: "Shared text-to-text stack with attention bridges",
    nodeSizes: "512 hidden size, multi-head attention across encoder & decoder",
    parameters: "≈61 million parameters",
    effects: "Unified text-to-text pipeline benefits from KD temperature scaling."
  },
  MobileNetV2: {
    label: "MobileNetV2",
    architecture: "Depthwise separable CNN with inverted residual blocks",
    layerStructure: "Conv → Bottleneck blocks → Pointwise projections → classifier",
    nodeSizes: "Channel sizes shrink via bottlenecks (1×1 + depthwise convs)",
    parameters: "≈3.5 million parameters",
    effects: "Pruning targets depthwise filters for fast edge deployment."
  },
  "ResNet-18": {
    label: "ResNet-18",
    architecture: "18-layer residual CNN",
    layerStructure: "Conv1 → Residual blocks (×4) → Global pool → FC classifier",
    nodeSizes: "64–512 channels; skip connections stabilize deep training",
    parameters: "≈11.7 million parameters",
    effects: "Pruning removes redundant residual filters while preserving skips."
  },
  default: {
    label: "Baseline Model",
    architecture: "High-capacity teacher network",
    layerStructure: "Feature extractor → latent blocks → classifier",
    nodeSizes: "Dense hidden units sized for accuracy first",
    parameters: "Tens of millions of parameters",
    effects: "Acts as the knowledge source for student compression."
  }
};

const getBaselineInfo = (modelKey) => {
  return baselineModelInfo[modelKey] || baselineModelInfo.default;
};

// Human-readable model type label for sidebar header
const getModelTypeLabel = (modelKey) => {
  if (!modelKey) return "Neural Network";
  const key = String(modelKey).toLowerCase();
  if (key.includes("distill") || key.includes("bert") || key.includes("t5")) {
    return "NLP Transformer";
  }
  if (key.includes("mobilenet") || key.includes("resnet")) {
    return "Vision CNN";
  }
  if (key.includes("uploaded")) {
    return "Uploaded Student Model";
  }
  return "Neural Network";
};

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

// Helper to get pruning ratio from metrics
const getPruningRatio = (metrics) => {
  if (metrics?.pruning_analysis?.pruning_details?.pruning_ratio) {
    const ratioStr = metrics.pruning_analysis.pruning_details.pruning_ratio;
    if (typeof ratioStr === 'string' && ratioStr.includes('%')) {
      return parseFloat(ratioStr) / 100;
    } else if (!isNaN(Number(ratioStr))) {
      return Number(ratioStr);
    }
  }
  return 0.3;
};

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
          color={isPruned ? "#ff0000" : color} 
          opacity={isPruned ? 0.9 : effectiveOpacity}
          transparent
          emissive={isPruned ? "#ff0000" : (isActive ? color : "#000")}
          emissiveIntensity={isPruned ? 0.8 : (isActive ? 0.3 : 0)}
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

// Layer Block Component - Rounded Rectangle for Layers
function LayerBlock({ position, width = 1.2, height = 0.8, depth = 0.3, color = "#4fc3f7", label = "", isActive = false, isPruned = false, opacity = 1, layerIndex = 0, isInput = false, isOutput = false, onNodeClick }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (isActive && meshRef.current && !isPruned) {
      meshRef.current.scale.x = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.05;
      meshRef.current.scale.y = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.05;
    }
    
    if (isPruned && meshRef.current) {
      meshRef.current.scale.x = 0.7 + Math.sin(state.clock.elapsedTime * 5) * 0.1;
      meshRef.current.scale.y = 0.7 + Math.sin(state.clock.elapsedTime * 5) * 0.1;
    }
  });

  const handleClick = (event) => {
    event.stopPropagation();
    if (onNodeClick) {
      onNodeClick({
        label,
        layerIndex,
        isPruned,
        color,
        position
      });
    }
  };

  // Pruned nodes: red, visible but faded, smaller
  const effectiveOpacity = isPruned ? opacity * 0.6 : opacity; // More visible for pruned (60% of base opacity)
  const blockColor = isPruned ? "#ff0000" : color; // Pure red (#ff0000) for pruned
  const scale = isPruned ? 0.7 : 1.0;

  return (
    <group position={position}>
      <Box
        ref={meshRef}
        args={[width * scale, height * scale, depth * scale]}
        onClick={handleClick}
        style={{ cursor: 'pointer' }}
      >
        <meshStandardMaterial
          color={blockColor}
          opacity={effectiveOpacity}
          transparent
          emissive={isPruned ? "#ff0000" : (isActive ? color : "#000")}
          emissiveIntensity={isPruned ? 0.8 : (isActive ? 0.3 : 0)} // Brighter red glow for pruned nodes
          roughness={0.3}
          metalness={0.1}
        />
      </Box>
      
      {/* Label - Always show for clarity */}
      {label && (
        <Html position={[0, height * scale * 0.6 + 0.3, 0]} center>
          <div style={{
            background: isPruned ? 'rgba(255, 68, 68, 0.95)' : 
                        isInput ? 'rgba(76, 175, 80, 0.95)' :
                        isOutput ? 'rgba(33, 150, 243, 0.95)' :
                        'rgba(0,0,0,0.9)',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '8px',
            fontSize: '12px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            border: isPruned ? '2px solid #ff0000' : '1px solid #fff',
            boxShadow: isPruned ? '0 0 10px #ff0000' : '0 0 5px rgba(0,0,0,0.5)',
            pointerEvents: 'none',
            minWidth: '80px',
            textAlign: 'center'
          }}>
            {label}
          </div>
        </Html>
      )}
    </group>
  );
}

// Directional Edge Component - With Arrowhead and weight-based thickness
function DirectionalEdge({ start, end, isActive = false, isPruned = false, strength = 1, isDotted = false, weight = 0.5 }) {
  const lineRef = useRef();
  const arrowRef = useRef();
  
  useFrame((state) => {
    if (isActive && lineRef.current && !isPruned) {
      lineRef.current.material.opacity = 0.4 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
    
    if (isPruned && lineRef.current) {
      lineRef.current.material.opacity = 0.1;
    }
  });

  const direction = new THREE.Vector3().subVectors(end, start).normalize();
  
  // Arrowhead position (slightly before end to avoid overlap)
  const arrowOffset = direction.clone().multiplyScalar(-0.3);
  const arrowPosition = new THREE.Vector3().addVectors(end, arrowOffset);
  
  // Calculate arrow rotation
  const arrowRotation = new THREE.Euler();
  arrowRotation.y = Math.atan2(direction.x, direction.z);
  arrowRotation.x = -Math.asin(direction.y);

  // Line thickness based on weight magnitude
  const lineThickness = isPruned ? 0.5 : Math.max(0.5, weight * 3); // 0.5-3px based on weight
  const effectiveOpacity = isPruned ? 0.4 : Math.min(0.8, 0.3 + weight * 0.5);
  // Color: red for pruned connections, gray for active connections
  const lineColor = isPruned ? "#ff0000" : "#666666";

  // Dotted line for pruned connections (red)
  if (isDotted || isPruned) {
    const segments = 20;
    const dots = [];
    
    for (let i = 0; i < segments; i += 2) { // Skip every other segment for dotted effect
      const t = i / segments;
      const dotPos = new THREE.Vector3().lerpVectors(start, end, t);
      dots.push(dotPos);
    }
    
    return (
      <group>
        {dots.map((dot, idx) => (
          <mesh key={idx} position={dot}>
            <sphereGeometry args={[0.04, 8, 8]} />
            <meshStandardMaterial
              color="#ff0000" // Red for pruned connections
              opacity={effectiveOpacity}
              transparent
            />
          </mesh>
        ))}
      </group>
    );
  }

  return (
    <group>
      {/* Main edge line with weight-based thickness */}
      <line ref={lineRef}>
        <bufferGeometry attach="geometry">
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([start.x, start.y, start.z, end.x, end.y, end.z])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          attach="material"
          color={lineColor}
          opacity={effectiveOpacity}
          transparent
          linewidth={lineThickness}
        />
      </line>

      {/* Arrowhead */}
      {!isPruned && (
        <group position={arrowPosition} rotation={arrowRotation}>
          <mesh ref={arrowRef}>
            <coneGeometry args={[0.06 * lineThickness, 0.15 * lineThickness, 8]} />
            <meshStandardMaterial
              color={lineColor}
              opacity={effectiveOpacity}
              transparent
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

// Distillation Flow Component - Shows knowledge transfer from teacher to student
function DistillationFlow({ start, end, controlPoint, isActive = false, strength = 0.4 }) {
  const curveRef = useRef();
  const arrowRef = useRef();
  
  useFrame((state) => {
    if (isActive && curveRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1;
      curveRef.current.material.opacity = 0.5 + pulse;
    }
  });

  // Create quadratic Bezier curve through control point
  const curve = new THREE.QuadraticBezierCurve3(start, controlPoint, end);
  const points = curve.getPoints(50);
  
  // Arrow position on the curve (near the end)
  const arrowT = 0.85;
  const arrowPos = curve.getPointAt(arrowT);
  const tangent = curve.getTangentAt(arrowT);
  
  const arrowRotation = new THREE.Euler();
  arrowRotation.y = Math.atan2(tangent.x, tangent.z);
  arrowRotation.x = -Math.asin(tangent.y);

  return (
    <group>
      {/* Curved flow line */}
      <line ref={curveRef}>
        <bufferGeometry attach="geometry">
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          attach="material"
          color="#333333"
          opacity={isActive ? 0.6 : 0.3}
          transparent
          linewidth={2}
          dashed={true}
          dashSize={0.3}
          gapSize={0.2}
        />
      </line>
      
      {/* Arrow showing direction of knowledge transfer */}
      {isActive && (
        <group position={arrowPos} rotation={arrowRotation}>
          <mesh ref={arrowRef}>
            <coneGeometry args={[0.08, 0.2, 8]} />
            <meshStandardMaterial
              color="#333333"
              opacity={0.8}
              transparent
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

// Teacher-Student Distillation Layout - Clear structural visualization
function createSimpleModelLayout(modelStructure, step, metrics) {
  if (!modelStructure || !modelStructure.nodes || modelStructure.nodes.length === 0) {
    return { nodes: [], connections: [], distillationFlows: [], stepInfo: null };
  }

  // Group nodes by layer
  const layers = {};
  modelStructure.nodes.forEach((node, idx) => {
    const layerIndex = node.layerIndex !== undefined ? node.layerIndex : idx;
    if (!layers[layerIndex]) {
      layers[layerIndex] = {
        index: layerIndex,
        nodes: [],
        label: node.label || `Layer ${layerIndex + 1}`,
        type: node.layerType || 'Unknown'
      };
    }
    layers[layerIndex].nodes.push({ ...node, originalIndex: idx });
  });

  const layerCount = Object.keys(layers).length;
  const layoutNodes = [];
  const layoutConnections = [];
  const distillationFlows = [];
  
  // Three-panel layout: Teacher (left) | Distillation (center) | Student (right)
  const panelWidth = 12;
  const teacherCenterX = -panelWidth;
  const distillationCenterX = 0;
  const studentCenterX = panelWidth;
  const horizontalSpacing = 3.0;
  const maxNodesPerLayer = 6; // Limit for clarity
  
  // Calculate node importance based on layer position and metrics
  const calculateImportance = (layerIndex, nodeIndex, totalNodes) => {
    // Important nodes: input, output, and middle layers with higher indices
    const layerImportance = layerIndex === 0 || layerIndex === layerCount - 1 ? 1.2 : 0.8;
    const nodeImportance = 0.7 + (nodeIndex / totalNodes) * 0.3;
    return layerImportance * nodeImportance;
  };
  
  // Create Teacher Model (Left Panel)
  Object.values(layers).sort((a, b) => a.index - b.index).forEach((layer, layerIdx) => {
    const x = teacherCenterX + (layerIdx - (layerCount - 1) / 2) * horizontalSpacing;
    const nodeCount = Math.min(layer.nodes.length, maxNodesPerLayer);
    const isInput = layer.index === 0;
    const isOutput = layer.index === layerCount - 1;
    
    const verticalSpacing = nodeCount > 1 ? 2.5 / Math.max(1, nodeCount - 1) : 0;
    const startY = -(nodeCount - 1) * verticalSpacing / 2;
    
    layer.nodes.slice(0, maxNodesPerLayer).forEach((node, nodeIdx) => {
      const y = nodeCount > 1 ? startY + nodeIdx * verticalSpacing : 0;
      const importance = calculateImportance(layer.index, nodeIdx, nodeCount);
      const nodeSize = 0.3 + importance * 0.4; // Node size based on importance
      
      layoutNodes.push({
        id: `teacher_${layer.index}_${nodeIdx}`,
        position: [x, y, 0],
        width: nodeSize,
        height: nodeSize,
        depth: 0.2,
        color: "#666666", // Neutral gray for structure
        label: isInput ? "Input" : (isOutput ? "Output" : ""),
        layerIndex: layer.index,
        nodeIndex: nodeIdx,
        isActive: step >= 1,
        isPruned: false,
        isOutput: isOutput,
        isInput: isInput,
        isTeacher: true,
        opacity: 1.0,
        importance: importance
      });
    });
  });
  
  // Create Student Model (Right Panel) - Smaller and pruned
  const pruningRatio = metrics?.pruning_analysis?.pruning_details?.pruning_ratio || 30;
  const studentLayers = Math.max(2, Math.floor(layerCount * 0.7)); // Student has fewer layers
  
  // Progressive pruning: show pruning happening during step 4, complete by step 5+
  const isPruningStep = step === 4; // Step 4: pruning in progress
  const isPrunedStep = step >= 5; // Step 5+: pruning complete
  
  for (let layerIdx = 0; layerIdx < studentLayers; layerIdx++) {
    const originalLayerIdx = Math.floor((layerIdx / studentLayers) * (layerCount - 1));
    const layer = Object.values(layers).sort((a, b) => a.index - b.index)[originalLayerIdx];
    if (!layer) continue;
    
    const x = studentCenterX + (layerIdx - (studentLayers - 1) / 2) * horizontalSpacing;
    const nodeCount = Math.max(2, Math.min(Math.floor(layer.nodes.length * 0.6), maxNodesPerLayer));
    const isInput = layerIdx === 0;
    const isOutput = layerIdx === studentLayers - 1;
    
    const verticalSpacing = nodeCount > 1 ? 2.0 / Math.max(1, nodeCount - 1) : 0;
    const startY = -(nodeCount - 1) * verticalSpacing / 2;
    
    // Calculate which nodes should be pruned (based on importance/position)
    const nodesToPrune = Math.floor(nodeCount * (pruningRatio / 100));
    
    for (let nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
      const y = nodeCount > 1 ? startY + nodeIdx * verticalSpacing : 0;
      const importance = calculateImportance(layerIdx, nodeIdx, nodeCount);
      
      // Prune nodes from the end (less important ones)
      // Show pruning starting from step 4, complete by step 5+
      const shouldPrune = nodeIdx >= (nodeCount - nodesToPrune) && nodesToPrune > 0;
      const isPruned = (isPruningStep || isPrunedStep) && shouldPrune;
      
      const nodeSize = (0.2 + importance * 0.3) * 0.8; // Smaller than teacher
      
      // Progressive opacity: during pruning (step 4) fade to 0.5, after (step 5+) fade to 0.3
      // Keep visible enough to clearly see they're being removed (red)
      const opacity = isPruned ? (isPruningStep ? 0.5 : 0.3) : 1.0;
      
      // Color: red for pruned nodes, gray for active nodes
      const nodeColor = isPruned ? "#ff0000" : "#666666";
      
      layoutNodes.push({
        id: `student_${layerIdx}_${nodeIdx}`,
        position: [x, y, 0],
        width: nodeSize,
        height: nodeSize,
        depth: 0.2,
        color: nodeColor,
        label: isInput ? "Input" : (isOutput ? "Output" : ""),
        layerIndex: layerIdx,
        nodeIndex: nodeIdx,
        isActive: step >= 2 && !isPruned,
        isPruned: isPruned,
        isOutput: isOutput,
        isInput: isInput,
        isStudent: true,
        opacity: opacity,
        importance: importance
      });
    }
  }
  
  // Create connections within Teacher Model
  const teacherNodes = layoutNodes.filter(n => n.isTeacher);
  Object.values(layers).sort((a, b) => a.index - b.index).forEach((layer, layerIdx) => {
    if (layerIdx < layerCount - 1) {
      const nextLayer = Object.values(layers).sort((a, b) => a.index - b.index)[layerIdx + 1];
      const currentLayerNodes = teacherNodes.filter(n => n.layerIndex === layer.index);
      const nextLayerNodes = teacherNodes.filter(n => n.layerIndex === nextLayer.index);
      
      currentLayerNodes.forEach((sourceNode) => {
        nextLayerNodes.forEach((targetNode) => {
          // Weight magnitude based on importance and random factor
          const weightMagnitude = 0.3 + (sourceNode.importance + targetNode.importance) / 2 * 0.4;
          
          layoutConnections.push({
            start: new THREE.Vector3(...sourceNode.position),
            end: new THREE.Vector3(...targetNode.position),
            isActive: step >= 1,
            isPruned: false,
            strength: weightMagnitude, // Line thickness = weight magnitude
            weight: weightMagnitude
          });
        });
      });
    }
  });
  
  // Create connections within Student Model
  const studentNodes = layoutNodes.filter(n => n.isStudent);
  for (let layerIdx = 0; layerIdx < studentLayers - 1; layerIdx++) {
    const currentLayerNodes = studentNodes.filter(n => n.layerIndex === layerIdx);
    const nextLayerNodes = studentNodes.filter(n => n.layerIndex === layerIdx + 1);
    
    currentLayerNodes.forEach((sourceNode) => {
      nextLayerNodes.forEach((targetNode) => {
        // Connection is pruned if either node is pruned
        const isPruned = (isPruningStep || isPrunedStep) && (sourceNode.isPruned || targetNode.isPruned);
        const weightMagnitude = 0.2 + (sourceNode.importance + targetNode.importance) / 2 * 0.3;
        
        layoutConnections.push({
          start: new THREE.Vector3(...sourceNode.position),
          end: new THREE.Vector3(...targetNode.position),
          isActive: step >= 2 && !isPruned,
          isPruned: isPruned,
          strength: isPruned ? 0.05 : weightMagnitude, // Very thin if pruned
          weight: weightMagnitude,
          isDotted: isPruned // Mark as dotted for rendering
        });
      });
    });
  }
  
  // Create Knowledge Distillation Flow Indicators (Center Panel)
  // Arrows from teacher to student showing knowledge transfer
  if (step >= 3) { // Show during KD step
    const teacherOutputNodes = teacherNodes.filter(n => n.isOutput);
    const studentInputNodes = studentNodes.filter(n => n.isInput);
    
    teacherOutputNodes.forEach((teacherNode, idx) => {
      const studentNode = studentInputNodes[Math.min(idx, studentInputNodes.length - 1)];
      if (studentNode) {
        distillationFlows.push({
          start: new THREE.Vector3(...teacherNode.position),
          end: new THREE.Vector3(...studentNode.position),
          controlPoint: new THREE.Vector3(
            distillationCenterX,
            (teacherNode.position[1] + studentNode.position[1]) / 2,
            0
          ),
          isActive: step >= 3,
          strength: 0.4
        });
      }
    });
  }
  
  // Step-specific information
  const stepInfo = {
    0: { title: "Model Initialized", description: "All layers loaded and ready" },
    1: { title: "Input Processing", description: "Processing input data through embedding layers" },
    2: { title: "Forward Pass", description: "Data flowing through all layers" },
    3: { title: "Knowledge Distillation", description: "Learning from teacher model" },
    4: { title: "Pruning", description: "Removing redundant connections" },
    5: { title: "Fine-tuning", description: "Recovering performance after pruning" },
    6: { title: "Complete", description: "Model optimized and ready" }
  };
  
  return { 
    nodes: layoutNodes, 
    connections: layoutConnections,
    distillationFlows: distillationFlows,
    stepInfo: stepInfo[step] || stepInfo[0]
  };
}

// Non-linear Model Layout - Creates a realistic neural network structure (kept for backward compatibility)
function createModelLayout(modelStructure, step, metrics) {
  if (!modelStructure || !modelStructure.nodes || modelStructure.nodes.length === 0) {
    return { nodes: [], connections: [], stepInfo: null };
  }

  // Group nodes by layer
  const layers = {};
  modelStructure.nodes.forEach((node, idx) => {
    const layerIndex = node.layerIndex !== undefined ? node.layerIndex : idx;
    if (!layers[layerIndex]) {
      layers[layerIndex] = {
        index: layerIndex,
        nodes: [],
        label: node.label || `Layer ${layerIndex + 1}`,
        type: node.layerType || 'Unknown'
      };
    }
    layers[layerIndex].nodes.push({ ...node, originalIndex: idx });
  });

  const layerCount = Object.keys(layers).length;
  const layoutNodes = [];
  const layoutConnections = [];
  
  // Create a non-linear layout: Input at center-left, branches out, converges to Output
  const baseRadius = 8.0;
  const angleStep = (Math.PI * 1.5) / Math.max(1, layerCount - 1); // Spread layers in an arc
  
  // Organize layers in a non-linear pattern
  Object.values(layers).sort((a, b) => a.index - b.index).forEach((layer, layerIdx) => {
    const isInput = layer.index === 0;
    const isOutput = layer.index === layerCount - 1;
    const nodeCount = layer.nodes.length;
    
    // Position layers in an arc pattern (non-linear)
    const angle = isInput ? -Math.PI / 4 : (isOutput ? Math.PI / 4 : (layerIdx - 1) * angleStep - Math.PI / 4);
    const radius = isInput ? baseRadius * 0.3 : (isOutput ? baseRadius * 0.3 : baseRadius * (0.5 + layerIdx * 0.15));
    const centerX = isInput ? -baseRadius * 0.8 : (isOutput ? baseRadius * 0.8 : 0);
    const centerY = 0;
    
    // Arrange nodes in each layer
    const nodeSpacing = nodeCount > 1 ? Math.min(2.0, 4.0 / nodeCount) : 0;
    const startOffset = -(nodeCount - 1) * nodeSpacing / 2;
    
    layer.nodes.forEach((node, nodeIdx) => {
      // Non-linear positioning: nodes spread vertically, layers spread in arc
      const nodeX = centerX + Math.cos(angle) * radius;
      const nodeY = centerY + Math.sin(angle) * radius + startOffset + nodeIdx * nodeSpacing;
      const nodeZ = (nodeIdx % 2 === 0 ? 0.5 : -0.5) * 0.3; // Slight depth variation
      
      // Determine pruning state based on current step
      // Once pruning happens (step >= 4), mark nodes as pruned and keep them pruned
      const pruningRatio = metrics?.pruning_analysis?.pruning_details?.pruning_ratio || 30;
      const isPruned = step >= 4 && (nodeIdx >= Math.floor(nodeCount * (1 - pruningRatio / 100)));
      
      // Node properties
      const nodeSize = isOutput ? 0.8 : (isInput ? 1.0 : 1.2);
      let nodeColor = node.color || "#4fc3f7";
      if (isPruned) {
        nodeColor = "#ff0000"; // Bright red for pruned nodes
      } else if (step === 3) {
        nodeColor = "#ff6b35"; // Orange during KD
      }
      
      // Determine visibility based on current step
      // Pruned nodes should always be visible after pruning step
      const isVisible = step === 0 ? isInput : 
                       step === 1 ? (isInput || layerIdx <= 1) :
                       step === 2 ? true : // All visible during forward pass
                       step === 3 ? true : // All visible during KD
                       step === 4 ? true : // All visible during pruning
                       step === 5 ? true : // All visible during fine-tuning
                       true; // All visible at final step
      
      layoutNodes.push({
        id: node.id || `node_${layer.index}_${nodeIdx}`,
        position: [nodeX, nodeY, nodeZ],
        width: nodeSize,
        height: nodeSize,
        depth: 0.3,
        color: nodeColor,
        label: node.label || layer.label,
        layerIndex: layer.index,
        nodeIndex: nodeIdx,
        isActive: step >= 2 && !isPruned,
        isPruned: isPruned,
        isOutput: isOutput,
        isInput: isInput,
        layerType: layer.type,
        isVisible: isVisible,
        opacity: isPruned ? 0.8 : (isVisible ? 1.0 : 0.3) // Higher opacity for pruned nodes so they're clearly visible
      });
    });
  });
  
  // Create connections - non-linear paths between layers
  Object.values(layers).sort((a, b) => a.index - b.index).forEach((layer, layerIdx) => {
    if (layerIdx < layerCount - 1) {
      const nextLayer = Object.values(layers).sort((a, b) => a.index - b.index)[layerIdx + 1];
      
      layer.nodes.forEach((node, nodeIdx) => {
        const sourceNode = layoutNodes.find(n => n.layerIndex === layer.index && n.nodeIndex === nodeIdx);
        if (sourceNode) {
          // Connect to nodes in next layer (create branching pattern)
          nextLayer.nodes.forEach((nextNode, nextIdx) => {
            const targetNode = layoutNodes.find(n => n.layerIndex === nextLayer.index && n.nodeIndex === nextIdx);
            if (targetNode) {
              // Connect with some branching (not just linear)
              const shouldConnect = nextLayer.nodes.length === 1 || 
                                   nodeIdx === nextIdx || 
                                   Math.abs(nodeIdx - nextIdx) <= 1 ||
                                   (nodeIdx === 0 && nextIdx === 0) ||
                                   (nodeIdx === layer.nodes.length - 1 && nextIdx === nextLayer.nodes.length - 1);
              
              if (shouldConnect) {
                // Connection is pruned if either source or target node is pruned
                // Once pruning happens (step >= 4), keep connections pruned
                const isPruned = step >= 4 && (sourceNode.isPruned || targetNode.isPruned);
                const isVisible = step >= 2; // Connections visible from forward pass
                
                layoutConnections.push({
                  start: new THREE.Vector3(...sourceNode.position),
                  end: new THREE.Vector3(...targetNode.position),
                  isActive: step >= 2 && !isPruned,
                  isPruned: isPruned,
                  strength: isPruned ? 0.3 : 0.7, // Slightly higher strength for visibility
                  isVisible: isVisible,
                  pruningReason: isPruned ? "Pruned connection" : ""
                });
              }
            }
          });
        }
      });
    }
  });
  
  // Step-specific information (only show current step)
  const stepInfo = {
    0: { title: "Model Initialized", description: "All layers loaded and ready" },
    1: { title: "Input Processing", description: "Processing input data through embedding layers" },
    2: { title: "Forward Pass", description: "Data flowing through all layers" },
    3: { title: "Knowledge Distillation", description: "Learning from teacher model" },
    4: { title: "Pruning", description: "Removing redundant connections" },
    5: { title: "Fine-tuning", description: "Recovering performance after pruning" },
    6: { title: "Complete", description: "Model optimized and ready" }
  };
  
  return { 
    nodes: layoutNodes.filter(n => n.isVisible !== false), 
    connections: layoutConnections.filter(c => c.isVisible !== false),
    stepInfo: stepInfo[step] || stepInfo[0]
  };
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
  // Pruned connections should be clearly visible in red
  const effectiveOpacity = isPruned ? 0.6 : strength; // Higher opacity for pruned connections

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

function NeuralNetwork({ step, selectedModel, onNodeClick, metrics, displayName, stepTitle, modelStructure }) {
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
          colors: ["#722ed1", "#531dab", "#391085", "#22075e"], // Purple gradient for baseline
          spacing: 3.2, // Increased spacing
          layerNames: ["Encoder", "Decoder", "Attention", "Output"]
        };
      case "MobileNetV2":
        return {
          layers: [6, 5, 4, 3], // Reduced from 8,6,4,3
          colors: ["#52c41a", "#389e0d", "#237804", "#135200"], // Green gradient for baseline
          spacing: 3.0, // Increased spacing
          layerNames: ["Conv", "Depthwise", "Pointwise", "Output"]
        };
      case "ResNet-18":
        return {
          layers: [5, 4, 3, 2], // Reduced from 6,5,4,3
          colors: ["#fa8c16", "#d46b08", "#ad4e00", "#873800"], // Orange gradient for baseline
          spacing: 3.3, // Increased spacing
          layerNames: ["Conv1", "ResBlock", "ResBlock", "Output"]
        };
      case "uploaded_custom":
        // Use real model structure if available
        if (modelStructure && modelStructure.nodes && modelStructure.nodes.length > 0) {
          const layerCount = modelStructure.layer_count || modelStructure.nodes.length;
          const layerNames = modelStructure.nodes.map((node, idx) => 
            node.label || node.layerType || `Layer ${idx + 1}`
          );
          const layers = Array(layerCount).fill(3); // Represent each layer with 3 nodes for visualization
          return {
            layers: layers,
            colors: modelStructure.nodes.map(n => n.color || "#ff6b35"), // Orange/red for uploaded model
            spacing: 3.2,
            layerNames: layerNames,
            useRealStructure: true
          };
        }
        // Fallback to estimated structure - different colors for uploaded model
        const ratio = Math.max(0.2, Math.min(0.7, getPruningRatio(metrics)));
        const hiddenSize = Math.max(3, Math.round(6 - ratio * 4));
        return {
          layers: [6, hiddenSize, Math.max(3, hiddenSize - 1), 3],
          colors: ["#ff6b35", "#ff8c42", "#ffa366", "#ffb88c"], // Orange gradient for uploaded model
          spacing: 3.2,
          layerNames: ["Input", "KD Core", "Pruned", "Output"]
        };
      case "uploaded_placeholder":
        return {
          layers: [4, 3, 2, 1],
          colors: ["#cfd8dc", "#b0bec5", "#90a4ae", "#78909c"],
          spacing: 2.8,
          layerNames: ["Awaiting Upload", "—", "—", "Output"]
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
  
  // Use teacher-student distillation layout for uploaded models
  if (selectedModel === "uploaded_custom" && modelStructure && config.useRealStructure) {
    const modelLayout = createSimpleModelLayout(modelStructure, step, metrics);
    nodes.push(...modelLayout.nodes);
    connections.push(...modelLayout.connections);
    // Store distillation flows and step info for display
    window.distillationFlows = modelLayout.distillationFlows || [];
    window.currentStepInfo = modelLayout.stepInfo;
  } else {
  
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
    const isPruned = step >= 4;
    const isActive = step >= layerIndex + 1;
    const pruningRatio = getPruningRatio(metrics);
    for (let i = 0; i < layerSize; i++) {
      const y = (layerSize - 1) / 2 - i;
      const z = (seededFloat(selectedModel || 'default-model-seed', 'z', layerIndex, i) - 0.5) * 1.0;
      let shouldPrune = false;
      let pruningReason = "";
      let nodeLabel = `N${layerIndex+1}-${i+1}`;
      if (isPruned) {
        shouldPrune = (i >= Math.floor((1 - pruningRatio) * layerSize));
        if (shouldPrune) pruningReason = "Pruned by ratio";
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
  } // End of else block for default node generation

  // Generate connections with pruning logic (only if not using real structure)
  if (!(selectedModel === "uploaded_custom" && modelStructure && config.useRealStructure)) {
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
  } // End of connection generation for default models (closing the if block)

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
      
      {/* Render Nodes and Connections - Hierarchical layout for uploaded models */}
      {selectedModel === "uploaded_custom" && modelStructure && config.useRealStructure ? (
        <>
          {/* Current Step Display - Only show current step */}
          {window.currentStepInfo && (
            <Html position={[0, 6, 0]} center>
              <div style={{
                background: 'rgba(0, 0, 0, 0.9)',
                color: 'white',
                padding: '12px 20px',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: 'bold',
                textAlign: 'center',
                border: '2px solid #4fc3f7',
                boxShadow: '0 0 20px rgba(79, 195, 247, 0.5)',
                minWidth: '300px',
                pointerEvents: 'none'
              }}>
                <div style={{ fontSize: '18px', marginBottom: '4px' }}>
                  {window.currentStepInfo.title}
                </div>
                <div style={{ fontSize: '13px', opacity: 0.9, fontWeight: 'normal' }}>
                  {window.currentStepInfo.description}
                </div>
              </div>
            </Html>
          )}
          
          {/* Section Labels: Teacher | Distillation | Student */}
          <Html position={[-12, 4, 0]} center>
            <div style={{
              background: 'rgba(0, 0, 0, 0.8)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 'bold',
              pointerEvents: 'none'
            }}>
              Teacher Model
            </div>
          </Html>
          <Html position={[0, 4, 0]} center>
            <div style={{
              background: 'rgba(0, 0, 0, 0.8)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 'bold',
              pointerEvents: 'none'
            }}>
              Knowledge Transfer
            </div>
          </Html>
          <Html position={[12, 4, 0]} center>
            <div style={{
              background: 'rgba(0, 0, 0, 0.8)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: 'bold',
              pointerEvents: 'none'
            }}>
              Student Model
            </div>
          </Html>

          {/* Render nodes - size based on importance */}
          {nodes.map((node) => {
            return (
              <LayerBlock
                key={node.id}
                position={node.position}
                width={node.width}
                height={node.height}
                depth={node.depth}
                color={node.color}
                label={node.label}
                isActive={node.isActive}
                isPruned={node.isPruned}
                opacity={node.opacity !== undefined ? node.opacity : (node.isPruned ? 0.8 : 1)}
                layerIndex={node.layerIndex}
                isInput={node.isInput}
                isOutput={node.isOutput}
                onNodeClick={onNodeClick}
              />
            );
          })}
          
          {/* Render connections with weight-based thickness */}
          {connections.map((conn, idx) => {
            return (
              <DirectionalEdge
                key={`edge-${idx}`}
                start={conn.start}
                end={conn.end}
                isActive={conn.isActive}
                isPruned={conn.isPruned}
                strength={conn.strength || 0.8}
                weight={conn.weight || conn.strength || 0.5}
                isDotted={conn.isDotted || false}
              />
            );
          })}
          
          {/* Render distillation flows (knowledge transfer arrows) */}
          {window.distillationFlows && window.distillationFlows.map((flow, idx) => {
            return (
              <DistillationFlow
                key={`flow-${idx}`}
                start={flow.start}
                end={flow.end}
                controlPoint={flow.controlPoint}
                isActive={flow.isActive}
                strength={flow.strength}
              />
            );
          })}
        </>
      ) : (
        <>
          {/* Standard visualization for baseline models */}
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
        </>
      )}
      
          {/* Layer Labels - Clear labels for each layer */}
          {selectedModel === "uploaded_custom" && modelStructure && config.useRealStructure && (
            <>
              {nodes.filter(n => n.isInput || n.isOutput || n.layerIndex % 2 === 0).map((node) => (
                <Html key={`label-${node.id}`} position={[node.position[0], node.position[1] + node.height * 0.7, node.position[2]]} center>
           <div style={{
                    background: node.isInput ? 'rgba(76, 175, 80, 0.9)' : 
                               node.isOutput ? 'rgba(33, 150, 243, 0.9)' : 
                               'rgba(0, 0, 0, 0.7)',
             color: 'white',
                    padding: '4px 8px',
                    borderRadius: '6px',
                    fontSize: '11px',
             fontWeight: 'bold',
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                    border: '1px solid rgba(255, 255, 255, 0.3)'
                  }}>
                    {node.label}
           </div>
         </Html>
              ))}
            </>
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
  const { 
    trainingComplete: stateTrainingComplete, 
    selectedModel: stateSelectedModel, 
    metrics: stateMetrics,
    uploadedModelMeta: stateUploadedModelMeta
  } = location.state || {};

  // Load persisted evaluation results if not passed via state
  const [persistedResults, setPersistedResults] = useState(null);

  useEffect(() => {
    if (!stateMetrics) {
      const persisted = localStorage.getItem('kd_pruning_evaluation_results');
      if (persisted) {
        try {
          const parsed = JSON.parse(persisted);
          setPersistedResults(parsed);
        } catch (error) {
          console.error('Error parsing persisted results:', error);
        }
      }
    }
  }, [stateMetrics]);

  useEffect(() => {
    if (stateUploadedModelMeta) {
      setUploadedModelMeta(stateUploadedModelMeta);
      localStorage.setItem('kd_uploaded_model_meta', JSON.stringify(stateUploadedModelMeta));
    }
  }, [stateUploadedModelMeta]);

  // Use state values if available, otherwise use persisted values
  const trainingComplete = stateTrainingComplete || (persistedResults ? true : false);
  const selectedModel = stateSelectedModel || (persistedResults ? persistedResults.selectedModel : null) || "distillBert";
  const metrics = stateMetrics || persistedResults;
  const [uploadedModelMeta, setUploadedModelMeta] = useState(() => {
    if (stateUploadedModelMeta) return stateUploadedModelMeta;
    try {
      const cached = localStorage.getItem('kd_uploaded_model_meta');
      return cached ? JSON.parse(cached) : null;
    } catch {
      return null;
    }
  });
  const [started, setStarted] = useState(false);
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
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
  const [modelStructure, setModelStructure] = useState(null);
  const metricsSource = vizMetrics || metrics || persistedResults;
  const studentMetrics = metricsSource?.model_performance?.metrics || {};
  const teacherComparison = metricsSource?.teacher_vs_student?.comparison || {};
  const pruningImpact = metricsSource?.pruning_analysis?.impact_analysis || {};
  const baselineSummary = getBaselineInfo(selectedModel);
  
  // Fetch model structure from backend
  useEffect(() => {
    const fetchModelStructure = async () => {
      if (trainingComplete && uploadedModelMeta) {
        try {
          const response = await fetch(`${SOCKET_URL}/visualize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          });
          const data = await response.json();
          if (data.success && data.data) {
            setModelStructure(data.data);
            console.log('Model structure fetched:', data.data);
          }
        } catch (error) {
          console.error('Error fetching model structure:', error);
        }
      }
    };
    
    fetchModelStructure();
  }, [trainingComplete, uploadedModelMeta]);
  const uploadedSummary = {
    architecture: uploadedModelMeta
      ? `Custom checkpoint (${uploadedModelMeta.name}) distilled from ${baselineSummary.label}`
      : "Awaiting upload from Training Page",
    layerStructure: `KD pipeline mirrors ${baselineSummary.label} layers before pruning.`,
    nodeSizes: studentMetrics?.num_params
      ? `${studentMetrics.num_params.toLocaleString?.() || studentMetrics.num_params} active parameters after pruning`
      : "Student parameters reported after training",
    parameters: studentMetrics?.size_mb
      ? `${studentMetrics.size_mb} model size • ${studentMetrics.latency_ms || 'N/A'} ms latency`
      : "Size and latency reported after KD",
    effects: pruningImpact?.parameter_reduction
      ? `Compression: ${pruningImpact.parameter_reduction} parameter reduction, ${pruningImpact.speed_improvement || 'N/A'} speed boost.`
      : "Compression metrics will appear after training."
  };

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

    // Listen for evaluation metrics (4 categories)
    socket.on("evaluation_metrics", (data) => {
      console.log("Received evaluation metrics in Visualization:", data);
      setVizMetrics((prev) => {
        const merged = { ...prev };
        merged.evaluation_metrics = data;
        return merged;
      });
    });

    // Listen for model structure updates
    socket.on("model_structure", (data) => {
      console.log("Received model structure in Visualization:", data);
      if (data.success && data.structure) {
        setModelStructure(data.structure);
      }
    });

    const interval = setInterval(testServerConnection, 15000);
    return () => {
      clearInterval(interval);
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
      socket.off("training_metrics");
      socket.off("evaluation_metrics");
      socket.off("model_structure");
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

  // Check if training is complete - restrict access if not
  const isTrainingComplete = trainingComplete || (persistedResults && persistedResults.trainingComplete);
  
  if (!isTrainingComplete) {
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
          <Content style={{ padding: "20px", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Card style={{ maxWidth: 600, textAlign: 'center', borderRadius: '16px' }}>
              <Title level={2} style={{ color: '#ff4d4f', marginBottom: 16 }}>
                Training Not Complete
              </Title>
              <Paragraph style={{ fontSize: '16px', marginBottom: 24 }}>
                You must complete training on the Training page before accessing the Visualization page.
              </Paragraph>
              <Button type="primary" size="large" onClick={() => navigate('/training')}>
                Go to Training Page
              </Button>
            </Card>
          </Content>
        </Layout>
        <Footer />
      </>
    );
  }

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
          <div style={{ maxWidth: 1600, margin: '0 auto' }}>
            {/* Header Section: Page Type and Instructions */}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              {/* Page Type Section - Shows both Baseline and Uploaded Model */}
              <Col xs={24}>
                <Card style={{ borderRadius: '12px', background: 'linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%)' }}>
                  <div style={{ textAlign: 'center', marginBottom: 16 }}>
                    <div style={{ fontSize: '20px', marginBottom: '12px' }}> Page Type</div>
                    <Row gutter={[24, 16]}>
                      {/* Baseline Model */}
                      <Col xs={24} md={12}>
                        <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.6)', borderRadius: '8px' }}>
                          <Title level={5} style={{ margin: '0 0 4px 0', color: '#1890ff' }}>
                            🔵 Baseline Model
                          </Title>
                          <Title level={4} style={{ margin: 0, color: '#333' }}>
                            {baselineSummary.label}
                          </Title>
                          <Paragraph style={{ margin: '4px 0 0 0', color: '#666', fontSize: '13px' }}>
                            {getModelTypeLabel(selectedModel)}
                          </Paragraph>
                        </div>
                      </Col>
                      {/* Uploaded Model */}
                      <Col xs={24} md={12}>
                        <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.6)', borderRadius: '8px' }}>
                          <Title level={5} style={{ margin: '0 0 4px 0', color: '#ff6b35' }}>
                            🧪 Uploaded Model
                          </Title>
                          {uploadedModelMeta ? (
                            <>
                              <Title level={4} style={{ margin: 0, color: '#333' }}>
                                {uploadedModelMeta?.name || "Your Model"}
                              </Title>
                              <Paragraph style={{ margin: '4px 0 0 0', color: '#666', fontSize: '13px' }}>
                                Trained & Compressed
                              </Paragraph>
                            </>
                          ) : (
                            <>
                              <Title level={4} style={{ margin: 0, color: '#999' }}>
                                Not Available
                              </Title>
                              <Paragraph style={{ margin: '4px 0 0 0', color: '#999', fontSize: '13px' }}>
                                Upload a model on Training page
                              </Paragraph>
                            </>
                          )}
                        </div>
                      </Col>
                    </Row>
                  </div>
                </Card>
              </Col>
              {/* Instructions Section - Below Model Type */}
              <Col xs={24}>
                <Card style={{ borderRadius: '12px', background: 'linear-gradient(135deg, #fff7e6 0%, #fff2d9 100%)' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '20px', marginBottom: '8px' }}> Instructions</div>
                    <Title level={5} style={{ margin: '0 0 8px 0', color: '#d46b08' }}>
                      How to Use
                    </Title>
                    <div style={{ fontSize: '13px', color: '#666', lineHeight: '1.6' }}>
                      <div> Mouse: Rotate view • Scroll: Zoom</div>
                      <div> Auto-play or step manually</div>
                    </div>
                  </div>
                </Card>
              </Col>
            </Row>

            {/* Two Simulations Side-by-Side */}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              {/* Left: Baseline Model Simulation */}
              <Col xs={24} lg={12}>
                <Row gutter={[16, 16]}>
                  {/* Step Guide Above Baseline Simulation */}
                  {started && (
                    <Col xs={24}>
                      <Card style={{ 
                        marginBottom: 16,
                        borderRadius: '12px',
                        background: 'linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%)',
                        border: '2px solid #1890ff',
                        textAlign: 'center'
                      }}>
                        <Title level={4} style={{ color: '#1890ff', marginBottom: 8 }}>
                          🔵 Baseline Model: {baselineSummary.label}
                        </Title>
                        <Paragraph style={{ color: '#333', fontSize: '15px', marginBottom: 0 }}>
                          <strong>Current Step:</strong> {stepInfo.title}
                          <br />
                          <span style={{ fontSize: '13px', opacity: 0.8 }}>{stepInfo.description}</span>
                        </Paragraph>
                      </Card>
                    </Col>
                  )}

                  {/* Top: Baseline Model */}
                  <Col xs={24}>
                    <Card 
                      className="visualization-container"
                      style={{ 
                        height: '75vh', 
                        background: '#ffffff',
                        border: '1px solid #d9d9d9',
                        borderRadius: '12px',
                        overflow: 'hidden',
                        padding: 0,
                        position: 'relative',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}
                    >
                      <div style={{ 
                        position: 'absolute',
                        top: 10,
                        left: 10,
                        zIndex: 10,
                        background: 'rgba(255, 255, 255, 0.9)',
                        padding: '8px 16px',
                        borderRadius: '8px',
                        fontWeight: 'bold',
                        color: '#1890ff'
                      }}>
                        Baseline Model: {baselineSummary.label}
                      </div>
                      {!started ? (
                        <div style={{ 
                          height: '100%', 
                          display: 'flex', 
                          flexDirection: 'column',
                          justifyContent: 'center', 
                          alignItems: 'center',
                          color: '#333',
                          textAlign: 'center',
                          padding: '0 12px'
                        }}>
                          <div style={{ fontSize: '3rem', marginBottom: '12px', fontWeight: 'bold' }}></div>
                          <Title level={2} className="page-hero-title" style={{ marginBottom: '12px', color: '#333' }}>
                            Baseline Model View
                          </Title>
                          <Paragraph className="page-hero-subtitle" style={{ color: '#666', fontSize: '1.05rem', marginBottom: '20px', fontWeight: '400' }}>
                            This is the <strong>{baselineSummary.label}</strong> baseline model - a pre-trained reference model that shows the original, uncompressed neural network structure. 
                            Click below to visualize how it works and compare it with your trained model.
                          </Paragraph>
                          <Button 
                            type="primary" 
                            size="large" 
                            onClick={startSimulation}
                            style={{ 
                              height: '44px', 
                              fontSize: '15px',
                              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                              border: 'none'
                            }}
                          >
                            Start Visualization
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
                              background: '#ffffff',
                              width: '100%',
                              height: '100%',
                              display: 'block'
                            }}
                            gl={{ 
                              antialias: true, 
                              alpha: false,
                              powerPreference: "high-performance"
                            }}
                            onCreated={({ gl, scene }) => {
                              gl.setClearColor('#ffffff', 1);
                              scene.background = new THREE.Color('#ffffff');
                            }}
                            dpr={Math.min(window.devicePixelRatio, 2)}
                          >
                            <ambientLight intensity={0.4} />
                            <pointLight position={[10, 10, 10]} intensity={1} />
                            <pointLight position={[-10, -10, -10]} intensity={0.5} />
                            
                            <NeuralNetwork 
                              step={step} 
                              selectedModel={selectedModel} 
                              displayName={baselineSummary.label}
                              stepTitle={stepInfo.title}
                              onNodeClick={handleNodeClick} 
                              metrics={metricsSource}
                              modelStructure={null}
                            />
                            
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
                          </Canvas>
                        </div>
                      )}
                    </Card>
                  </Col>
                </Row>
                  </Col>
                  
              {/* Right: Uploaded Model Simulation */}
              <Col xs={24} lg={12}>
                <Row gutter={[16, 16]}>
                  {/* Step Guide Above Uploaded Model Simulation */}
                  {started && uploadedModelMeta && (
                    <Col xs={24}>
                      <Card style={{ 
                        marginBottom: 16,
                        borderRadius: '12px',
                        background: 'linear-gradient(135deg, #fff2f0 0%, #fff1f0 100%)',
                        border: '2px solid #ff6b35',
                        textAlign: 'center'
                      }}>
                        <Title level={4} style={{ color: '#ff6b35', marginBottom: 8 }}>
                          🧪 Your Uploaded Model: {uploadedModelMeta?.name || "Your Model"}
                        </Title>
                        <Paragraph style={{ color: '#333', fontSize: '15px', marginBottom: 0 }}>
                          <strong>Current Step:</strong> {stepInfo.title}
                          <br />
                          <span style={{ fontSize: '13px', opacity: 0.8 }}>{stepInfo.description}</span>
                        </Paragraph>
                      </Card>
                    </Col>
                  )}

                  {/* Uploaded Model Simulation */}
                  <Col xs={24}>
                    <Card
                      className="visualization-container"
                      style={{
                        height: '75vh',
                        background: '#ffffff',
                        border: '1px solid #d9d9d9',
                        borderRadius: '12px',
                        overflow: 'hidden',
                        padding: 0,
                        position: 'relative',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}
                    >
                      <div style={{ 
                        position: 'absolute',
                        top: 10,
                        left: 10,
                        zIndex: 10,
                        background: 'rgba(255, 255, 255, 0.9)',
                        padding: '8px 16px',
                        borderRadius: '8px',
                        fontWeight: 'bold',
                        color: '#ff6b35'
                      }}>
                        Trained Uploaded Model: {uploadedModelMeta?.name || "Your Model"}
                      </div>
                      {!uploadedModelMeta ? (
                        <div style={{
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'center',
                          alignItems: 'center',
                          color: '#333',
                          textAlign: 'center',
                          padding: '0 16px',
                        }}>
                          <div style={{ fontSize: '3rem', marginBottom: '12px' }}>📤</div>
                          <Title level={3} style={{ color: '#ff6b35', marginBottom: 12 }}>No Uploaded Model</Title>
                          <Paragraph style={{ color: '#666' }}>
                            Upload a custom model on the Training Page to see its visualization here.
                          </Paragraph>
                        </div>
                      ) : !started ? (
                        <div style={{
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'center',
                          alignItems: 'center',
                          color: '#333',
                          textAlign: 'center',
                          padding: '0 14px'
                        }}>
                          <div style={{ fontSize: '3rem', marginBottom: '12px', fontWeight: 'bold' }}></div>
                          <Title level={4} style={{ color: '#ff6b35', marginBottom: 12 }}>{uploadedModelMeta?.name || "Your Model"}</Title>
                          <Paragraph style={{ color: '#666', marginBottom: 20, fontSize: '1.05rem', lineHeight: '1.6' }}>
                            This is <strong>your uploaded model</strong> that has been trained with Knowledge Distillation and Pruning on the Training page. 
                            It's the compressed, optimized version of your model. Click below to visualize how it compares to the baseline model above.
                          </Paragraph>
                          <Button
                            type="primary"
                            size="large"
                            onClick={startSimulation}
                            style={{
                              height: '44px',
                              fontSize: '15px',
                              background: 'linear-gradient(135deg, #ff6b35 0%, #f7931e 100%)',
                              border: 'none'
                            }}
                          >
                            Start Visualization
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
                              background: '#ffffff',
                              width: '100%',
                              height: '100%',
                              display: 'block'
                            }}
                            gl={{
                              antialias: true,
                              alpha: false,
                              powerPreference: "high-performance"
                            }}
                            onCreated={({ gl, scene }) => {
                              gl.setClearColor('#ffffff', 1);
                              scene.background = new THREE.Color('#ffffff');
                            }}
                            dpr={Math.min(window.devicePixelRatio, 2)}
                          >
                            <ambientLight intensity={0.4} />
                            <pointLight position={[10, 10, 10]} intensity={1} />
                            <pointLight position={[-10, -10, -10]} intensity={0.5} />
                            <NeuralNetwork
                              step={step}
                              selectedModel={uploadedModelMeta ? "uploaded_custom" : "uploaded_placeholder"}
                              displayName={uploadedModelMeta?.name || "Custom Upload"}
                              stepTitle={stepInfo.title}
                              onNodeClick={handleNodeClick}
                              metrics={metricsSource}
                              modelStructure={modelStructure}
                            />
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
                          </Canvas>
                        </div>
                      )}
                    </Card>
                  </Col>

                </Row>
              </Col>
            </Row>

            {/* Step Information - Below Simulations */}
            {started && (
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24}>
                  <Card style={{ borderRadius: '12px', textAlign: 'center' }}>
                         <Title level={3} style={{ marginBottom: 8, color: '#1890ff' }}>
                           {stepInfo.title}
                         </Title>
                    <Paragraph style={{ fontSize: '15px', color: '#666', marginBottom: 16 }}>
                          {stepInfo.description}
                        </Paragraph>
                    <Progress 
                      percent={((step + 1) / 7) * 100} 
                      status="active"
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                      style={{ maxWidth: '600px', margin: '0 auto' }}
                    />
                    <div style={{ textAlign: 'center', marginTop: 8, fontSize: '14px', color: '#666' }}>
                      Step {step + 1} of 7
                        </div>
                      </Card>
                </Col>
              </Row>
            )}

            {/* Control Buttons - Below Step Info */}
            {started && (
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24}>
                  <Card style={{ borderRadius: '12px', background: '#ffffff', border: '1px solid #d9d9d9' }}>
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                          <div style={{ display: 'flex', gap: 8 }}>
                            <Button 
                              onClick={prevStep} 
                              disabled={step === 0}
                              style={{ flex: 1 }}
                          size="large"
                            >
                              Previous
                            </Button>
                            <Button 
                              onClick={nextStep} 
                              disabled={step === 6}
                              type="primary"
                              style={{ flex: 1 }}
                          size="large"
                            >
                              Next
                            </Button>
                          </div>
                          
                                                     <Button 
                             onClick={() => setAutoPlay(!autoPlay)}
                             type={autoPlay ? "default" : "primary"}
                             style={{ width: '100%' }}
                        size="large"
                           >
                             {autoPlay ? 'Stop Auto-play' : 'Start Auto-play'}
                           </Button>
                          
                          <Button 
                            onClick={resetSimulation}
                            style={{ width: '100%' }}
                        size="large"
                          >
                            Reset Simulation
                          </Button>
                        </Space>
                      </Card>
                </Col>
              </Row>
            )}

            {/* Comparison Sections - Below Controls */}
            {started && uploadedModelMeta && metricsSource && (
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24}>
                  <Row gutter={[16, 16]}>
                    {/* Quick Model Comparison */}
                    <Col xs={24} lg={12}>
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Collapse
                          items={[{
                            key: '1',
                            label: <Title level={5} style={{ margin: 0 }}>📊 Quick Model Comparison</Title>,
                            children: (
                              <div>
                                <div style={{ marginBottom: 16 }}>
                                  <AntText strong style={{ fontSize: '13px', color: '#1890ff' }}>Baseline Model ({baselineSummary.label}):</AntText>
                                  <ul style={{ fontSize: '12px', color: '#666', marginTop: 6, paddingLeft: 18 }}>
                                    <li>{baselineSummary.architecture}</li>
                                    <li>{baselineSummary.parameters}</li>
                                  </ul>
                            </div>
                            <Divider style={{ margin: '12px 0' }} />
                                <div>
                                  <AntText strong style={{ fontSize: '13px', color: '#ff6b35' }}>Your Uploaded Model ({uploadedModelMeta?.name || "Your Model"}):</AntText>
                                  <ul style={{ fontSize: '12px', color: '#666', marginTop: 6, paddingLeft: 18 }}>
                                    {studentMetrics?.num_params && (
                                      <li>Parameters: {studentMetrics.num_params.toLocaleString?.() || studentMetrics.num_params}</li>
                                    )}
                                    {studentMetrics?.size_mb && (
                                      <li>Size: {studentMetrics.size_mb} MB (compressed)</li>
                                    )}
                                    {studentMetrics?.latency_ms && (
                                      <li>Latency: {studentMetrics.latency_ms} ms (faster)</li>
                                    )}
                                  </ul>
                            </div>
                          </div>
                            )
                          }]}
                        />
                      </Card>
                    </Col>

                      {/* Visualization Legend */}
                    <Col xs={24} lg={12}>
                      <Card style={{ marginBottom: 16, borderRadius: '12px' }}>
                        <Collapse
                          items={[{
                            key: '1',
                            label: <Title level={5} style={{ margin: 0 }}>📖 Visualization Legend</Title>,
                            children: (
                              <div style={{ fontSize: '12px', color: '#666', lineHeight: '1.8' }}>
                                <div style={{ marginBottom: 8 }}>
                                  <strong style={{ color: '#4fc3f7' }}>🔵 Blue nodes:</strong> Active layers processing data
                          </div>
                                <div style={{ marginBottom: 8 }}>
                                  <strong style={{ color: '#ff4444' }}>🔴 Red nodes:</strong> Pruned (removed) layers after compression
                          </div>
                                <div style={{ marginBottom: 8 }}>
                                  <strong style={{ color: '#888' }}>⚪ Gray lines:</strong> Active connections between layers
                          </div>
                                <div style={{ marginBottom: 0 }}>
                                  <strong style={{ color: '#ff0000' }}>🔴 Red lines:</strong> Pruned (removed) connections
                          </div>
                          </div>
                            )
                          }]}
                        />
                      </Card>
                    </Col>
                  </Row>
                </Col>
              </Row>
            )}

            {/* Key Differences Section - Below Comparison Sections */}
            {started && uploadedModelMeta && metricsSource && (
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24}>
                  <Card style={{ 
                    marginBottom: 16,
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #fff7e6 0%, #fff1d9 100%)',
                    border: '2px solid #faad14'
                  }}>
                    <div style={{ textAlign: 'center', marginBottom: 16 }}>
                      <Title level={4} style={{ color: '#faad14', marginBottom: 4 }}>
                        ⚖️ Key Differences
                          </Title>
                      <Paragraph style={{ color: '#666', fontSize: '12px' }}>
                        Baseline vs Your Uploaded Model
                      </Paragraph>
                    </div>
                    
                    <Row gutter={[12, 12]}>
                      {/* Accuracy Comparison */}
                      {metricsSource?.teacher_vs_student?.comparison?.accuracy && (
                        <Col xs={24} md={8}>
                          <Card size="small" style={{ 
                            background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)',
                            border: '1px solid #52c41a'
                          }}>
                            <Title level={5} style={{ color: '#52c41a', marginBottom: 8, textAlign: 'center', fontSize: '14px' }}>
                              📊 Accuracy
                              </Title>
                            <div style={{ textAlign: 'center', marginBottom: 4 }}>
                              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#333' }}>
                                Baseline: {metricsSource.teacher_vs_student.comparison.accuracy.teacher}
                                          </div>
                              <div style={{ fontSize: '13px', color: '#666', marginTop: 2 }}>
                                Your Model: {metricsSource.teacher_vs_student.comparison.accuracy.student}
                                        </div>
                              <div style={{ 
                                fontSize: '15px', 
                                fontWeight: 'bold', 
                                color: '#52c41a',
                                marginTop: 6
                              }}>
                                {metricsSource.teacher_vs_student.comparison.accuracy.difference}
                              </div>
                            </div>
                            <Paragraph style={{ fontSize: '11px', color: '#666', textAlign: 'center', margin: 0 }}>
                              {metricsSource.teacher_vs_student.comparison.accuracy.explanation}
                            </Paragraph>
                                    </Card>
                                  </Col>
                      )}

                      {/* Model Size Comparison */}
                      {metricsSource?.teacher_vs_student?.comparison?.model_size && (
                        <Col xs={24} md={8}>
                          <Card size="small" style={{ 
                            background: 'linear-gradient(135deg, #f6ffed 0%, #f0f9ff 100%)',
                            border: '1px solid #52c41a'
                          }}>
                            <Title level={5} style={{ color: '#52c41a', marginBottom: 8, textAlign: 'center', fontSize: '14px' }}>
                              💾 Model Size
                            </Title>
                            <div style={{ textAlign: 'center', marginBottom: 4 }}>
                              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#333' }}>
                                Baseline: {metricsSource.teacher_vs_student.comparison.model_size.teacher}
                                    </div>
                              <div style={{ fontSize: '13px', color: '#666', marginTop: 2 }}>
                                Your Model: {metricsSource.teacher_vs_student.comparison.model_size.student}
                                  </div>
                              <div style={{ 
                                fontSize: '15px', 
                                fontWeight: 'bold', 
                                color: '#52c41a',
                                marginTop: 6
                              }}>
                                {metricsSource.teacher_vs_student.comparison.model_size.difference}
                                    </div>
                                  </div>
                            <Paragraph style={{ fontSize: '11px', color: '#666', textAlign: 'center', margin: 0 }}>
                              {metricsSource.teacher_vs_student.comparison.model_size.explanation}
                            </Paragraph>
                          </Card>
                                </Col>
                      )}

                      {/* Inference Speed Comparison */}
                      {metricsSource?.teacher_vs_student?.comparison?.inference_speed && (
                        <Col xs={24} md={8}>
                          <Card size="small" style={{ 
                            background: 'linear-gradient(135deg, #fff2f0 0%, #fff1f0 100%)',
                            border: '1px solid #ff6b35'
                          }}>
                            <Title level={5} style={{ color: '#ff6b35', marginBottom: 8, textAlign: 'center', fontSize: '14px' }}>
                              ⚡ Inference Speed
                            </Title>
                            <div style={{ textAlign: 'center', marginBottom: 4 }}>
                              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#333' }}>
                                Baseline: {metricsSource.teacher_vs_student.comparison.inference_speed.teacher}
                                   </div>
                              <div style={{ fontSize: '13px', color: '#666', marginTop: 2 }}>
                                Your Model: {metricsSource.teacher_vs_student.comparison.inference_speed.student}
                                 </div>
                              <div style={{ 
                                fontSize: '15px', 
                                fontWeight: 'bold', 
                                color: '#52c41a',
                                marginTop: 6
                              }}>
                                {metricsSource.teacher_vs_student.comparison.inference_speed.difference}
                                   </div>
                                 </div>
                            <Paragraph style={{ fontSize: '11px', color: '#666', textAlign: 'center', margin: 0 }}>
                              {metricsSource.teacher_vs_student.comparison.inference_speed.explanation}
                            </Paragraph>
                          </Card>
                               </Col>
                      )}
                             </Row>
                         </Card>
                </Col>
              </Row>
            )}

            {/* Understanding the Differences - Step by Step (One at a time) - Below Key Differences */}
            {started && uploadedModelMeta && metricsSource && (
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24}>
                  <Card style={{ 
                    borderRadius: '12px',
                    background: step === 0 ? 'rgba(230, 247, 255, 0.5)' :
                                step === 1 ? 'rgba(230, 247, 255, 0.5)' :
                                step === 2 ? 'rgba(230, 247, 255, 0.5)' :
                                step === 3 ? 'rgba(255, 242, 240, 0.5)' :
                                step === 4 ? 'rgba(255, 242, 240, 0.5)' :
                                step === 5 ? 'rgba(246, 255, 237, 0.5)' :
                                'rgba(246, 255, 237, 0.5)',
                    border: '1px solid #d9d9d9'
                  }}>
                    <Title level={5} style={{ marginBottom: 8, fontSize: '14px' }}>
                      📋 Understanding the Differences - Step {step + 1}
                    </Title>
                    {step === 0 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Baseline Model (Left) - The "Teacher":</strong> The original, uncompressed reference model ({baselineSummary.label}). 
                        This is the "teacher" model that serves as the performance benchmark. It's larger, has more parameters ({baselineSummary.parameters}), 
                        and represents the model before any compression techniques were applied.
                      </Paragraph>
                    )}
                    {step === 1 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Processing Input:</strong> Both models are processing the same input data. The baseline model processes it through all its 
                        uncompressed layers, showing the full computational path.
                      </Paragraph>
                    )}
                    {step === 2 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Forward Pass:</strong> Data flows through all layers in both models. Notice how the baseline model has more connections 
                        and nodes compared to your uploaded model.
                      </Paragraph>
                    )}
                    {step === 3 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Knowledge Distillation:</strong> Your uploaded model learned from the baseline teacher model. This is where it captured 
                        the teacher's knowledge and confidence levels, not just correct answers.
                      </Paragraph>
                    )}
                    {step === 4 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Pruning:</strong> Your uploaded model ({uploadedModelMeta?.name || "Your Model"}) had {metricsSource?.pruning_analysis?.pruning_details?.pruning_ratio || '30%'} of unnecessary weights removed. 
                        Notice the red nodes/connections showing what was pruned. This makes it smaller and faster.
                      </Paragraph>
                    )}
                    {step === 5 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Fine-tuning:</strong> Your model is being fine-tuned after pruning to recover performance. The compressed model adjusts 
                        to its new, smaller structure.
                      </Paragraph>
                    )}
                    {step === 6 && (
                      <Paragraph style={{ color: '#333', fontSize: '13px', lineHeight: '1.6', marginBottom: 0 }}>
                        <strong>Key Takeaway:</strong> Compression techniques (KD + Pruning) reduced your model's size and improved inference speed 
                        while preserving accuracy. Compare the metrics above to see the improvements!
                      </Paragraph>
                    )}
                  </Card>
              </Col>
            </Row>
            )}
          </div>

          {/* Back to Training Button at Bottom */}
          <div style={{ textAlign: 'center', marginTop: '40px', marginBottom: '20px' }}>
            <Button 
              onClick={() => navigate('/training')}
              type="default"
              size="large"
              style={{ 
                padding: '0 40px',
                height: '48px',
                fontSize: '16px'
              }}
            >
              Back to Training
            </Button>
          </div>
        </Content>
      </Layout>
      <Footer />
    </>
  );
};

export default Visualization;
//gegege