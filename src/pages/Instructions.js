import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Typography, Card, Alert, Row, Col, Divider } from 'antd';
import { 
  RocketOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  BarChartOutlined, 
  EyeOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  FileTextOutlined,
  ApiOutlined,
  ScissorOutlined,
  LineChartOutlined,
  SafetyOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import './Instructions.css';
import Footer from '../components/Footer';

const { Title, Paragraph, Text } = Typography;

function Instructions() {
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

      {/* Main Content */}
      <div className="instructions-container">

        {/* Quick Start Guide */}
        <Card className="quick-start-card" bordered={false}>
          <div className="quick-start-header">
            <Title level={2} className="section-title">
              <RocketOutlined className="section-icon" /> Quick Start Guide
            </Title>
            <Text className="section-subtitle">Get started in 4 simple steps</Text>
          </div>
          
          <Row gutter={[24, 24]} className="steps-row">
            <Col xs={24} sm={12} lg={6}>
              <div className="step-card step-1">
                <div className="step-number">1</div>
                <div className="step-icon-wrapper">
                  <FileTextOutlined className="step-icon" />
                </div>
                <Title level={4} className="step-title">Explore Models</Title>
                <Paragraph className="step-description">
                  Visit the <Link to="/models" className="step-link">Models</Link> page to see available baseline models and their performance metrics.
                </Paragraph>
              </div>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <div className="step-card step-2">
                <div className="step-number">2</div>
                <div className="step-icon-wrapper">
                  <UploadOutlined className="step-icon" />
                </div>
                <Title level={4} className="step-title">Select & Upload</Title>
                <Paragraph className="step-description">
                  Go to <Link to="/training" className="step-link">Training</Link>, choose a baseline model, then upload your custom model file.
                </Paragraph>
              </div>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <div className="step-card step-3">
                <div className="step-number">3</div>
                <div className="step-icon-wrapper">
                  <PlayCircleOutlined className="step-icon" />
                </div>
                <Title level={4} className="step-title">Start Training</Title>
                <Paragraph className="step-description">
                  Click "Start Training" to begin Knowledge Distillation and Pruning. Watch real-time progress and metrics.
                </Paragraph>
              </div>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <div className="step-card step-4">
                <div className="step-number">4</div>
                <div className="step-icon-wrapper">
                  <EyeOutlined className="step-icon" />
                </div>
                <Title level={4} className="step-title">Visualize Results</Title>
                <Paragraph className="step-description">
                  View side-by-side 3D visualizations and comparison metrics on the <Link to="/visualization" className="step-link">Visualization</Link> page.
                </Paragraph>
              </div>
            </Col>
          </Row>
        </Card>

        {/* Detailed Training Process */}
        <Card className="detail-card" bordered={false}>
          <div className="detail-header">
            <Title level={2} className="section-title">
              <ApiOutlined className="section-icon" /> Training Process Explained
            </Title>
            <Paragraph className="section-description">
              Understand what happens behind the scenes when you train your model
            </Paragraph>
          </div>

          <div className="process-timeline">
            <div className="timeline-item">
              <div className="timeline-marker marker-1">
                <FileTextOutlined />
              </div>
              <div className="timeline-content">
                <Title level={4} className="timeline-title">
                  <SafetyOutlined className="timeline-icon" /> File Validation
                </Title>
                <Paragraph className="timeline-description">
                  Your uploaded file is checked for:
                </Paragraph>
                <ul className="timeline-list">
                  <li><CheckCircleOutlined className="list-icon" /> File type: <code>.pt</code>, <code>.pth</code>, or <code>.bin</code></li>
                  <li><CheckCircleOutlined className="list-icon" /> File size: Maximum 500MB</li>
                  <li><CheckCircleOutlined className="list-icon" /> Not a baseline model (DistilBERT, ResNet-18, MobileNetV2, T5 Small)</li>
                </ul>
              </div>
            </div>

            <div className="timeline-connector"></div>

            <div className="timeline-item">
              <div className="timeline-marker marker-2">
                <ApiOutlined />
              </div>
              <div className="timeline-content">
                <Title level={4} className="timeline-title">
                  <ThunderboltOutlined className="timeline-icon" /> Knowledge Distillation
                </Title>
                <Paragraph className="timeline-description">
                  Your model learns from the selected baseline model. The baseline acts as a <strong>"teacher"</strong> transferring knowledge to your model (the <strong>"student"</strong>). This process runs for multiple epochs to ensure effective knowledge transfer.
                </Paragraph>
                <div className="info-box">
                  <Text className="info-text">
                    <strong>What you'll see:</strong> Real-time loss values, training progress, and phase indicators.
                  </Text>
                </div>
              </div>
            </div>

            <div className="timeline-connector"></div>

            <div className="timeline-item">
              <div className="timeline-marker marker-3">
                <ScissorOutlined />
              </div>
              <div className="timeline-content">
                <Title level={4} className="timeline-title">
                  <ScissorOutlined className="timeline-icon" /> Model Pruning
                </Title>
                <Paragraph className="timeline-description">
                  After Knowledge Distillation, the system removes less important weights (30% pruning) to reduce model size while maintaining performance. This makes your model smaller and faster.
                </Paragraph>
                <div className="info-box">
                  <Text className="info-text">
                    <strong>Result:</strong> Smaller file size, faster inference, with minimal accuracy loss.
                  </Text>
                </div>
              </div>
            </div>

            <div className="timeline-connector"></div>

            <div className="timeline-item">
              <div className="timeline-marker marker-4">
                <BarChartOutlined />
              </div>
              <div className="timeline-content">
                <Title level={4} className="timeline-title">
                  <LineChartOutlined className="timeline-icon" /> Metrics & Comparison
                </Title>
                <Paragraph className="timeline-description">
                  After training completes, you'll see detailed comparison metrics between your trained model and the baseline:
                </Paragraph>
                <Row gutter={[16, 16]} className="metrics-grid">
                  <Col xs={24} sm={12} md={8}>
                    <div className="metric-item">
                      <div className="metric-label">F1-Score</div>
                      <div className="metric-desc">Balanced performance measure</div>
                    </div>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <div className="metric-item">
                      <div className="metric-label">Accuracy</div>
                      <div className="metric-desc">Overall prediction correctness</div>
                    </div>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <div className="metric-item">
                      <div className="metric-label">Size Reduction</div>
                      <div className="metric-desc">How much smaller your model became</div>
                    </div>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <div className="metric-item">
                      <div className="metric-label">Inference Latency</div>
                      <div className="metric-desc">Speed improvement in milliseconds</div>
                    </div>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <div className="metric-item">
                      <div className="metric-label">Model Complexity</div>
                      <div className="metric-desc">Computational complexity comparison</div>
                    </div>
                  </Col>
                </Row>
              </div>
            </div>
          </div>
        </Card>

        {/* File Requirements */}
        <Row gutter={[24, 24]}>
          <Col xs={24} lg={12}>
            <Card className="requirements-card success-card" bordered={false}>
              <div className="card-header-success">
                <CheckCircleOutlined className="card-header-icon" />
                <Title level={3} className="card-title">Accepted Files</Title>
              </div>
              <Divider className="card-divider" />
              <div className="requirements-list">
                <div className="requirement-item success">
                  <CheckCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>File Types:</Text>
                    <Text className="requirement-detail"> <code>.pt</code>, <code>.pth</code>, or <code>.bin</code></Text>
                  </div>
                </div>
                <div className="requirement-item success">
                  <CheckCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>Maximum Size:</Text>
                    <Text className="requirement-detail"> 500MB</Text>
                  </div>
                </div>
                <div className="requirement-item success">
                  <CheckCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>Custom Models:</Text>
                    <Text className="requirement-detail"> Your own trained PyTorch models</Text>
                  </div>
                </div>
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card className="requirements-card error-card" bordered={false}>
              <div className="card-header-error">
                <CloseCircleOutlined className="card-header-icon" />
                <Title level={3} className="card-title">Blocked Files</Title>
              </div>
              <Divider className="card-divider" />
              <div className="requirements-list">
                <div className="requirement-item error">
                  <CloseCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>Baseline Models:</Text>
                    <Text className="requirement-detail"> DistilBERT, ResNet-18, MobileNetV2, T5 Small</Text>
                  </div>
                </div>
                <div className="requirement-item error">
                  <CloseCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>Wrong File Types:</Text>
                    <Text className="requirement-detail"> Images, spreadsheets, datasets, etc.</Text>
                  </div>
                </div>
                <div className="requirement-item error">
                  <CloseCircleOutlined className="requirement-icon" />
                  <div>
                    <Text strong>Oversized Files:</Text>
                    <Text className="requirement-detail"> Files larger than 500MB</Text>
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Visualization Guide */}
        <Card className="visualization-card" bordered={false}>
          <div className="visualization-header">
            <EyeOutlined className="section-icon-large" />
            <Title level={2} className="section-title">3D Visualization Guide</Title>
            <Paragraph className="section-description">
              Explore your models in stunning 3D after training completes
            </Paragraph>
          </div>

          <Alert
            type="info"
            message="Access Requirement"
            description="The Visualization page is only accessible after training is complete. Complete training first, then visit the Visualization page."
            showIcon
            className="visualization-alert"
          />

          <Row gutter={[24, 24]} className="visualization-features">
            <Col xs={24} md={12}>
              <div className="feature-box">
                <div className="feature-icon-wrapper feature-1">
                  <BarChartOutlined className="feature-icon" />
                </div>
                <Title level={4} className="feature-title">Baseline Model (Top)</Title>
                <Paragraph className="feature-description">
                  The 3D simulation at the top shows your selected baseline model. Each model has unique colors and structure to help you distinguish them.
                </Paragraph>
              </div>
            </Col>
            <Col xs={24} md={12}>
              <div className="feature-box">
                <div className="feature-icon-wrapper feature-2">
                  <ThunderboltOutlined className="feature-icon" />
                </div>
                <Title level={4} className="feature-title">Your Trained Model (Bottom)</Title>
                <Paragraph className="feature-description">
                  The bottom simulation shows your uploaded model after KD + Pruning. Red nodes indicate pruned (removed) connections, showing how the model was optimized.
                </Paragraph>
              </div>
            </Col>
            <Col xs={24} md={12}>
              <div className="feature-box">
                <div className="feature-icon-wrapper feature-3">
                  <EyeOutlined className="feature-icon" />
                </div>
                <Title level={4} className="feature-title">Interactive Controls</Title>
                <Paragraph className="feature-description">
                  Rotate, zoom, and explore both models in 3D. The cream background provides excellent contrast for viewing the neural network structures.
                </Paragraph>
              </div>
            </Col>
            <Col xs={24} md={12}>
              <div className="feature-box">
                <div className="feature-icon-wrapper feature-4">
                  <LineChartOutlined className="feature-icon" />
                </div>
                <Title level={4} className="feature-title">Visual Comparison</Title>
                <Paragraph className="feature-description">
                  Side-by-side comparison helps you understand how Knowledge Distillation and Pruning transformed your model's architecture and performance.
                </Paragraph>
              </div>
            </Col>
          </Row>
        </Card>

      </div>
      <Footer />
    </>
  );
}

export default Instructions;
