import { Navbar, Nav, Container, Row, Col, Button, Accordion, Card as BootstrapCard } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Typography, Tag } from 'antd';
import { 
  RocketOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  EyeOutlined
} from '@ant-design/icons';
import './Home.css';
import './Instructions.css';
import Footer from '../components/Footer';

const { Title, Paragraph, Text } = Typography;

function Home() {
  const [visualizationUnlocked, setVisualizationUnlocked] = useState(() => {
    return localStorage.getItem('visualization_unlocked') === 'true';
  });

  useEffect(() => {
    const handleStorageChange = () => {
      setVisualizationUnlocked(localStorage.getItem('visualization_unlocked') === 'true');
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);
  return (
    <>
      <Navbar bg="black" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand as={Link} to="/">Knowledge Distillation-Pruning Simulator</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              <Nav.Link as={Link} to="/">Home</Nav.Link>
              <Nav.Link as={Link} to="/instructions">Instructions</Nav.Link>
              <Nav.Link as={Link} to="/models">Models</Nav.Link>
              <Nav.Link as={Link} to="/training">Training</Nav.Link>
              <Nav.Link as={Link} to="/visualization" disabled={!visualizationUnlocked} style={{ pointerEvents: visualizationUnlocked ? 'auto' : 'none', opacity: visualizationUnlocked ? 1 : 0.5 }}>Visualization</Nav.Link>
              <Nav.Link as={Link} to="/assessment">Assessment</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      
      <div className="instructions-hero">
        <div className="instructions-hero-content">
          <div className="hero-icon-wrapper">
            <RocketOutlined className="hero-icon" />
          </div>
          <Title level={1} className="hero-title">
            Welcome to Knowledge Distillation & Pruning Simulator
          </Title>
          <Paragraph className="hero-subtitle">
            Transform your neural network models through <Text strong style={{ fontSize: '1.5rem', color: '#bae0ff' }}>Knowledge Distillation</Text> and <Text strong style={{ fontSize: '1.5rem', color: '#91caff' }}>Model Pruning</Text>. 
            Create smaller, faster, and more efficient models in just a few simple steps.
          </Paragraph>
          <div className="hero-badges">
            <Tag color="blue" className="hero-badge">
              <ThunderboltOutlined /> Fast Training
            </Tag>
            <Tag color="green" className="hero-badge">
              <LineChartOutlined /> Real Metrics
            </Tag>
            <Tag color="purple" className="hero-badge">
              <EyeOutlined /> 3D Visualization
            </Tag>
          </div>
        </div>
      </div>

      <Container fluid className="home-container">
        <Row className="mb-5">
          <Col lg={6} className="mb-4">
            <BootstrapCard className="h-100 shadow-sm home-wide-card">
              <BootstrapCard.Body className="p-4">
                <BootstrapCard.Title className="h4 text-primary mb-3 home-card-title">
                  <i className="fas fa-graduation-cap me-2"></i>
                  Knowledge Distillation
                </BootstrapCard.Title>
                <BootstrapCard.Text className="mb-3">
                  <strong>Knowledge Distillation</strong> is a model compression technique where a large, complex model (called the "teacher") transfers its learned knowledge to a smaller, more efficient model (called the "student").
                </BootstrapCard.Text>
                <BootstrapCard.Text className="mb-3">
                  <strong>How it works:</strong> The student model learns not only from the ground truth labels but also from the "soft" outputs (probabilities) of the teacher model, which contain richer information than hard labels.
                </BootstrapCard.Text>
                <BootstrapCard.Text>
                  <strong>Benefits:</strong> The student model can achieve similar or even better performance than the teacher while being much smaller and faster.
                </BootstrapCard.Text>
              </BootstrapCard.Body>
            </BootstrapCard>
          </Col>
          
          <Col lg={6} className="mb-4">
            <BootstrapCard className="h-100 shadow-sm home-wide-card">
              <BootstrapCard.Body className="p-4">
                <BootstrapCard.Title className="h4 text-success mb-3 home-card-title">
                  <i className="fas fa-cut me-2"></i>
                  Model Pruning
                </BootstrapCard.Title>
                <BootstrapCard.Text className="mb-3">
                  <strong>Model Pruning</strong> is a technique that removes less important connections (weights) from a neural network, effectively making it sparser while maintaining most of its accuracy.
                </BootstrapCard.Text>
                <BootstrapCard.Text className="mb-3">
                  <strong>How it works:</strong> The algorithm identifies and removes weights that contribute little to the model's performance, often setting them to zero or completely removing them.
                </BootstrapCard.Text>
                <BootstrapCard.Text>
                  <strong>Benefits:</strong> Reduces model size, speeds up inference, and can even improve generalization by reducing overfitting.
                </BootstrapCard.Text>
              </BootstrapCard.Body>
            </BootstrapCard>
          </Col>
        </Row>

        <BootstrapCard className="mb-5 shadow-sm">
          <BootstrapCard.Body className="p-4">
            <BootstrapCard.Title className="h3 text-center mb-4 home-section-title">
              <i className="fas fa-rocket me-2"></i>
              Uses of Model Compression
            </BootstrapCard.Title>
            <Row>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-mobile-alt fa-2x text-primary mb-2"></i>
                  <h5 className="home-use-title">Mobile & Edge Devices</h5>
                  <p className="text-muted">Deploy AI models on smartphones, IoT devices, and embedded systems with limited computational resources.</p>
                </div>
              </Col>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-tachometer-alt fa-2x text-success mb-2"></i>
                  <h5 className="home-use-title">Real-time Applications</h5>
                  <p className="text-muted">Enable faster inference for applications requiring real-time responses like autonomous vehicles and robotics.</p>
                </div>
              </Col>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-server fa-2x text-warning mb-2"></i>
                  <h5 className="home-use-title">Cost Reduction</h5>
                  <p className="text-muted">Reduce computational costs and energy consumption in cloud deployments and data centers.</p>
                </div>
              </Col>
            </Row>
          </BootstrapCard.Body>
        </BootstrapCard>

        <BootstrapCard className="mb-5 shadow-sm">
          <BootstrapCard.Body className="p-4">
            <BootstrapCard.Title className="h3 text-center mb-4 home-section-title">
              <i className="fas fa-info-circle me-2"></i>
              Learn More About These Techniques
            </BootstrapCard.Title>
            <Accordion>
              <Accordion.Item eventKey="0">
                <Accordion.Header>
                  <strong>Knowledge Distillation - Detailed Process</strong>
                </Accordion.Header>
                <Accordion.Body>
                  <div className="home-steps">
                    <div className="home-step">
                      <div className="home-step-title">Step 1: Teacher Training</div>
                      <div className="home-step-desc">A large, complex model (teacher) is trained on the target dataset until it achieves high accuracy.</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 2: Student Architecture</div>
                      <div className="home-step-desc">A smaller, simpler model (student) is designed with fewer parameters and layers.</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 3: Knowledge Transfer</div>
                      <div className="home-step-desc">The student learns from both the ground truth labels and the teacher's soft predictions (logits).</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 4: Distillation Loss</div>
                      <div className="home-step-desc">The training uses a combination of classification loss and distillation loss to transfer knowledge effectively.</div>
                    </div>
                  </div>
                </Accordion.Body>
              </Accordion.Item>
              
              <Accordion.Item eventKey="1">
                <Accordion.Header>
                  <strong>Model Pruning - Detailed Process</strong>
                </Accordion.Header>
                <Accordion.Body>
                  <div className="home-steps">
                    <div className="home-step">
                      <div className="home-step-title">Step 1: Model Training</div>
                      <div className="home-step-desc">Train the model normally until it achieves good performance on the target task.</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 2: Importance Assessment</div>
                      <div className="home-step-desc">Evaluate the importance of each weight using criteria like magnitude, gradient, or sensitivity analysis.</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 3: Weight Removal</div>
                      <div className="home-step-desc">Remove or zero out the least important weights based on the assessment criteria.</div>
                    </div>
                    <div className="home-step">
                      <div className="home-step-title">Step 4: Fine-tuning</div>
                      <div className="home-step-desc">Fine-tune the pruned model to recover any lost accuracy from the pruning process.</div>
                    </div>
                  </div>
                </Accordion.Body>
              </Accordion.Item>
              
              <Accordion.Item eventKey="2">
                <Accordion.Header>
                  <strong>Why Model Compression Matters</strong>
                </Accordion.Header>
                <Accordion.Body>
                  <div className="row">
                    <div className="col-md-4">
                      <h6>Efficiency</h6>
                      <p>Faster inference times enable real-time applications and better user experience.</p>
                    </div>
                    <div className="col-md-4">
                      <h6>Cost Savings</h6>
                      <p>Reduced computational requirements lead to lower deployment and operational costs.</p>
                    </div>
                    <div className="col-md-4">
                      <h6>Sustainability</h6>
                      <p>Lower energy consumption contributes to more environmentally friendly AI systems.</p>
                    </div>
                  </div>
                </Accordion.Body>
              </Accordion.Item>
            </Accordion>
          </BootstrapCard.Body>
        </BootstrapCard>

        <div className="text-center">
          <BootstrapCard className="shadow-sm home-cta-card">
            <BootstrapCard.Body className="p-5">
              <h2 className="mb-3 home-cta-title">Ready to Explore?</h2>
              <p className="lead mb-4 home-cta-subtitle">
                Start your journey by learning about the available models and then experience the compression process through interactive training and visualization.
              </p>
              <div className="d-flex justify-content-center gap-3 flex-wrap">
                <Button as={Link} to="/models" variant="primary" size="lg">
                  <i className="fas fa-cube me-2"></i>
                  Explore Models
                </Button>
                <Button as={Link} to="/instructions" variant="outline-primary" size="lg">
                  <i className="fas fa-play me-2"></i>
                  Get Started
                </Button>
              </div>
            </BootstrapCard.Body>
          </BootstrapCard>
        </div>
      </Container>
      <Footer />
    </>
  );
}

export default Home;