import { Navbar, Nav, Container, Row, Col, Card, Button, Accordion } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './Home.css';
import Footer from '../components/Footer';

function Home() {
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
      
      <Container className="home-container">
        <div className="text-center mb-5">
          <h1 className="display-4 mb-3" style={{ fontSize: '4rem', fontWeight: 'bold', color: '#1890ff', marginBottom: '2rem' }}>
            🚀 Welcome to KD & Pruning Simulator
          </h1>
          <p className="lead" style={{ fontSize: '1.4rem', fontWeight: '400', color: '#666', lineHeight: '1.6' }}>
            An interactive educational tool to understand <strong style={{ color: '#1890ff' }}>Knowledge Distillation</strong> and <strong style={{ color: '#52c41a' }}>Model Pruning</strong> techniques for neural network compression.
          </p>
        </div>

        {/* Main Concepts Section */}
        <Row className="mb-5">
          <Col lg={6} className="mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Body className="p-4">
                <Card.Title className="h4 text-primary mb-3">
                  <i className="fas fa-graduation-cap me-2"></i>
                  Knowledge Distillation (KD)
                </Card.Title>
                <Card.Text className="mb-3">
                  <strong>Knowledge Distillation</strong> is a model compression technique where a large, complex model (called the "teacher") transfers its learned knowledge to a smaller, more efficient model (called the "student").
                </Card.Text>
                <Card.Text className="mb-3">
                  <strong>How it works:</strong> The student model learns not only from the ground truth labels but also from the "soft" outputs (probabilities) of the teacher model, which contain richer information than hard labels.
                </Card.Text>
                <Card.Text>
                  <strong>Benefits:</strong> The student model can achieve similar or even better performance than the teacher while being much smaller and faster.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
          
          <Col lg={6} className="mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Body className="p-4">
                <Card.Title className="h4 text-success mb-3">
                  <i className="fas fa-cut me-2"></i>
                  Model Pruning
                </Card.Title>
                <Card.Text className="mb-3">
                  <strong>Model Pruning</strong> is a technique that removes less important connections (weights) from a neural network, effectively making it sparser while maintaining most of its accuracy.
                </Card.Text>
                <Card.Text className="mb-3">
                  <strong>How works:</strong> The algorithm identifies and removes weights that contribute little to the model's performance, often setting them to zero or completely removing them.
                </Card.Text>
                <Card.Text>
                  <strong>Benefits:</strong> Reduces model size, speeds up inference, and can even improve generalization by reducing overfitting.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Uses and Applications Section */}
        <Card className="mb-5 shadow-sm">
          <Card.Body className="p-4">
            <Card.Title className="h3 text-center mb-4">
              <i className="fas fa-rocket me-2"></i>
              Uses of Model Compression
            </Card.Title>
            <Row>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-mobile-alt fa-2x text-primary mb-2"></i>
                  <h5>Mobile & Edge Devices</h5>
                  <p className="text-muted">Deploy AI models on smartphones, IoT devices, and embedded systems with limited computational resources.</p>
                </div>
              </Col>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-tachometer-alt fa-2x text-success mb-2"></i>
                  <h5>Real-time Applications</h5>
                  <p className="text-muted">Enable faster inference for applications requiring real-time responses like autonomous vehicles and robotics.</p>
                </div>
              </Col>
              <Col md={4} className="mb-3">
                <div className="text-center">
                  <i className="fas fa-server fa-2x text-warning mb-2"></i>
                  <h5>Cost Reduction</h5>
                  <p className="text-muted">Reduce computational costs and energy consumption in cloud deployments and data centers.</p>
                </div>
              </Col>
            </Row>
          </Card.Body>
        </Card>

        {/* Detailed Explanations Accordion */}
        <Card className="mb-5 shadow-sm">
          <Card.Body className="p-4">
            <Card.Title className="h3 text-center mb-4">
              <i className="fas fa-info-circle me-2"></i>
              Learn More About These Techniques
            </Card.Title>
            <Accordion>
              <Accordion.Item eventKey="0">
                <Accordion.Header>
                  <strong>Knowledge Distillation - Detailed Process</strong>
                </Accordion.Header>
                <Accordion.Body>
                  <div className="row">
                    <div className="col-md-6">
                      <h6>Step 1: Teacher Training</h6>
                      <p>A large, complex model (teacher) is trained on the target dataset until it achieves high accuracy.</p>
                      
                      <h6>Step 2: Student Architecture</h6>
                      <p>A smaller, simpler model (student) is designed with fewer parameters and layers.</p>
                    </div>
                    <div className="col-md-6">
                      <h6>Step 3: Knowledge Transfer</h6>
                      <p>The student learns from both the ground truth labels and the teacher's soft predictions (logits).</p>
                      
                      <h6>Step 4: Distillation Loss</h6>
                      <p>The training uses a combination of classification loss and distillation loss to transfer knowledge effectively.</p>
                    </div>
                  </div>
                </Accordion.Body>
              </Accordion.Item>
              
              <Accordion.Item eventKey="1">
                <Accordion.Header>
                  <strong>Model Pruning - Detailed Process</strong>
                </Accordion.Header>
                <Accordion.Body>
                  <div className="row">
                    <div className="col-md-6">
                      <h6>Step 1: Model Training</h6>
                      <p>Train the model normally until it achieves good performance on the target task.</p>
                      
                      <h6>Step 2: Importance Assessment</h6>
                      <p>Evaluate the importance of each weight using criteria like magnitude, gradient, or sensitivity analysis.</p>
                    </div>
                    <div className="col-md-6">
                      <h6>Step 3: Weight Removal</h6>
                      <p>Remove or zero out the least important weights based on the assessment criteria.</p>
                      
                      <h6>Step 4: Fine-tuning</h6>
                      <p>Fine-tune the pruned model to recover any lost accuracy from the pruning process.</p>
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
          </Card.Body>
        </Card>

        {/* Get Started Section */}
        <div className="text-center">
          <Card className="shadow-sm">
            <Card.Body className="p-5">
              <h2 className="mb-3">Ready to Explore?</h2>
              <p className="lead mb-4">
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
            </Card.Body>
          </Card>
        </div>
      </Container>
      <Footer />
    </>
  );
}

export default Home;