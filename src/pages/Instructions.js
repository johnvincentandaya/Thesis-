import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Typography, Card } from 'antd';
import './Instructions.css'; // Link the new CSS file
import Footer from '../components/Footer';

const { Title, Paragraph } = Typography;

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
        <Title className="page-hero-title" style={{ fontWeight: 800 }}>How to Use the KD-Pruning Simulator</Title>
        <Paragraph className="instructions-subtitle page-hero-subtitle">
          This simulator allows you to explore <strong className="hero-accent-primary">Knowledge Distillation</strong> and <strong className="hero-accent-success">Model Pruning</strong> techniques interactively.
          Follow the steps below to start:
        </Paragraph>

        {/* Instructions Steps - four separate cards (no outer white card) */}
        <div className="instructions-steps-grid">
            <div className="instructions-step-card">
              <div className="step-badge">1</div>
              <div className="step-title">Models</div>
              <div className="step-desc">Go to the <Link to="/models">Models</Link> page and see the models descriptions.</div>
            </div>
            <div className="instructions-step-card">
              <div className="step-badge">2</div>
              <div className="step-title">Train Your Model</div>
              <div className="step-desc">Navigate to the <Link to="/training">Training</Link> page to choose and train a student model using KD and Pruning.</div>
            </div>
            <div className="instructions-step-card">
              <div className="step-badge">3</div>
              <div className="step-title">Visualize Results</div>
              <div className="step-desc">Explore the impact of KD & Pruning and check performance of the model on the <Link to="/visualization">Visualization</Link> page.</div>
            </div>
            <div className="instructions-step-card">
              <div className="step-badge">4</div>
              <div className="step-title">Assessment</div>
              <div className="step-desc">Take the assessment on the <Link to="/assessment">Assessment</Link> page.</div>
            </div>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Instructions;
