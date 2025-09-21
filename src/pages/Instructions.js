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
        <Title>How to Use the KD-Pruning Simulator</Title>
        <Paragraph>
          This simulator allows you to explore <b>Knowledge Distillation</b> and <b>Model Pruning</b> techniques interactively.
          Follow the steps below to start:
        </Paragraph>

        {/* Instructions Card */}
        <Card className="instructions-card">
          <Paragraph>
            <b> 1. Models:</b> Go to the <Link to="/models">Models</Link> page and see the models descriptions. <br /><br />
            <b> 2. Train Your Model:</b> Navigate to the <Link to="/training">Training</Link> page to choose and train a student model using KD and Pruning. <br /><br />
            <b> 3. Visualize Results:</b> Explore the impact of KD & Pruning and Check perfomance of the model on the <Link to="/visualization">Visualization</Link> page.<br /><br />
            <b> 4. Assessment:</b> Take the assessment on the <Link to="/assessment">Assessment</Link> page.
          </Paragraph>
        </Card>
      </div>
      <Footer />
    </>
  );
}

export default Instructions;
