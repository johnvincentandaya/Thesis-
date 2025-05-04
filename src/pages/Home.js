import { Navbar, Nav, Container, Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './Home.css';

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
        <h1>Welcome to KD & Pruning Simulator</h1>
        <p>
          Explore how <strong>Knowledge Distillation</strong> and <strong>Model Pruning</strong> work through an interactive simulation.
        </p>
        <Row className="home-card-row">
          <Col>
            <Card className="home-card">
              <Card.Body>
                <Card.Title className="home-card-title">Knowledge Distillation</Card.Title>
                <Card.Text className="home-card-text">
                  <strong>Knowledge Distillation (KD)</strong> is a technique where a large, complex model (teacher) transfers its learned knowledge to a smaller, more efficient model (student).
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
          <Col>
            <Card className="home-card">
              <Card.Body>
                <Card.Title className="home-card-title">Pruning</Card.Title>
                <Card.Text className="home-card-text">
                  <strong>Model Pruning</strong> removes less important connections from a neural network, reducing its size while maintaining most of its accuracy.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <Button className="btn btn-dark btn-lg home-get-started-btn">
          <Link to="/instructions" className="home-get-started-link">Get Started</Link>
        </Button>
      </Container>
    </>
  );
}

export default Home;