import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

function Assessment() {
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
        <Container className="mt-5">
          <h1 className="text-center">Assessment Page</h1>
          <p className="text-center">This is a placeholder for the Assessment page content.</p>
        </Container>
      </>
    );
}

export default Assessment;