import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <Container>
        <Row className="py-4">
          <Col md={6} className="text-center text-md-start">
            <h5 className="footer-title">KD-Pruning Simulator</h5>
            <p className="footer-description">
              An interactive educational platform for learning about Knowledge Distillation and Model Pruning techniques.
            </p>
          </Col>
          <Col md={6} className="text-center text-md-end">
            <div className="footer-links">
              <Link to="/" className="footer-link">Home</Link>
              <Link to="/instructions" className="footer-link">Instructions</Link>
              <Link to="/models" className="footer-link">Models</Link>
              <Link to="/training" className="footer-link">Training</Link>
              <Link to="/visualization" className="footer-link">Visualization</Link>
              <Link to="/assessment" className="footer-link">Assessment</Link>
            </div>
            <p className="footer-copyright">
              © 2024 KD-Pruning Simulator. Educational tool for AI model optimization.
            </p>
          </Col>
        </Row>
      </Container>
    </footer>
  );
};

export default Footer;
