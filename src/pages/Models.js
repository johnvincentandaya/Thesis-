import React from "react";
import { Container, Row, Col, Card, Navbar, Nav, Table } from "react-bootstrap";
import { Link, useNavigate } from "react-router-dom";
import "./Models.css";

const Models = () => {
    const navigate = useNavigate();

    const models = [
        {
            name: "distillBert",
            description:
                "DistillBERT is a smaller, faster, and lighter version of BERT, designed for natural language processing tasks like text classification and question answering.",
            metrics: {
                f1Score: "92%",
                accuracy: "90%",
                sizeReduction: "40%",
                latency: "50ms",
                complexity: "Medium",
            },
        },
        {
            name: "T5-small",
            description:
                "T5-small is a smaller version of the T5 (Text-to-Text Transfer Transformer) model, capable of performing a wide range of NLP tasks by converting them into a text-to-text format.",
            metrics: {
                f1Score: "88%",
                accuracy: "85%",
                sizeReduction: "35%",
                latency: "70ms",
                complexity: "High",
            },
        },
        {
            name: "MobileNetV2",
            description:
                "MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification and object detection on mobile and embedded devices.",
            metrics: {
                f1Score: "85%",
                accuracy: "83%",
                sizeReduction: "50%",
                latency: "30ms",
                complexity: "Low",
            },
        },
        {
            name: "ResNet-18",
            description:
                "ResNet-18 is a deep residual network with 18 layers, known for its ability to train very deep networks by using skip connections to avoid vanishing gradients.",
            metrics: {
                f1Score: "90%",
                accuracy: "88%",
                sizeReduction: "25%",
                latency: "60ms",
                complexity: "High",
            },
        },
    ];

    const handleCardClick = (modelName) => {
        navigate("/training", { state: { selectedModel: modelName } }); // Pass the selected model to the training page
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
            <Container className="mt-5">
                <h1 className="text-center mb-4">Model Descriptions</h1>
                <Row>
                    {models.map((model, index) => (
                        <Col key={index} md={6} className="mb-4">
                            <Card
                                className="w-100"
                                onClick={() => handleCardClick(model.name)} // Pass the model name on click
                                style={{ cursor: "pointer" }}
                            >
                                <Card.Body>
                                    <Card.Title>{model.name}</Card.Title>
                                    <Card.Text>{model.description}</Card.Text>
                                    <Table striped bordered hover size="sm">
                                        <tbody>
                                            <tr>
                                                <td><strong>F1-Score</strong></td>
                                                <td>{model.metrics.f1Score}</td>
                                            </tr>
                                            <tr>
                                                <td><strong>Accuracy</strong></td>
                                                <td>{model.metrics.accuracy}</td>
                                            </tr>
                                            <tr>
                                                <td><strong>Model Size Reduction</strong></td>
                                                <td>{model.metrics.sizeReduction}</td>
                                            </tr>
                                            <tr>
                                                <td><strong>Inference Latency</strong></td>
                                                <td>{model.metrics.latency}</td>
                                            </tr>
                                            <tr>
                                                <td><strong>Model Complexity</strong></td>
                                                <td>{model.metrics.complexity}</td>
                                            </tr>
                                        </tbody>
                                    </Table>
                                </Card.Body>
                            </Card>
                        </Col>
                    ))}
                </Row>
            </Container>
        </>
    );
};

export default Models;
