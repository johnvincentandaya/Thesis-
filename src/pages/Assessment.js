import React, { useState, useEffect } from "react";
import { Navbar, Nav, Container, Button, Form, Alert, ProgressBar, Card, Badge } from "react-bootstrap";
import { Link } from "react-router-dom";
import {
  CheckCircleFill,
  XCircleFill,
  Trophy,
  Clock,
  Award,
  Lightbulb,
  Book
} from "react-bootstrap-icons";
import { BsCheckLg, BsX } from "react-icons/bs";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Assessment.css";

function Assessment() {
  const [answers, setAnswers] = useState({});
  const [score, setScore] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [timeSpent, setTimeSpent] = useState(0);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [showExplanation, setShowExplanation] = useState({});

  // Quiz questions and answers with explanations
  const questions = [
    {
      question: "What is the key difference between Knowledge Distillation and Pruning in model compression?",
      options: [
        "Knowledge Distillation reduces the model size by removing redundant nodes, while Pruning transfers knowledge from a larger model to a smaller one.",
        "Knowledge Distillation transfers knowledge from a larger model (teacher) to a smaller model (student), while Pruning removes unnecessary parts of the model.",
        "Pruning reduces the model size by transferring knowledge, while Knowledge Distillation removes unnecessary layers.",
        "Pruning transfers knowledge, while Knowledge Distillation compresses the model by decreasing layer size.",
      ],
      correctAnswer: 1,
      explanation: "Knowledge Distillation focuses on transferring the 'knowledge' (soft predictions, confidence levels) from a larger teacher model to a smaller student model, while Pruning removes unnecessary weights and connections to reduce model size and complexity.",
      category: "Knowledge Distillation"
    },
    {
      question: "What role do student and teacher models play in Knowledge Distillation?",
      options: [
        "The teacher model is a simpler model that teaches the student.",
        "The student model learns from the teacher's predictions and mimics its behavior, improving generalization.",
        "The teacher model is used to prune the student model.",
        "The teacher model is not used in Knowledge Distillation.",
      ],
      correctAnswer: 1,
      explanation: "The teacher model provides 'soft' predictions (probabilities) that contain more information than hard labels, helping the student model learn not just what to predict, but also how confident the teacher is about each prediction.",
      category: "Knowledge Distillation"
    },
    {
      question: "How does Pruning affect model structure and performance, and how is this reflected in our visualizations?",
      options: [
        "Pruning decreases accuracy but reduces model size, with pruned components shown as highlighted nodes in the visualization.",
        "Pruning improves model performance but increases complexity, with nodes removed in the visualization.",
        "Pruning removes redundant components, and pruned nodes are displayed as red in the visualization.",
        "Pruning has no impact on performance or structure.",
      ],
      correctAnswer: 2,
      explanation: "Pruning removes redundant or less important connections and nodes, typically resulting in a small accuracy trade-off for significant size and speed improvements. In visualizations, pruned components are often shown in different colors (like red) to indicate their inactive state.",
      category: "Pruning"
    },
    {
      question: "If a model shows a 2.5× compression ratio and a 1.8% accuracy drop, would you consider it a successful compression? Why or why not?",
      options: [
        "Yes, because the compression ratio is high and the accuracy drop is small.",
        "No, because the accuracy drop is too high relative to the compression ratio.",
        "Yes, because the compression is more important than accuracy.",
        "No, because the compression ratio is low.",
      ],
      correctAnswer: 0,
      explanation: "A 2.5× compression ratio (60% size reduction) with only a 1.8% accuracy drop represents an excellent trade-off. The model becomes significantly smaller and faster while maintaining most of its performance, making it suitable for deployment on resource-constrained devices.",
      category: "Compression Analysis"
    },
    {
      question: "How would you interpret a scenario where the latency improved, but the F1-score dropped significantly after pruning?",
      options: [
        "The model is performing better because latency is improved, and F1-score doesn't matter.",
        "The model's efficiency increased, but the drop in F1-score indicates a loss in accuracy.",
        "The F1-score drop suggests the model is performing worse overall, despite latency improvement.",
        "This indicates a successful compression process.",
      ],
      correctAnswer: 2,
      explanation: "While latency improvement is beneficial, a significant F1-score drop indicates that the model's overall performance has degraded. This suggests that too much pruning was applied, removing important connections that were necessary for accurate predictions.",
      category: "Performance Analysis"
    },
    {
      question: "What does the 'Complexity Metrics' section tell you about the model, and why is it important?",
      options: [
        "It shows the model's computational load and how hard it is to deploy in real-world environments.",
        "It indicates the model's size and how many layers are present.",
        "It measures how fast the model runs on GPUs.",
        "It only measures the accuracy of the model.",
      ],
      correctAnswer: 0,
      explanation: "Complexity metrics (like FLOPs, memory usage, inference time) help understand the computational requirements and deployment feasibility of the model. This is crucial for real-world applications where resources are limited.",
      category: "Model Complexity"
    },
  ];

  const trueFalseQuestions = [
    {
      question: "Knowledge Distillation is a technique where a smaller model (student) learns from a larger, pre-trained model (teacher).",
      correctAnswer: true,
      explanation: "True! Knowledge Distillation involves training a smaller student model to mimic the behavior of a larger, more complex teacher model, transferring not just the predictions but also the confidence levels and decision-making patterns.",
      category: "Knowledge Distillation"
    },
    {
      question: "Pruning removes unnecessary parts of a model to reduce its size and computational complexity.",
      correctAnswer: true,
      explanation: "True! Pruning identifies and removes weights, connections, or entire neurons that contribute little to the model's performance, resulting in a smaller, faster model.",
      category: "Pruning"
    },
    {
      question: "Pruning does not affect the accuracy of the model since it only focuses on reducing latency.",
      correctAnswer: false,
      explanation: "False! Pruning typically results in a small accuracy trade-off. While the goal is to minimize this impact, removing connections can affect the model's ability to make accurate predictions.",
      category: "Pruning"
    },
    {
      question: "The teacher model in Knowledge Distillation is responsible for generating soft predictions that guide the student model's learning process.",
      correctAnswer: true,
      explanation: "True! The teacher model provides 'soft' predictions (probability distributions) rather than hard labels, giving the student model richer information about the teacher's confidence and decision-making process.",
      category: "Knowledge Distillation"
    },
    {
      question: "Pruning improves accuracy by increasing the model's number of parameters.",
      correctAnswer: false,
      explanation: "False! Pruning reduces the number of parameters by removing connections. While this can sometimes improve generalization (reducing overfitting), it typically results in a small accuracy decrease.",
      category: "Pruning"
    },
    {
      question: "A model's complexity metrics give an insight into how computationally expensive and resource-intensive the model is.",
      correctAnswer: true,
      explanation: "True! Complexity metrics like FLOPs (floating point operations), memory usage, and inference time help assess the computational requirements and deployment feasibility of the model.",
      category: "Model Complexity"
    },
  ];

  // Timer effect
  useEffect(() => {
    if (!isSubmitted) {
      const timer = setInterval(() => {
        setTimeSpent(prev => prev + 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [isSubmitted]);

  // Handle answer selection
  const handleAnswerChange = (questionIndex, selectedOption) => {
    setAnswers((prev) => ({
      ...prev,
      [questionIndex]: selectedOption,
    }));
  };

  // Toggle explanation visibility
  const toggleExplanation = (questionIndex) => {
    setShowExplanation(prev => ({
      ...prev,
      [questionIndex]: !prev[questionIndex]
    }));
  };

  // Calculate score and show results
  const calculateScore = () => {
    let totalScore = 0;
    let correctAnswers = [];

    // Check multiple-choice questions
    questions.forEach((q, index) => {
      if (answers[index] === q.correctAnswer) {
        totalScore++;
        correctAnswers.push(index);
      }
    });

    // Check true/false questions
    trueFalseQuestions.forEach((q, index) => {
      const tfIndex = questions.length + index;
      if (answers[tfIndex] === q.correctAnswer) {
        totalScore++;
        correctAnswers.push(tfIndex);
      }
    });

    setScore(totalScore);
    setShowResults(true);
    setIsSubmitted(true);
  };

  // Reset quiz
  const resetQuiz = () => {
    setAnswers({});
    setScore(null);
    setShowResults(false);
    setCurrentQuestion(0);
    setTimeSpent(0);
    setIsSubmitted(false);
    setShowExplanation({});
  };

  // Get performance level
  const getPerformanceLevel = () => {
    const totalQuestions = questions.length + trueFalseQuestions.length;
    const percentage = (score / totalQuestions) * 100;
    
    if (percentage >= 90) return { level: "Excellent", color: "success", icon: Trophy };
    if (percentage >= 80) return { level: "Good", color: "info", icon: Award };
    if (percentage >= 70) return { level: "Fair", color: "warning", icon: Lightbulb };
    return { level: "Needs Improvement", color: "danger", icon: Book };
  };

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const totalQuestions = questions.length + trueFalseQuestions.length;
  const answeredQuestions = Object.keys(answers).length;
  const progressPercentage = (answeredQuestions / totalQuestions) * 100;

  // Returns true if all questions (MCQ and TF) have an answer
  const allQuestionsAnswered = () => {
    for (let i = 0; i < questions.length; i++) {
      if (typeof answers[i] === 'undefined') return false;
    }
    for (let i = 0; i < trueFalseQuestions.length; i++) {
      if (typeof answers[questions.length + i] === 'undefined') return false;
    }
    return true;
  };

  return (
    <>
      {/* Navbar */}
      <Navbar bg="black" variant="dark" expand="lg" className="shadow-sm">
        <Container>
          <Navbar.Brand as={Link} to="/" className="fw-bold">
            <Award className="me-2" />
            KD-Pruning Simulator
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              <Nav.Link as={Link} to="/">Home</Nav.Link>
              <Nav.Link as={Link} to="/instructions">Instructions</Nav.Link>
              <Nav.Link as={Link} to="/models">Models</Nav.Link>
              <Nav.Link as={Link} to="/training">Training</Nav.Link>
              <Nav.Link as={Link} to="/visualization">Visualization</Nav.Link>
              <Nav.Link as={Link} to="/assessment" className="active fw-bold">Assessment</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {/* Main Content */}
      <Container className="mt-4 mb-5">
        {/* Header */}
        <div className="text-center mb-5">
          <h1 className="display-4 fw-bold text-primary mb-3">
            <Book className="me-3" />
            Knowledge Assessment
          </h1>
          <p className="lead text-muted">
            Test your understanding of Knowledge Distillation and Model Pruning concepts
          </p>
        </div>

        {/* Progress and Timer */}
        {!isSubmitted && (
          <Card className="mb-4 shadow-sm">
            <Card.Body>
              <div className="row align-items-center">
                <div className="col-md-6">
                  <div className="d-flex align-items-center mb-2">
                    <Clock className="me-2 text-primary" />
                    <span className="fw-semibold">Time: {formatTime(timeSpent)}</span>
                  </div>
                  <ProgressBar 
                    now={progressPercentage} 
                    variant="primary" 
                    className="mb-2"
                    style={{ height: '8px' }}
                  />
                  <small className="text-muted">
                    {answeredQuestions} of {totalQuestions} questions answered
                  </small>
                </div>
                <div className="col-md-6 text-md-end">
                  <Badge bg="info" className="fs-6 px-3 py-2">
                    {Math.round(progressPercentage)}% Complete
                  </Badge>
                </div>
              </div>
            </Card.Body>
          </Card>
        )}

        {/* Quiz Form */}
        {!isSubmitted ? (
          <Form className="quiz-form">
            {/* Multiple Choice Questions */}
            {questions.map((q, index) => (
              <Card key={index} className="mb-4 shadow-sm question-card">
                <Card.Body>
                  <div className="d-flex justify-content-between align-items-start mb-3">
                    <Badge bg="primary" className="mb-2">
                      Question {index + 1}
                    </Badge>
                    <Badge bg="secondary" className="mb-2">
                      {q.category}
                    </Badge>
                  </div>
                  
                  <h5 className="fw-bold mb-4">{q.question}</h5>
                  
                  <div className="options-container">
                    {q.options.map((option, optionIndex) => (
                      <div 
                        key={optionIndex} 
                        className={`option-item ${answers[index] === optionIndex ? 'selected' : ''}`}
                        onClick={() => handleAnswerChange(index, optionIndex)}
                      >
                        <Form.Check
                          type="radio"
                          checked={answers[index] === optionIndex}
                          onChange={() => handleAnswerChange(index, optionIndex)}
                          className="me-3"
                        />
                        <span className="option-text">{option}</span>
                      </div>
                    ))}
                  </div>
                </Card.Body>
              </Card>
            ))}

            {/* True/False Questions */}
            {trueFalseQuestions.map((q, index) => (
              <Card key={questions.length + index} className="mb-4 shadow-sm question-card">
                <Card.Body>
                  <div className="d-flex justify-content-between align-items-start mb-3">
                    <Badge bg="success" className="mb-2">
                      Question {questions.length + index + 1}
                    </Badge>
                    <Badge bg="secondary" className="mb-2">
                      {q.category}
                    </Badge>
                  </div>
                  
                  <h5 className="fw-bold mb-4">{q.question}</h5>
                  
                  <div className="options-container">
                    <div 
                      className={`option-item ${answers[questions.length + index] === true ? 'selected' : ''}`}
                      onClick={() => handleAnswerChange(questions.length + index, true)}
                    >
                      <Form.Check
                        type="radio"
                        checked={answers[questions.length + index] === true}
                        onChange={() => handleAnswerChange(questions.length + index, true)}
                        className="me-3"
                      />
                      <span className="option-text">True</span>
                    </div>
                    <div 
                      className={`option-item ${answers[questions.length + index] === false ? 'selected' : ''}`}
                      onClick={() => handleAnswerChange(questions.length + index, false)}
                    >
                      <Form.Check
                        type="radio"
                        checked={answers[questions.length + index] === false}
                        onChange={() => handleAnswerChange(questions.length + index, false)}
                        className="me-3"
                      />
                      <span className="option-text">False</span>
                    </div>
                  </div>
                </Card.Body>
              </Card>
            ))}

            {/* Submit Button */}
            <div className="text-center mt-4">
              <Button 
                variant="primary" 
                size="lg" 
                onClick={calculateScore}
                disabled={!allQuestionsAnswered()}
                className="px-5 py-3 fw-bold"
              >
                <BsCheckLg className="me-2" />
                Submit Assessment
              </Button>
              {!allQuestionsAnswered() && (
                <p className="text-danger mt-2 fw-semibold">
                  Please answer all {totalQuestions} questions before submitting
                </p>
              )}
            </div>
          </Form>
        ) : (
          /* Results Section */
          <div className="results-section">
            {/* Score Summary */}
            <Card className="mb-4 shadow-lg border-0">
              <Card.Body className="text-center p-5">
                {(() => {
                  const performance = getPerformanceLevel();
                  const IconComponent = performance.icon;
                  return (
                    <>
                      <IconComponent className={`display-1 text-${performance.color} mb-3`} />
                      <h2 className={`text-${performance.color} fw-bold mb-3`}>
                        {performance.level} Performance!
                      </h2>
                      <h1 className="display-4 fw-bold mb-3">
                        {score} / {totalQuestions}
                      </h1>
                      <p className="lead text-muted mb-4">
                        You completed the assessment in {formatTime(timeSpent)}
                      </p>
                      <ProgressBar 
                        now={(score / totalQuestions) * 100} 
                        variant={performance.color}
                        className="mb-3"
                        style={{ height: '12px' }}
                      />
                      <p className="text-muted">
                        {Math.round((score / totalQuestions) * 100)}% Accuracy
                      </p>
                    </>
                  );
                })()}
              </Card.Body>
            </Card>

            {/* Detailed Results */}
            <div className="row g-4 align-items-start"> {/* Add gap and align-items-start for better spacing */}
              <div className="col-12 col-lg-8"> {/* Responsive: full width on mobile, 8/12 on large */}
                <h3 className="mb-4">Detailed Results</h3>
                
                {/* Multiple Choice Results */}
                {questions.map((q, index) => (
                  <Card key={index} className="mb-3 shadow-sm">
                    <Card.Body>
                      <div className="d-flex justify-content-between align-items-start mb-3">
                        <Badge bg="primary">Question {index + 1}</Badge>
                        {answers[index] === q.correctAnswer ? (
                          <CheckCircleFill className="text-success fs-4" />
                        ) : (
                          <XCircleFill className="text-danger fs-4" />
                        )}
                      </div>
                      
                      <h6 className="fw-bold mb-3">{q.question}</h6>
                      
                      <div className="mb-3">
                        <strong>Your Answer:</strong> 
                        <span className={`ms-2 ${answers[index] === q.correctAnswer ? 'text-success' : 'text-danger'}`}>
                          {typeof answers[index] !== 'undefined' ? q.options[answers[index]] : <span className="text-warning">No answer</span>}
                        </span>
                      </div>
                      
                      {answers[index] !== q.correctAnswer && (
                        <div className="mb-3">
                          <strong>Correct Answer:</strong> 
                          <span className="ms-2 text-success">
                            {q.options[q.correctAnswer]}
                          </span>
                        </div>
                      )}
                      
                      <Button
                        variant="outline-info"
                        size="sm"
                        onClick={() => toggleExplanation(index)}
                        className="mb-2"
                      >
                        {showExplanation[index] ? 'Hide' : 'Show'} Explanation
                      </Button>
                      
                      {showExplanation[index] && (
                        <Alert variant="info" className="mt-2">
                          <strong>Explanation:</strong> {q.explanation}
                        </Alert>
                      )}
                    </Card.Body>
                  </Card>
                ))}

                {/* True/False Results */}
                {trueFalseQuestions.map((q, index) => (
                  <Card key={questions.length + index} className="mb-3 shadow-sm">
                    <Card.Body>
                      <div className="d-flex justify-content-between align-items-start mb-3">
                        <Badge bg="success">Question {questions.length + index + 1}</Badge>
                        {answers[questions.length + index] === q.correctAnswer ? (
                          <CheckCircleFill className="text-success fs-4" />
                        ) : (
                          <XCircleFill className="text-danger fs-4" />
                        )}
                      </div>
                      
                      <h6 className="fw-bold mb-3">{q.question}</h6>
                      
                      <div className="mb-3">
                        <strong>Your Answer:</strong> 
                        <span className={`ms-2 ${answers[questions.length + index] === q.correctAnswer ? 'text-success' : 'text-danger'}`}>
                          {typeof answers[questions.length + index] !== 'undefined' ? (answers[questions.length + index] ? 'True' : 'False') : <span className="text-warning">No answer</span>}
                        </span>
                      </div>
                      
                      {answers[questions.length + index] !== q.correctAnswer && (
                        <div className="mb-3">
                          <strong>Correct Answer:</strong> 
                          <span className="ms-2 text-success">
                            {q.correctAnswer ? 'True' : 'False'}
                          </span>
                        </div>
                      )}
                      
                      <Button
                        variant="outline-info"
                        size="sm"
                        onClick={() => toggleExplanation(questions.length + index)}
                        className="mb-2"
                      >
                        {showExplanation[questions.length + index] ? 'Hide' : 'Show'} Explanation
                      </Button>
                      
                      {showExplanation[questions.length + index] && (
                        <Alert variant="info" className="mt-2">
                          <strong>Explanation:</strong> {q.explanation}
                        </Alert>
                      )}
                    </Card.Body>
                  </Card>
                ))}
              </div>
              {/* Sidebar */}
              <div className="col-12 col-lg-4"> {/* Responsive: full width on mobile, 4/12 on large */}
                <Card className="shadow-sm sticky-top" style={{ top: '20px' }}>
                  <Card.Body>
                    <h5 className="fw-bold mb-3">Quick Stats</h5>
                    
                    <div className="mb-3">
                      <div className="d-flex justify-content-between">
                        <span>Correct Answers:</span>
                        <Badge bg="success">{score}</Badge>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <div className="d-flex justify-content-between">
                        <span>Incorrect Answers:</span>
                        <Badge bg="danger">{totalQuestions - score}</Badge>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <div className="d-flex justify-content-between">
                        <span>Time Taken:</span>
                        <Badge bg="info">{formatTime(timeSpent)}</Badge>
                      </div>
                    </div>
                    
                    <div className="mb-4">
                      <div className="d-flex justify-content-between">
                        <span>Accuracy:</span>
                        <Badge bg="primary">{Math.round((score / totalQuestions) * 100)}%</Badge>
                      </div>
                    </div>
                    
                    <Button 
                      variant="outline-primary" 
                      onClick={resetQuiz}
                      className="w-100"
                    >
                      <Book className="me-2" />
                      Retake Assessment
                    </Button>
                  </Card.Body>
                </Card>
              </div>
            </div>
          </div>
        )}
      </Container>
    </>
  );
}

export default Assessment;