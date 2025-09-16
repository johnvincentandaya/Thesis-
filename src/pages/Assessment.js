import React, { useState, useEffect } from "react";
import {
  Navbar,
  Nav,
  Container,
  Button,
  Form,
  Alert,
  ProgressBar,
  Card,
  Badge
} from "react-bootstrap";
import { Link } from "react-router-dom";
import {
  Clock,
  Award,
  ArrowRight,
  ArrowLeft
} from "react-bootstrap-icons";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Assessment.css";
import Footer from "../components/Footer";

function Assessment() {
  const [answers, setAnswers] = useState({});
  const [score, setScore] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [timeSpent, setTimeSpent] = useState(0);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [quizStarted, setQuizStarted] = useState(false);
  const [shuffledQuestions, setShuffledQuestions] = useState([]);
  const [currentReview, setCurrentReview] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState(0);

  // === Question Bank ===
  const questionBank = [
    {
      question: "What is the main purpose of Knowledge Distillation?",
      options: [
        "To prune unnecessary weights from the model",
        "To transfer knowledge from a large teacher model to a smaller student model",
        "To reduce the number of FLOPs in a model",
        "To make the model deeper with more layers"
      ],
      correctAnswer: 1,
      explanation:
        "KD transfers the knowledge from a large teacher model into a smaller student model for efficiency."
    },
    {
      question: "What does pruning remove from a neural network?",
      options: [
        "Important features",
        "Random data samples",
        "Redundant or less important weights and nodes",
        "All hidden layers"
      ],
      correctAnswer: 2,
      explanation:
        "Pruning removes redundant or unimportant weights/connections to shrink the model."
    },
    {
      question: "In KD, what type of predictions does the teacher provide to the student?",
      options: [
        "Hard labels only",
        "Soft probability distributions",
        "Pruned outputs",
        "Random guesses"
      ],
      correctAnswer: 1,
      explanation:
        "The teacher provides soft probability distributions that carry richer information than hard labels."
    },
    {
      question: "Which is more likely to cause a significant accuracy drop if applied too aggressively?",
      options: ["Pruning", "Knowledge Distillation", "Both equally", "Neither"],
      correctAnswer: 0,
      explanation:
        "Aggressive pruning can severely reduce accuracy if critical weights are removed."
    },
    {
      question: "What does a compression ratio of 4× mean?",
      options: [
        "The model is four times larger",
        "The model size is reduced to 1/4th of its original size",
        "The model runs four times slower",
        "The accuracy dropped by 4%"
      ],
      correctAnswer: 1,
      explanation:
        "4× compression ratio means the model is four times smaller than the original."
    },
    {
      question: "Which process typically results in a smaller student model learning generalization patterns?",
      options: ["Knowledge Distillation", "Pruning", "Dropout", "Batch Normalization"],
      correctAnswer: 0,
      explanation:
        "KD helps the smaller student mimic the teacher's decision-making and generalize better."
    },
    {
      question: "What is a potential drawback of pruning?",
      options: [
        "It always increases accuracy",
        "It may remove important parameters and reduce accuracy",
        "It increases FLOPs",
        "It doubles the model size"
      ],
      correctAnswer: 1,
      explanation:
        "If not done carefully, pruning can remove essential parameters and hurt accuracy."
    },
    {
      question: "Which metric best indicates the computational savings after pruning?",
      options: ["Accuracy", "FLOPs", "Precision", "Recall"],
      correctAnswer: 1,
      explanation: "FLOPs (floating point operations) directly show computational efficiency."
    },
    {
      question: "What role does temperature scaling play in KD?",
      options: [
        "It controls the hardness/softness of the teacher’s probability outputs",
        "It adjusts the pruning threshold",
        "It reduces FLOPs",
        "It increases dropout"
      ],
      correctAnswer: 0,
      explanation:
        "Temperature scaling smooths teacher predictions, making knowledge transfer easier."
    },
    {
      question:
        "True or False: Pruning always maintains the exact same accuracy as the original model.",
      correctAnswer: false,
      explanation:
        "Pruning usually causes a small accuracy drop, but the goal is to minimize it."
    },
    {
      question:
        "True or False: KD can be seen as training the student with both hard labels and teacher soft labels.",
      correctAnswer: true,
      explanation:
        "KD uses a combination of true labels and teacher’s soft labels for training."
    },
    {
      question: "True or False: Structured pruning removes entire neurons, filters, or layers.",
      correctAnswer: true,
      explanation:
        "Structured pruning eliminates larger structural components, not just weights."
    },
    {
      question: "True or False: Unstructured pruning removes weights based on their magnitude.",
      correctAnswer: true,
      explanation:
        "Unstructured pruning eliminates small-magnitude weights regardless of their position."
    },
    {
      question: "Which is more suitable for deployment on mobile devices?",
      options: [
        "Uncompressed teacher model",
        "Pruned or distilled student model",
        "Random baseline model",
        "Overparameterized model"
      ],
      correctAnswer: 1,
      explanation:
        "Smaller, efficient models (from KD or pruning) are ideal for resource-limited devices."
    },
    {
      question: "What is the main benefit of pruning?",
      options: [
        "Increased accuracy",
        "Reduced model size and faster inference",
        "More training data",
        "More complex model"
      ],
      correctAnswer: 1,
      explanation: "Pruning reduces size and improves inference speed."
    },
    {
      question: "What is the main drawback of KD compared to pruning?",
      options: [
        "It requires training a student model",
        "It removes too many weights",
        "It increases FLOPs",
        "It always decreases accuracy"
      ],
      correctAnswer: 0,
      explanation:
        "KD requires retraining a smaller model, unlike pruning which modifies an existing one."
    },
    {
      question: "True or False: In pruning, red nodes in visualization usually indicate pruned components.",
      correctAnswer: true,
      explanation: "Red nodes typically highlight pruned or inactive parts."
    },
    {
      question: "True or False: KD can improve the student’s generalization even beyond the teacher’s accuracy.",
      correctAnswer: true,
      explanation:
        "Sometimes the student surpasses the teacher due to regularization effects of KD."
    },
    {
      question: "Which technique reduces parameters without retraining a new student model?",
      options: ["Knowledge Distillation", "Pruning", "Batch Normalization", "Data Augmentation"],
      correctAnswer: 1,
      explanation:
        "Pruning compresses an existing model without requiring a student."
    },
    {
      question: "Which technique often requires both teacher and student models during training?",
      options: ["Pruning", "Knowledge Distillation", "Dropout", "Quantization"],
      correctAnswer: 1,
      explanation: "KD involves both teacher and student during training."
    }
  ];

  // Shuffle questions
  const shuffleArray = (array) => {
    let shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  };

  const startQuiz = () => {
    setShuffledQuestions(shuffleArray(questionBank));
    setAnswers({});
    setScore(null);
    setShowResults(false);
    setTimeSpent(0);
    setIsSubmitted(false);
    setQuizStarted(true);
    setCurrentQuestion(0);
  };

  // Timer effect
  useEffect(() => {
    if (quizStarted && !isSubmitted) {
      const timer = setInterval(() => {
        setTimeSpent((prev) => prev + 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [quizStarted, isSubmitted]);

  // Answer change
  const handleAnswerChange = (questionIndex, selectedOption) => {
    setAnswers((prev) => ({
      ...prev,
      [questionIndex]: selectedOption
    }));
  };

  const calculateScore = () => {
    let totalScore = 0;
    shuffledQuestions.forEach((q, index) => {
      if (answers[index] === q.correctAnswer) totalScore++;
    });
    setScore(totalScore);
    setShowResults(true);
    setIsSubmitted(true);
    setCurrentReview(0);
  };

  const resetQuiz = () => {
    setQuizStarted(false);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const totalQuestions = shuffledQuestions.length;
  const answeredQuestions = Object.keys(answers).length;
  const progressPercentage = (answeredQuestions / totalQuestions) * 100;

  const goNext = () => {
    if (currentQuestion < totalQuestions - 1) {
      setCurrentQuestion((prev) => prev + 1);
    }
  };

  const goPrev = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion((prev) => prev - 1);
    }
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

      <Container className="mt-4 mb-5">
        {!quizStarted ? (
          <div className="text-center p-5">
            <h1 className="fw-bold mb-4" style={{ fontSize: '3.5rem', color: '#1890ff' }}>
              🧠 Knowledge Assessment
            </h1>
            <p className="lead text-muted mb-4" style={{ fontSize: '1.3rem', fontWeight: '400' }}>
              Test your understanding of Knowledge Distillation and Pruning concepts
            </p>
            <Button variant="primary" size="lg" onClick={startQuiz}>
              Start Quiz
            </Button>
          </div>
        ) : !isSubmitted ? (
          <>
            {/* Timer and Progress */}
            <Card className="mb-4 shadow-sm">
              <Card.Body>
                <div className="d-flex justify-content-between align-items-center">
                  <div>
                    <Clock className="me-2 text-primary" />
                    Time: {formatTime(timeSpent)}
                  </div>
                  <Badge bg="info">{Math.round(progressPercentage)}% Complete</Badge>
                </div>
                <ProgressBar now={progressPercentage} variant="primary" className="mt-2" />
              </Card.Body>
            </Card>

            {/* One Question at a Time */}
            <Form>
              {shuffledQuestions.length > 0 && (
                <Card className="mb-4 shadow-sm">
                  <Card.Body>
                    <Badge bg="primary" className="mb-2">
                      Question {currentQuestion + 1} of {totalQuestions}
                    </Badge>
                    <h5 className="fw-bold mb-3">
                      {shuffledQuestions[currentQuestion].question}
                    </h5>

                    {shuffledQuestions[currentQuestion].options ? (
                      shuffledQuestions[currentQuestion].options.map(
                        (option, optionIndex) => (
                          <div
                            key={optionIndex}
                            className={`option-item ${
                              answers[currentQuestion] === optionIndex ? "selected" : ""
                            }`}
                            onClick={() => handleAnswerChange(currentQuestion, optionIndex)}
                          >
                            <Form.Check
                              type="radio"
                              checked={answers[currentQuestion] === optionIndex}
                              onChange={() =>
                                handleAnswerChange(currentQuestion, optionIndex)
                              }
                              className="me-2"
                            />
                            <span>{option}</span>
                          </div>
                        )
                      )
                    ) : (
                      <>
                        <div
                          className={`option-item ${
                            answers[currentQuestion] === true ? "selected" : ""
                          }`}
                          onClick={() => handleAnswerChange(currentQuestion, true)}
                        >
                          <Form.Check
                            type="radio"
                            checked={answers[currentQuestion] === true}
                            onChange={() => handleAnswerChange(currentQuestion, true)}
                            className="me-2"
                          />
                          True
                        </div>
                        <div
                          className={`option-item ${
                            answers[currentQuestion] === false ? "selected" : ""
                          }`}
                          onClick={() => handleAnswerChange(currentQuestion, false)}
                        >
                          <Form.Check
                            type="radio"
                            checked={answers[currentQuestion] === false}
                            onChange={() => handleAnswerChange(currentQuestion, false)}
                            className="me-2"
                          />
                          False
                        </div>
                      </>
                    )}
                  </Card.Body>
                </Card>
              )}

              {/* Navigation */}
              <div className="d-flex justify-content-between">
                <Button
                  variant="secondary"
                  disabled={currentQuestion === 0}
                  onClick={goPrev}
                >
                  <ArrowLeft className="me-2" /> Previous
                </Button>

                {currentQuestion < totalQuestions - 1 ? (
                  <Button
                    variant="primary"
                    onClick={goNext}
                    disabled={typeof answers[currentQuestion] === "undefined"}
                  >
                    Next <ArrowRight className="ms-2" />
                  </Button>
                ) : (
                  <Button
                    variant="success"
                    onClick={calculateScore}
                    disabled={answeredQuestions !== totalQuestions}
                  >
                    Submit Assessment
                  </Button>
                )}
              </div>
            </Form>
          </>
        ) : (
          <>
            {/* Results Summary */}
            <Card className="mb-4 shadow-sm">
              <Card.Body className="text-center">
                <h2 className="fw-bold mb-3">Your Score</h2>
                <h1>{score} / {totalQuestions}</h1>
                <p className="text-muted">Time: {formatTime(timeSpent)}</p>
              </Card.Body>
            </Card>

            {/* Review Only Wrong Answers */}
            {(() => {
              const wrongAnswers = [];
              shuffledQuestions.forEach((q, index) => {
                if (answers[index] !== q.correctAnswer) {
                  wrongAnswers.push({ question: q, questionIndex: index, userAnswer: answers[index] });
                }
              });
              
              if (wrongAnswers.length === 0) {
                return (
                  <Card className="mb-4 shadow-sm">
                    <Card.Body className="text-center">
                      <h5 className="fw-bold mb-3 text-success">🎉 Perfect Score!</h5>
                      <p className="text-muted">You got all questions correct! Great job!</p>
                    </Card.Body>
                  </Card>
                );
              }
              
              return (
                <Card className="mb-4 shadow-sm">
                  <Card.Body>
                    <h5 className="fw-bold mb-3">Review Incorrect Answers ({currentReview + 1} of {wrongAnswers.length})</h5>
                    <p><strong>Question:</strong> {wrongAnswers[currentReview].question.question}</p>
                    <p>
                      <strong>Your Answer:</strong>{" "}
                      {typeof wrongAnswers[currentReview].userAnswer !== "undefined"
                        ? wrongAnswers[currentReview].question.options
                          ? wrongAnswers[currentReview].question.options[wrongAnswers[currentReview].userAnswer]
                          : wrongAnswers[currentReview].userAnswer ? "True" : "False"
                        : "No Answer"}
                    </p>
                    <Alert variant="info">
                      <strong>Explanation:</strong> {wrongAnswers[currentReview].question.explanation}
                    </Alert>
                    <div className="d-flex justify-content-between">
                      <Button
                        variant="secondary"
                        disabled={currentReview === 0}
                        onClick={() => setCurrentReview((prev) => prev - 1)}
                      >
                        <ArrowLeft className="me-2" /> Previous
                      </Button>
                      <Button
                        variant="secondary"
                        disabled={currentReview === wrongAnswers.length - 1}
                        onClick={() => setCurrentReview((prev) => prev + 1)}
                      >
                        Next <ArrowRight className="ms-2" />
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              );
            })()}

            <div className="text-center">
              <Button variant="outline-primary" onClick={resetQuiz}>
                Retake Quiz
              </Button>
            </div>
          </>
        )}
      </Container>
      <Footer />
    </>
  );
}

export default Assessment;
