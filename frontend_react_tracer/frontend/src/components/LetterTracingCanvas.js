import React, { useRef, useEffect, useState, useCallback } from "react";
import toast from "react-hot-toast";
import { motion, AnimatePresence } from "framer-motion";
import {
  RotateCcw,
  Sparkles,
  Lightbulb,
  Target,
  TrendingUp,
} from "lucide-react";

const LetterTracingCanvas = ({ letter, onScoreUpdate }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokes, setStrokes] = useState([]);
  const [currentStroke, setCurrentStroke] = useState([]);
  const [feedback, setFeedback] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showSparkles, setShowSparkles] = useState(false);
  const [analysis, setAnalysis] = useState({});
  const [improvements, setImprovements] = useState([]);
  const [showTips, setShowTips] = useState(false);
  const [renderKey, setRenderKey] = useState(0);

  // New state management approach
  const [currentLetter, setCurrentLetter] = useState(letter);
  const [feedbackForLetter, setFeedbackForLetter] = useState({});
  const [shouldResetFeedback, setShouldResetFeedback] = useState(false);

  // Use callback to prevent unnecessary re-renders
  const updateFeedback = useCallback(
    (newFeedback) => {
      console.log("updateFeedback called with:", newFeedback);
      console.log("Current letter:", currentLetter);

      // Force immediate state updates
      setFeedback(newFeedback);
      setAnalysis(newFeedback.analysis || {});
      setImprovements(newFeedback.improvements || []);
      setRenderKey((prev) => prev + 1);

      console.log("State updates triggered, render key incremented");

      // Store feedback for current letter
      if (newFeedback) {
        setFeedbackForLetter((prev) => {
          const newState = {
            ...prev,
            [currentLetter]: newFeedback,
          };
          console.log("Updated feedbackForLetter:", newState);
          return newState;
        });
      }

      // Force a re-render after a short delay to ensure state is updated
      setTimeout(() => {
        console.log("Forcing re-render check - feedback should be visible now");
        setRenderKey((prev) => prev + 1);
        const category = newFeedback?.category || "good";
        const emoji =
          category === "excellent"
            ? "🌟"
            : category === "good"
            ? "👍"
            : category === "fair"
            ? "💪"
            : "🔄";
        toast.success(
          `${emoji} Letter ${currentLetter}: ${
            newFeedback?.message || "Great effort!"
          }`,
          {
            duration: 2500,
          }
        );
      }, 100);
    },
    [currentLetter]
  );

  // Handle letter changes
  useEffect(() => {
    console.log("Letter changed from", currentLetter, "to", letter);

    // Only reset if it's a different letter and we should reset
    if (currentLetter !== letter && shouldResetFeedback) {
      console.log("Resetting feedback for new letter");
      setFeedback(null);
      setAnalysis({});
      setImprovements([]);
      setShowTips(false);
      setShouldResetFeedback(false);
    }

    setCurrentLetter(letter);

    // Set up canvas for new letter
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      canvas.width = 400;
      canvas.height = 400;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLetterOutline(ctx, letter);
    }

    // Reset strokes
    setStrokes([]);
    setCurrentStroke([]);
  }, [letter, currentLetter, shouldResetFeedback]);

  // Restore feedback if we have it for this letter
  useEffect(() => {
    if (feedbackForLetter[letter] && !feedback) {
      console.log("Restoring feedback for letter:", letter);
      updateFeedback(feedbackForLetter[letter]);
    }
  }, [letter, feedbackForLetter, feedback, updateFeedback]);

  // Debug effect
  useEffect(() => {
    console.log("Feedback state changed:", feedback);
    console.log("Analysis state changed:", analysis);
    console.log("Improvements state changed:", improvements);
  }, [feedback, analysis, improvements]);

  const drawLetterOutline = (ctx, letter) => {
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    const centerX = 200;
    const centerY = 200;
    const size = 80;

    switch (letter) {
      case "A":
        // A shape - two diagonal lines and a horizontal crossbar
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 4, centerY);
        ctx.lineTo(centerX + size / 4, centerY);
        ctx.stroke();
        break;
      case "B":
        // B shape - vertical line with two curves
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY - size / 2,
          centerX + size / 6,
          centerY - size / 6
        );
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY,
          centerX - size / 3,
          centerY
        );
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY,
          centerX + size / 6,
          centerY + size / 6
        );
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY + size / 2,
          centerX - size / 3,
          centerY + size / 2
        );
        ctx.stroke();
        break;
      case "C":
        // C shape - open curve
        ctx.beginPath();
        ctx.arc(centerX, centerY, size / 2, Math.PI / 6, -Math.PI / 6, false);
        ctx.stroke();
        break;
      case "D":
        // D shape - vertical line with curve
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.quadraticCurveTo(
          centerX + size / 3,
          centerY - size / 2,
          centerX + size / 3,
          centerY
        );
        ctx.quadraticCurveTo(
          centerX + size / 3,
          centerY + size / 2,
          centerX - size / 3,
          centerY + size / 2
        );
        ctx.stroke();
        break;
      case "E":
        // E shape - vertical line with three horizontals
        ctx.beginPath();
        ctx.moveTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 6, centerY);
        ctx.stroke();
        break;
      case "F":
        // F shape - vertical line with two horizontals
        ctx.beginPath();
        ctx.moveTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 6, centerY);
        ctx.stroke();
        break;
      case "G":
        // G shape - C with tail
        ctx.beginPath();
        ctx.arc(centerX, centerY, size / 2, Math.PI / 6, -Math.PI / 6, false);
        ctx.moveTo(centerX + size / 2, centerY);
        ctx.lineTo(centerX + size / 3, centerY);
        ctx.lineTo(centerX + size / 3, centerY + size / 4);
        ctx.stroke();
        break;
      case "H":
        // H shape - two verticals with horizontal
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 3, centerY);
        ctx.stroke();
        break;
      case "I":
        // I shape - vertical with top and bottom lines
        ctx.beginPath();
        ctx.moveTo(centerX - size / 4, centerY - size / 2);
        ctx.lineTo(centerX + size / 4, centerY - size / 2);
        ctx.moveTo(centerX, centerY - size / 2);
        ctx.lineTo(centerX, centerY + size / 2);
        ctx.moveTo(centerX - size / 4, centerY + size / 2);
        ctx.lineTo(centerX + size / 4, centerY + size / 2);
        ctx.stroke();
        break;
      case "J":
        // J shape - curve with dot
        ctx.beginPath();
        ctx.moveTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 3);
        ctx.quadraticCurveTo(
          centerX + size / 3,
          centerY + size / 2,
          centerX + size / 6,
          centerY + size / 2
        );
        ctx.quadraticCurveTo(
          centerX,
          centerY + size / 2,
          centerX,
          centerY + size / 3
        );
        ctx.stroke();
        break;
      case "K":
        // K shape - vertical with two diagonals
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 3, centerY - size / 3);
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 3, centerY + size / 3);
        ctx.stroke();
        break;
      case "L":
        // L shape - vertical with bottom horizontal
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.stroke();
        break;
      case "M":
        // M shape - mountain shape
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY + size / 2);
        ctx.lineTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX, centerY);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.stroke();
        break;
      case "N":
        // N shape - two verticals with diagonal
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY + size / 2);
        ctx.lineTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.stroke();
        break;
      case "O":
        // O shape - circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, size / 2, 0, 2 * Math.PI);
        ctx.stroke();
        break;
      case "P":
        // P shape - vertical with top curve
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY - size / 2,
          centerX + size / 6,
          centerY - size / 6
        );
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY,
          centerX - size / 3,
          centerY
        );
        ctx.stroke();
        break;
      case "Q":
        // Q shape - O with tail
        ctx.beginPath();
        ctx.arc(centerX, centerY, size / 2, 0, 2 * Math.PI);
        ctx.moveTo(centerX + size / 3, centerY + size / 3);
        ctx.lineTo(centerX + size / 2, centerY + size / 2);
        ctx.stroke();
        break;
      case "R":
        // R shape - P with diagonal tail
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY - size / 2,
          centerX + size / 6,
          centerY - size / 6
        );
        ctx.quadraticCurveTo(
          centerX + size / 6,
          centerY,
          centerX - size / 3,
          centerY
        );
        ctx.moveTo(centerX - size / 3, centerY);
        ctx.lineTo(centerX + size / 3, centerY + size / 3);
        ctx.stroke();
        break;
      case "S":
        // S shape - snake curve
        ctx.beginPath();
        ctx.arc(
          centerX - size / 6,
          centerY - size / 4,
          size / 4,
          Math.PI / 4,
          (3 * Math.PI) / 4
        );
        ctx.arc(
          centerX + size / 6,
          centerY + size / 4,
          size / 4,
          (-3 * Math.PI) / 4,
          -Math.PI / 4
        );
        ctx.stroke();
        break;
      case "T":
        // T shape - horizontal with vertical
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.moveTo(centerX, centerY - size / 2);
        ctx.lineTo(centerX, centerY + size / 2);
        ctx.stroke();
        break;
      case "U":
        // U shape - curve with verticals
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 3);
        ctx.quadraticCurveTo(
          centerX - size / 3,
          centerY + size / 2,
          centerX,
          centerY + size / 2
        );
        ctx.quadraticCurveTo(
          centerX + size / 3,
          centerY + size / 2,
          centerX + size / 3,
          centerY + size / 3
        );
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.stroke();
        break;
      case "V":
        // V shape - checkmark
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.stroke();
        break;
      case "W":
        // W shape - double V
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 6, centerY + size / 2);
        ctx.lineTo(centerX, centerY - size / 4);
        ctx.lineTo(centerX + size / 6, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.stroke();
        break;
      case "X":
        // X shape - crossing diagonals
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.moveTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.stroke();
        break;
      case "Y":
        // Y shape - V with vertical
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX, centerY);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX, centerY + size / 2);
        ctx.stroke();
        break;
      case "Z":
        // Z shape - horizontal, diagonal, horizontal
        ctx.beginPath();
        ctx.moveTo(centerX - size / 3, centerY - size / 2);
        ctx.lineTo(centerX + size / 3, centerY - size / 2);
        ctx.lineTo(centerX - size / 3, centerY + size / 2);
        ctx.lineTo(centerX + size / 3, centerY + size / 2);
        ctx.stroke();
        break;
      default:
        break;
    }

    ctx.setLineDash([]);
  };

  const startDrawing = (e) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCurrentStroke([[x, y]]);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCurrentStroke((prev) => [...prev, [x, y]]);

    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#667eea";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (currentStroke.length > 0) {
      const lastPoint = currentStroke[currentStroke.length - 1];
      ctx.beginPath();
      ctx.moveTo(lastPoint[0], lastPoint[1]);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      if (currentStroke.length > 0) {
        setStrokes((prev) => [...prev, currentStroke]);
        setCurrentStroke([]);
      }
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLetterOutline(ctx, letter);

    setStrokes([]);
    setCurrentStroke([]);
    // Don't clear feedback when clearing canvas - let user keep their results
  };

  const clearFeedback = () => {
    setFeedback(null);
    setAnalysis({});
    setImprovements([]);
    setShowTips(false);
    // Remove feedback for current letter
    setFeedbackForLetter((prev) => {
      const newState = { ...prev };
      delete newState[currentLetter];
      return newState;
    });
  };

  const analyzeStrokes = async () => {
    if (strokes.length === 0) return;

    setIsAnalyzing(true);
    console.log("Starting analysis..."); // Debug log
    console.log("Current feedback state before:", feedback); // Debug log

    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          strokes: strokes,
          letter: letter,
        }),
      });

      const data = await response.json();
      console.log("Backend response:", data); // Debug log

      // Always set feedback, even for low scores
      console.log("Setting feedback state:", data); // Debug log
      updateFeedback(data);

      console.log("Feedback state should be set now"); // Debug log

      // Show sparkles for excellent scores
      if (data.score > 0.75) {
        setShowSparkles(true);
        setTimeout(() => setShowSparkles(false), 2000);
      }

      if (onScoreUpdate) {
        onScoreUpdate(data.score);
      }
    } catch (error) {
      console.error("Error analyzing strokes:", error);
      // Always show feedback even on error
      const errorFeedback = {
        result: "Error",
        message:
          "Sorry, there was an error analyzing your handwriting. Please try again!",
        score: 0.3,
        category: "needs_improvement",
        improvements: ["Try again with careful strokes"],
      };
      console.log("Setting error feedback:", errorFeedback); // Debug log
      updateFeedback(errorFeedback);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getScoreColor = (score) => {
    if (score > 0.75) return "#4caf50";
    if (score > 0.55) return "#ff9800";
    if (score > 0.35) return "#ff5722";
    return "#f44336";
  };

  const getScoreText = (score) => {
    if (score > 0.75) return "Excellent";
    if (score > 0.55) return "Good";
    if (score > 0.35) return "Fair";
    return "Needs Improvement";
  };

  console.log("Rendering component with feedback:", feedback); // Debug log

  return (
    <div className="canvas-container">
      <div className="canvas-layout">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            className="tracing-canvas"
          />

          <div className="canvas-controls">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="control-button clear-button"
              onClick={clearCanvas}
            >
              <RotateCcw size={20} />
              Clear
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`control-button analyze-button ${
                strokes.length === 0 ? "disabled" : ""
              }`}
              onClick={analyzeStrokes}
              disabled={strokes.length === 0 || isAnalyzing}
            >
              {isAnalyzing ? "Analyzing..." : "Analyze"}
            </motion.button>
          </div>
        </div>

        {/* Feedback Section - Right Side */}
        <div className="feedback-sidebar" style={{ minHeight: "400px" }}>
          {feedback && feedback.result ? (
            <motion.div
              key={renderKey} // Force re-render when feedback changes
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="feedback-section"
            >
              <div className="feedback-header">
                <h3>Letter {letter} Results</h3>
                <div className="feedback-controls">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="tips-toggle-button"
                    onClick={() => setShowTips(!showTips)}
                  >
                    <Lightbulb size={16} />
                    {showTips ? "Hide" : "Show"} Tips
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="clear-feedback-button"
                    onClick={clearFeedback}
                    style={{ backgroundColor: "#f44336", color: "white" }}
                  >
                    <RotateCcw size={16} />
                    Clear Results
                  </motion.button>
                </div>
              </div>

              <div className="score-display">
                <div
                  className="score-circle"
                  style={{
                    backgroundColor: getScoreColor(feedback.score),
                    borderColor: getScoreColor(feedback.score),
                  }}
                >
                  <span className="score-percentage">
                    {Math.round(feedback.score * 100)}%
                  </span>
                  <span className="score-label">
                    {getScoreText(feedback.score)}
                  </span>
                </div>
              </div>

              <div className="feedback-message">
                <p
                  className={`feedback-text ${
                    feedback.result === "Wrong Letter" ? "wrong-letter" : ""
                  }`}
                >
                  {feedback.message || "Analysis completed!"}
                </p>
                {feedback.tip && (
                  <div className="letter-tip">
                    <strong>💡 Tip:</strong> {feedback.tip}
                  </div>
                )}
              </div>

              {/* Analysis Bars */}
              {analysis && Object.keys(analysis).length > 0 && (
                <div className="analysis-section">
                  <h4>📊 Detailed Analysis</h4>
                  <div className="analysis-bars">
                    {Object.entries(analysis).map(([key, value]) => (
                      <div key={key} className="analysis-bar">
                        <div className="bar-label">
                          <span>
                            {key.charAt(0).toUpperCase() + key.slice(1)}
                          </span>
                          <span>{Math.round(value * 100)}%</span>
                        </div>
                        <div className="bar-container">
                          <div
                            className="bar-fill"
                            style={{
                              width: `${value * 100}%`,
                              backgroundColor:
                                value > 0.7
                                  ? "#4caf50"
                                  : value > 0.5
                                  ? "#ff9800"
                                  : "#f44336",
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Improvement Tips - Always show if available */}
              {improvements && improvements.length > 0 && (
                <div className="improvement-tips">
                  <div className="tips-header">
                    <h4>
                      <TrendingUp size={16} />
                      Improvement Tips
                    </h4>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="tips-toggle-button"
                      onClick={() => setShowTips(!showTips)}
                    >
                      <Lightbulb size={16} />
                      {showTips ? "Hide" : "Show"} Details
                    </motion.button>
                  </div>

                  <AnimatePresence>
                    {showTips && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                      >
                        <ul>
                          {improvements.map((tip, index) => (
                            <li key={index}>{tip}</li>
                          ))}
                        </ul>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </motion.div>
          ) : (
            <div className="feedback-placeholder">
              <div className="placeholder-content">
                <h3>Letter {letter}</h3>
                <p>
                  Draw the letter {letter} in the canvas and click "Analyze" to
                  see your results!
                </p>
                <div className="placeholder-icon">✏️</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LetterTracingCanvas;
