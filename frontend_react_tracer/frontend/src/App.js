import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Star,
  Trophy,
  Volume2,
  RotateCcw,
  Home,
  BookOpen,
  Info,
  Target,
} from "lucide-react";
import LetterTracingCanvas from "./components/LetterTracingCanvas";
import "./App.css";

function App() {
  const [selectedLetter, setSelectedLetter] = useState("A");
  const [currentPage, setCurrentPage] = useState("home"); // home, practice, progress
  const [progress, setProgress] = useState({});
  const [totalScore, setTotalScore] = useState(0);
  const [currentStreak, setCurrentStreak] = useState(0);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);

  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const letterWords = {
    A: "Apple",
    B: "Ball",
    C: "Cat",
    D: "Dog",
    E: "Elephant",
    F: "Fish",
    G: "Giraffe",
    H: "House",
    I: "Ice",
    J: "Jump",
    K: "Kite",
    L: "Lion",
    M: "Moon",
    N: "Nest",
    O: "Orange",
    P: "Pig",
    Q: "Queen",
    R: "Rainbow",
    S: "Sun",
    T: "Tree",
    U: "Umbrella",
    V: "Violin",
    W: "Water",
    X: "Xylophone",
    Y: "Yellow",
    Z: "Zebra",
  };

  // Detailed step-by-step instructions for each letter
  const letterInstructions = {
    A: [
      "1. Start at the top point",
      "2. Draw a diagonal line down to the left",
      "3. Draw another diagonal line down to the right",
      "4. Connect them with a horizontal line in the middle",
    ],
    B: [
      "1. Start with a straight line down",
      "2. Add a curve at the top going to the right",
      "3. Add another curve in the middle going to the right",
    ],
    C: [
      "1. Start at the top",
      "2. Draw a curve that goes around to the bottom",
      "3. Like drawing a smile or crescent moon",
    ],
    D: [
      "1. Start with a straight line down",
      "2. Add a big curve from top to bottom",
      "3. Connect the ends to make a complete shape",
    ],
    E: [
      "1. Draw a straight line down",
      "2. Add a horizontal line at the top",
      "3. Add a horizontal line in the middle",
      "4. Add a horizontal line at the bottom",
    ],
    F: [
      "1. Draw a straight line down",
      "2. Add a horizontal line at the top",
      "3. Add a horizontal line in the middle",
    ],
    G: [
      "1. Start like C - draw a curve from top to bottom",
      "2. Add a small horizontal line at the bottom going to the right",
    ],
    H: [
      "1. Draw two straight lines down",
      "2. Connect them with a horizontal line in the middle",
    ],
    I: [
      "1. Draw a straight line down",
      "2. Add a dot on top",
      "3. Add a line at the bottom",
    ],
    J: [
      "1. Draw a curve that goes down and curves to the left",
      "2. Add a dot on top",
    ],
    K: [
      "1. Draw a straight line down",
      "2. Add a diagonal line from the middle going up and right",
      "3. Add another diagonal line from the middle going down and right",
    ],
    L: [
      "1. Draw a straight line down",
      "2. Add a horizontal line at the bottom going to the right",
    ],
    M: [
      "1. Start at the top",
      "2. Go down, then up to the middle",
      "3. Go down again, then up to the top",
      "4. Finally go down",
    ],
    N: [
      "1. Draw a straight line down",
      "2. Draw a diagonal line up and to the right",
      "3. Draw another straight line down",
    ],
    O: [
      "1. Start at the top",
      "2. Draw a complete circle or oval",
      "3. Make sure it's smooth and round",
    ],
    P: [
      "1. Draw a straight line down",
      "2. Add a curve at the top going to the right",
    ],
    Q: [
      "1. Draw a circle like O",
      "2. Add a small diagonal line at the bottom right going down and right",
    ],
    R: [
      "1. Draw a straight line down",
      "2. Add a curve at the top going to the right",
      "3. Add a small diagonal line from the middle going down and right",
    ],
    S: [
      "1. Start at the top",
      "2. Draw a curve that goes down and around",
      "3. Then up and around - like a snake",
    ],
    T: [
      "1. Draw a horizontal line at the top",
      "2. Draw a straight line down from the middle",
    ],
    U: ["1. Draw a curve that goes down", "2. Then curves up at the bottom"],
    V: [
      "1. Draw two diagonal lines that meet at the bottom",
      "2. Like drawing a checkmark",
    ],
    W: ["1. Like V, but with three points", "2. Down, up, down, up"],
    X: [
      "1. Draw a diagonal line from top-left to bottom-right",
      "2. Draw another diagonal line from top-right to bottom-left",
      "3. They cross in the middle",
    ],
    Y: [
      "1. Draw two diagonal lines that meet in the middle",
      "2. Draw one straight line down from the middle",
    ],
    Z: [
      "1. Draw a horizontal line at the top",
      "2. Draw a diagonal line down and to the right",
      "3. Draw a horizontal line at the bottom",
    ],
  };

  useEffect(() => {
    const savedProgress = localStorage.getItem("handwritingProgress");
    if (savedProgress) {
      setProgress(JSON.parse(savedProgress));
    }
  }, []);

  const updateProgress = (letter, score) => {
    const newProgress = {
      ...progress,
      [letter]: {
        score: Math.max(score, progress[letter]?.score || 0),
        attempts: (progress[letter]?.attempts || 0) + 1,
        lastPracticed: new Date().toISOString(),
      },
    };
    setProgress(newProgress);
    localStorage.setItem("handwritingProgress", JSON.stringify(newProgress));

    // Update total score
    const total = Object.values(newProgress).reduce(
      (sum, p) => sum + p.score,
      0
    );
    setTotalScore(total);

    // Check for celebration
    if (score > 0.75) {
      setShowCelebration(true);
      setTimeout(() => setShowCelebration(false), 3000);
    }
  };

  const speakLetter = (letter) => {
    const msg = new SpeechSynthesisUtterance(
      `Letter ${letter}, like in ${letterWords[letter]}`
    );
    msg.lang = "en-US";
    msg.rate = 0.8;
    window.speechSynthesis.speak(msg);
  };

  const clearProgress = () => {
    setProgress({});
    setTotalScore(0);
    setCurrentStreak(0);
    localStorage.removeItem("handwritingProgress");
  };

  const resetAllData = () => {
    if (
      window.confirm(
        "Are you sure you want to reset all progress data? This cannot be undone."
      )
    ) {
      clearProgress();
      // Also clear any current feedback
      setShowCelebration(false);
    }
  };

  const getLetterColor = (letter) => {
    const letterProgress = progress[letter];
    if (!letterProgress) return "#e0e0e0";
    if (letterProgress.score > 0.75) return "#4caf50";
    if (letterProgress.score > 0.55) return "#ff9800";
    return "#f44336";
  };

  const HomePage = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="home-page"
    >
      <div className="hero-section">
        <h1 className="main-title">🎨 Fun Letter Tracing</h1>
        <p className="subtitle">Learn to write letters with fun and games!</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <Trophy className="stat-icon" />
          <h3>{totalScore.toFixed(1)}</h3>
          <p>Total Score</p>
        </div>
        <div className="stat-card">
          <Star className="stat-icon" />
          <h3>{Object.keys(progress).length}</h3>
          <p>Letters Practiced</p>
        </div>
        <div className="stat-card">
          <BookOpen className="stat-icon" />
          <h3>{currentStreak}</h3>
          <p>Day Streak</p>
        </div>
      </div>

      <div className="action-buttons">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="primary-button"
          onClick={() => setCurrentPage("practice")}
        >
          Start Practicing
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="secondary-button"
          onClick={() => setCurrentPage("progress")}
        >
          View Progress
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="reset-button"
          onClick={resetAllData}
        >
          Reset Progress
        </motion.button>
      </div>
    </motion.div>
  );

  const PracticePage = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="practice-page"
    >
      <div className="header">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="nav-button"
          onClick={() => setCurrentPage("home")}
        >
          <Home size={20} />
        </motion.button>
        <h2>Practice Letters</h2>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="nav-button"
          onClick={() => setCurrentPage("progress")}
        >
          <BookOpen size={20} />
        </motion.button>
      </div>

      <div className="letter-info">
        <div className="letter-display">
          <h1 className="current-letter">{selectedLetter}</h1>
          <p className="letter-word">{letterWords[selectedLetter]}</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="sound-button"
          onClick={() => speakLetter(selectedLetter)}
        >
          <Volume2 size={24} />
        </motion.button>
      </div>

      <div className="letter-grid">
        {letters.map((letter) => (
          <motion.button
            key={letter}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`letter-button ${
              letter === selectedLetter ? "selected" : ""
            }`}
            style={{ backgroundColor: getLetterColor(letter) }}
            onClick={() => setSelectedLetter(letter)}
          >
            {letter}
            {progress[letter] && (
              <div className="progress-indicator">
                {progress[letter].score > 0.75 ? "⭐" : "✓"}
              </div>
            )}
          </motion.button>
        ))}
      </div>

      {/* Letter Instructions Panel */}
      <div className="instructions-panel">
        <div className="instructions-header">
          <h3>📝 How to Write Letter {selectedLetter}</h3>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="toggle-button"
            onClick={() => setShowInstructions(!showInstructions)}
          >
            {showInstructions ? "Hide" : "Show"} Instructions
          </motion.button>
        </div>

        <AnimatePresence>
          {showInstructions && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="instructions-content"
            >
              <div className="instruction-steps">
                {letterInstructions[selectedLetter].map((step, index) => (
                  <div key={index} className="instruction-step">
                    <span className="step-number">{index + 1}</span>
                    <p>{step}</p>
                  </div>
                ))}
              </div>
              <div className="instruction-tip">
                <Target size={16} />
                <span>
                  <strong>Tip:</strong> Take your time and follow each step
                  carefully!
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <LetterTracingCanvas
        letter={selectedLetter}
        onScoreUpdate={(score) => updateProgress(selectedLetter, score)}
      />
    </motion.div>
  );

  const ProgressPage = () => (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="progress-page"
    >
      <div className="header">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="nav-button"
          onClick={() => setCurrentPage("home")}
        >
          <Home size={20} />
        </motion.button>
        <h2>Your Progress</h2>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="nav-button"
          onClick={resetAllData}
        >
          <RotateCcw size={20} />
        </motion.button>
      </div>

      <div className="progress-summary">
        <div className="summary-card">
          <h3>Overall Progress</h3>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(Object.keys(progress).length / 26) * 100}%` }}
            ></div>
          </div>
          <p>{Object.keys(progress).length} of 26 letters practiced</p>
        </div>
      </div>

      {/* Progress Charts */}
      <div className="progress-charts">
        <div className="chart-section">
          <h3>📊 Performance Overview</h3>
          <div className="chart-grid">
            <div className="chart-card">
              <h4>Score Distribution</h4>
              <div className="score-distribution">
                <div className="score-bar">
                  <span>Excellent (75%+)</span>
                  <div className="bar">
                    <div
                      className="bar-fill excellent"
                      style={{
                        width: `${
                          (Object.values(progress).filter((p) => p.score > 0.75)
                            .length /
                            Math.max(Object.keys(progress).length, 1)) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <span>
                    {
                      Object.values(progress).filter((p) => p.score > 0.75)
                        .length
                    }
                  </span>
                </div>
                <div className="score-bar">
                  <span>Good (55-74%)</span>
                  <div className="bar">
                    <div
                      className="bar-fill good"
                      style={{
                        width: `${
                          (Object.values(progress).filter(
                            (p) => p.score > 0.55 && p.score <= 0.75
                          ).length /
                            Math.max(Object.keys(progress).length, 1)) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <span>
                    {
                      Object.values(progress).filter(
                        (p) => p.score > 0.55 && p.score <= 0.75
                      ).length
                    }
                  </span>
                </div>
                <div className="score-bar">
                  <span>Fair (35-54%)</span>
                  <div className="bar">
                    <div
                      className="bar-fill fair"
                      style={{
                        width: `${
                          (Object.values(progress).filter(
                            (p) => p.score > 0.35 && p.score <= 0.55
                          ).length /
                            Math.max(Object.keys(progress).length, 1)) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <span>
                    {
                      Object.values(progress).filter(
                        (p) => p.score > 0.35 && p.score <= 0.55
                      ).length
                    }
                  </span>
                </div>
                <div className="score-bar">
                  <span>Needs Improvement (0-34%)</span>
                  <div className="bar">
                    <div
                      className="bar-fill needs-improvement"
                      style={{
                        width: `${
                          (Object.values(progress).filter(
                            (p) => p.score <= 0.35
                          ).length /
                            Math.max(Object.keys(progress).length, 1)) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <span>
                    {
                      Object.values(progress).filter((p) => p.score <= 0.35)
                        .length
                    }
                  </span>
                </div>
              </div>
            </div>

            <div className="chart-card">
              <h4>Practice Frequency</h4>
              <div className="practice-stats">
                <div className="stat-item">
                  <span className="stat-label">Most Practiced:</span>
                  <span className="stat-value">
                    {Object.keys(progress).length > 0
                      ? Object.entries(progress).sort(
                          (a, b) => b[1].attempts - a[1].attempts
                        )[0][0]
                      : "None"}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Best Score:</span>
                  <span className="stat-value">
                    {Object.keys(progress).length > 0
                      ? `${
                          Object.entries(progress).sort(
                            (a, b) => b[1].score - a[1].score
                          )[0][0]
                        } (${Math.round(
                          Object.entries(progress).sort(
                            (a, b) => b[1].score - a[1].score
                          )[0][1].score * 100
                        )}%)`
                      : "None"}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Average Score:</span>
                  <span className="stat-value">
                    {Object.keys(progress).length > 0
                      ? `${Math.round(
                          (Object.values(progress).reduce(
                            (sum, p) => sum + p.score,
                            0
                          ) /
                            Object.keys(progress).length) *
                            100
                        )}%`
                      : "0%"}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Total Attempts:</span>
                  <span className="stat-value">
                    {Object.values(progress).reduce(
                      (sum, p) => sum + p.attempts,
                      0
                    )}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="letter-progress-grid">
        {letters.map((letter) => {
          const letterProgress = progress[letter];
          return (
            <div key={letter} className="letter-progress-card">
              <h3>{letter}</h3>
              {letterProgress ? (
                <>
                  <div className="score-display">
                    <span className="score">
                      {Math.round(letterProgress.score * 100)}%
                    </span>
                  </div>
                  <p>Attempts: {letterProgress.attempts}</p>
                  <p className="last-practiced">
                    {new Date(
                      letterProgress.lastPracticed
                    ).toLocaleDateString()}
                  </p>
                </>
              ) : (
                <p className="not-practiced">Not practiced yet</p>
              )}
            </div>
          );
        })}
      </div>
    </motion.div>
  );

  return (
    <div className="app">
      <AnimatePresence mode="wait">
        {currentPage === "home" && <HomePage key="home" />}
        {currentPage === "practice" && <PracticePage key="practice" />}
        {currentPage === "progress" && <ProgressPage key="progress" />}
      </AnimatePresence>

      {showCelebration && (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0, opacity: 0 }}
          className="celebration"
        >
          🎉 Great job! 🎉
        </motion.div>
      )}
    </div>
  );
}

export default App;
