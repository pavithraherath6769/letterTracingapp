from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import random
import pickle
from train_model import HandwritingAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize the handwriting analyzer
analyzer = HandwritingAnalyzer()

# Try to load existing model, otherwise train a new one
try:
    analyzer.load_model()
    print("✅ Loaded existing handwriting model")
except:
    print("🔄 No existing model found. Training new model...")
    analyzer.train_model()

# Educational tips for each letter with more detailed instructions
letter_tips = {
    'A': "Start at the top point, draw a diagonal line down to the left, then another diagonal line down to the right, and connect them with a horizontal line in the middle.",
    'B': "Start with a straight line down, then add two curves - one at the top going to the right, and one in the middle also going to the right.",
    'C': "Draw a curve that starts at the top and goes around to the bottom, like a smile or a crescent moon.",
    'D': "Start with a straight line down, then add a big curve from the top to the bottom, connecting the ends.",
    'E': "Draw a straight line down, then add three horizontal lines - one at the top, one in the middle, and one at the bottom.",
    'F': "Like E, but only two horizontal lines - one at the top and one in the middle.",
    'G': "Start like C, but add a small horizontal line at the bottom going to the right.",
    'H': "Two straight lines down with a horizontal line connecting them in the middle.",
    'I': "A straight line down with a dot on top and a line at the bottom.",
    'J': "A curve that goes down and then curves to the left, with a dot on top.",
    'K': "A straight line down with two diagonal lines coming out from the middle - one up and right, one down and right.",
    'L': "A straight line down with a horizontal line at the bottom going to the right.",
    'M': "Start at the top, go down, then up to the middle, then down again, then up to the top, then down.",
    'N': "A straight line down, then a diagonal line up and to the right, then another straight line down.",
    'O': "A perfect circle or oval shape - start at the top and go around smoothly.",
    'P': "A straight line down with a curve at the top going to the right.",
    'Q': "Like O, but add a small diagonal line at the bottom right going down and right.",
    'R': "Like P, but add a small diagonal line from the middle going down and to the right.",
    'S': "A curve that starts at the top, goes down and around, then up and around - like a snake.",
    'T': "A horizontal line at the top with a straight line down from the middle.",
    'U': "A curve that goes down, then curves up at the bottom.",
    'V': "Two diagonal lines that meet at the bottom - like a checkmark.",
    'W': "Like V, but with three points - down, up, down, up.",
    'X': "Two diagonal lines that cross in the middle - one from top-left to bottom-right, one from top-right to bottom-left.",
    'Y': "Two diagonal lines that meet in the middle, then one straight line down from the middle.",
    'Z': "A horizontal line at the top, a diagonal line down and to the right, then a horizontal line at the bottom."
}

# Detailed encouraging messages based on score with specific feedback
encouraging_messages = {
    'excellent': [
        "🌟 Amazing! Your letter looks perfect! You're a handwriting superstar!",
        "🎉 Fantastic job! Your strokes are smooth and well-formed!",
        "⭐ Outstanding! You've mastered this letter beautifully!",
        "🏆 Brilliant! Your handwriting is getting better every time!",
        "✨ Perfect! You're doing an excellent job with your writing!"
    ],
    'good': [
        "👍 Great work! Your letter looks really good!",
        "👏 Nice job! You're making steady progress!",
        "💪 Well done! Your handwriting is improving nicely!",
        "✨ Good effort! You're learning fast!",
        "🎯 Nice work! Keep practicing to make it even better!"
    ],
    'fair': [
        "💪 Nice try! Let's practice a bit more to make it smoother!",
        "🔄 Good attempt! Try tracing it again with slower, steadier strokes!",
        "📝 You're getting there! Keep practicing to improve!",
        "🎯 Almost there! One more try with careful strokes!",
        "🌟 Good effort! Try to make your strokes more consistent!"
    ],
    'needs_improvement': [
        "🔄 Let's try again! Take your time and go slowly!",
        "💪 Don't give up! Practice makes perfect - try again!",
        "📚 Keep practicing! Slow down and focus on smooth strokes!",
        "🌟 Try again! Remember to follow the letter shape carefully!",
        "🎯 Let's practice more! Try to make your strokes smoother!"
    ]
}

# Specific improvement tips based on analysis
improvement_tips = {
    'smoothness': [
        "Try to make your strokes smoother by moving your hand more steadily.",
        "Practice drawing smooth curves instead of jagged lines.",
        "Slow down a bit to make your strokes more fluid.",
        "Imagine you're drawing with honey - smooth and flowing!"
    ],
    'consistency': [
        "Try to make all your strokes about the same size.",
        "Keep your letter proportions consistent throughout.",
        "Practice making your strokes more uniform.",
        "Think about keeping everything balanced and even!"
    ],
    'spacing': [
        "Make sure your strokes are well-spaced and not too crowded.",
        "Try to keep even spacing between different parts of the letter.",
        "Give your letter some breathing room.",
        "Imagine each part of the letter needs its own space!"
    ],
    'shape': [
        "Try to follow the letter shape more closely.",
        "Make sure your letter looks like the example.",
        "Practice the basic shape before adding details.",
        "Look at the example and try to match it exactly!"
    ],
    'pressure': [
        "Try to use consistent pressure throughout your strokes.",
        "Don't press too hard - gentle pressure works best.",
        "Keep your hand relaxed while writing.",
        "Practice with light, even pressure!"
    ]
}

@app.route("/analyze", methods=["POST"])
def analyze_stroke():
    """Analyze handwriting strokes with detailed feedback"""
    data = request.json.get("strokes", [])
    letter = request.json.get("letter", "")
    
    if not data:
        return jsonify({"error": "No stroke data received"}), 400

    try:
        # Extract features using the analyzer
        features = analyzer.extract_features(data)
        
        # Scale features
        features_scaled = analyzer.scaler.transform([features])
        
        # Get prediction
        prediction = analyzer.model.predict(features_scaled)[0]
        
        # Ensure prediction is between 0 and 1
        prediction = max(0, min(1, prediction))
        
        # Analyze specific aspects of the handwriting
        analysis = analyze_handwriting_quality(data, features, letter)
        
        # Adjust prediction based on shape accuracy - make shape more important
        shape_accuracy = analysis.get('shape_accuracy', 0.5)
        
        # If shape accuracy is very low (wrong letter), heavily penalize
        if shape_accuracy < 0.3:
            adjusted_prediction = shape_accuracy * 0.8  # Heavy penalty for wrong letter
        else:
            adjusted_prediction = (prediction * 0.4) + (shape_accuracy * 0.6)  # Shape is more important
        
        # Determine result category with more realistic thresholds
        if shape_accuracy < 0.3:
            result = "Wrong Letter"
            category = "needs_improvement"
        elif adjusted_prediction > 0.75:
            result = "Excellent"
            category = "excellent"
        elif adjusted_prediction > 0.55:
            result = "Good"
            category = "good"
        elif adjusted_prediction > 0.35:
            result = "Fair"
            category = "fair"
        else:
            result = "Needs Improvement"
            category = "needs_improvement"

        # Get random encouraging message
        message = random.choice(encouraging_messages[category])
        
        # Get letter tip
        tip = letter_tips.get(letter, "Keep practicing!") if letter else "Keep practicing!"
        
        # Get specific improvement suggestions
        improvement_suggestions = get_improvement_suggestions(analysis, adjusted_prediction)
        
        # Add specific feedback for wrong letters
        if shape_accuracy < 0.3:
            message = f"Oops! That doesn't look like the letter '{letter}'. Try drawing the correct letter!"
            tip = f"Remember: {letter} has a specific shape. Look at the example and try again!"
            improvement_suggestions = [
                f"Make sure you're drawing the letter '{letter}', not a different letter",
                "Look at the letter outline for guidance",
                "Take your time to form the correct shape"
            ]
        
        return jsonify({
            "result": result,
            "score": float(adjusted_prediction),
            "message": message,
            "tip": tip,
            "category": category,
            "analysis": analysis,
            "improvements": improvement_suggestions
        })
        
    except Exception as e:
        print(f"Error analyzing strokes: {e}")
        return jsonify({
            "result": "Needs Improvement",
            "score": 0.3,
            "message": "Let's try again! Take your time.",
            "tip": "Practice makes perfect!",
            "category": "needs_improvement",
            "analysis": {},
            "improvements": ["Try again with careful strokes"]
        })

def analyze_handwriting_quality(strokes, features, letter):
    """Analyze specific aspects of handwriting quality with letter-specific validation"""
    analysis = {}
    
    if not strokes:
        return analysis
    
    # Analyze stroke smoothness
    smoothness_scores = []
    for stroke in strokes:
        if len(stroke) < 3:
            continue
        
        stroke_points = np.array(stroke)
        angles = []
        
        for i in range(1, len(stroke_points) - 1):
            v1 = stroke_points[i] - stroke_points[i-1]
            v2 = stroke_points[i+1] - stroke_points[i]
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            avg_angle = np.mean(angles)
            # Lower angles = smoother strokes
            smoothness = max(0, 1 - avg_angle / np.pi)
            smoothness_scores.append(smoothness)
    
    analysis['smoothness'] = np.mean(smoothness_scores) if smoothness_scores else 0
    
    # Analyze consistency
    stroke_lengths = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        stroke_points = np.array(stroke)
        total_length = 0
        for i in range(1, len(stroke_points)):
            dist = np.linalg.norm(stroke_points[i] - stroke_points[i-1])
            total_length += dist
        stroke_lengths.append(total_length)
    
    if stroke_lengths:
        consistency = 1 - (np.std(stroke_lengths) / np.mean(stroke_lengths)) if np.mean(stroke_lengths) > 0 else 0
        analysis['consistency'] = max(0, consistency)
    else:
        analysis['consistency'] = 0
    
    # Analyze spatial distribution
    all_points = []
    for stroke in strokes:
        all_points.extend(stroke)
    
    if all_points:
        points = np.array(all_points)
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # Check if the letter uses appropriate space
        aspect_ratio = width / height if height > 0 else 1
        analysis['spacing'] = 1 - abs(aspect_ratio - 1)  # Closer to 1 is better
        
        # Analyze pressure variation (simulated)
        analysis['pressure'] = random.uniform(0.4, 0.9)  # In real app, this would come from actual pressure data
        
        # Letter-specific shape analysis
        analysis['shape_accuracy'] = analyze_letter_shape(strokes, letter, points)
    
    return analysis

def analyze_letter_shape(strokes, letter, points):
    """Analyze how well the drawn shape matches the expected letter"""
    if not strokes or len(strokes) == 0:
        return 0.0
    
    # Get bounding box
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize points to 0-1 range
    if width == 0 or height == 0:
        return 0.0
    
    normalized_points = []
    for stroke in strokes:
        normalized_stroke = []
        for point in stroke:
            norm_x = (point[0] - min_x) / width
            norm_y = (point[1] - min_y) / height
            normalized_stroke.append([norm_x, norm_y])
        normalized_points.append(normalized_stroke)
    
    # Letter-specific shape validation
    shape_score = 0.0
    
    if letter == 'A':
        # A should have: diagonal lines meeting at top, horizontal crossbar
        shape_score = validate_letter_a(normalized_points)
    elif letter == 'B':
        # B should have: vertical line with two curves
        shape_score = validate_letter_b(normalized_points)
    elif letter == 'C':
        # C should have: open curve
        shape_score = validate_letter_c(normalized_points)
    elif letter == 'D':
        # D should have: vertical line with curve
        shape_score = validate_letter_d(normalized_points)
    elif letter == 'E':
        # E should have: vertical line with three horizontals
        shape_score = validate_letter_e(normalized_points)
    elif letter == 'F':
        # F should have: vertical line with two horizontals
        shape_score = validate_letter_f(normalized_points)
    elif letter == 'G':
        # G should have: C with tail
        shape_score = validate_letter_g(normalized_points)
    elif letter == 'H':
        # H should have: two verticals with horizontal
        shape_score = validate_letter_h(normalized_points)
    elif letter == 'I':
        # I should have: vertical with top and bottom lines
        shape_score = validate_letter_i(normalized_points)
    elif letter == 'J':
        # J should have: curve with hook
        shape_score = validate_letter_j(normalized_points)
    elif letter == 'K':
        # K should have: vertical with two diagonals
        shape_score = validate_letter_k(normalized_points)
    elif letter == 'L':
        # L should have: vertical with bottom horizontal
        shape_score = validate_letter_l(normalized_points)
    elif letter == 'M':
        # M should have: mountain shape
        shape_score = validate_letter_m(normalized_points)
    elif letter == 'N':
        # N should have: two verticals with diagonal
        shape_score = validate_letter_n(normalized_points)
    elif letter == 'O':
        # O should have: circle/oval
        shape_score = validate_letter_o(normalized_points)
    elif letter == 'P':
        # P should have: vertical with top curve
        shape_score = validate_letter_p(normalized_points)
    elif letter == 'Q':
        # Q should have: O with tail
        shape_score = validate_letter_q(normalized_points)
    elif letter == 'R':
        # R should have: P with diagonal tail
        shape_score = validate_letter_r(normalized_points)
    elif letter == 'S':
        # S should have: snake curve
        shape_score = validate_letter_s(normalized_points)
    elif letter == 'T':
        # T should have: horizontal with vertical
        shape_score = validate_letter_t(normalized_points)
    elif letter == 'U':
        # U should have: curve with verticals
        shape_score = validate_letter_u(normalized_points)
    elif letter == 'V':
        # V should have: checkmark
        shape_score = validate_letter_v(normalized_points)
    elif letter == 'W':
        # W should have: double V
        shape_score = validate_letter_w(normalized_points)
    elif letter == 'X':
        # X should have: crossing diagonals
        shape_score = validate_letter_x(normalized_points)
    elif letter == 'Y':
        # Y should have: V with vertical
        shape_score = validate_letter_y(normalized_points)
    elif letter == 'Z':
        # Z should have: horizontal, diagonal, horizontal
        shape_score = validate_letter_z(normalized_points)
    else:
        # Default shape validation
        shape_score = 0.5
    
    return max(0.0, min(1.0, shape_score))

def validate_letter_a(strokes):
    """Validate letter A shape"""
    if len(strokes) < 2:
        return 0.2
    
    # Check for diagonal lines and horizontal crossbar
    has_diagonals = False
    has_crossbar = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        # Check if stroke is diagonal
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonals = True
        elif dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_crossbar = True
    
    score = 0.0
    if has_diagonals:
        score += 0.6
    if has_crossbar:
        score += 0.4
    
    return score

def validate_letter_b(strokes):
    """Validate letter B shape"""
    if len(strokes) < 2:
        return 0.2
    
    # Check for vertical line and curves
    has_vertical = False
    has_curves = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
        elif len(stroke) > 5:  # Likely a curve
            has_curves = True
    
    score = 0.0
    if has_vertical:
        score += 0.5
    if has_curves:
        score += 0.5
    
    return score

def validate_letter_c(strokes):
    """Validate letter C shape"""
    if len(strokes) < 1:
        return 0.1
    
    # C should be a single curved stroke, not straight lines
    has_curve = False
    has_vertical = False
    has_horizontal = False
    
    for stroke in strokes:
        if len(stroke) < 3:
            continue
        
        # Check if stroke is curved (not straight)
        start = stroke[0]
        end = stroke[-1]
        mid = stroke[len(stroke)//2]
        
        # Calculate if stroke curves
        dx1 = mid[0] - start[0]
        dy1 = mid[1] - start[1]
        dx2 = end[0] - mid[0]
        dy2 = end[1] - mid[1]
        
        # Check stroke direction
        total_dx = end[0] - start[0]
        total_dy = end[1] - start[1]
        
        # If mid point is significantly different from straight line
        if abs(dx1 - dx2) > 0.15 or abs(dy1 - dy2) > 0.15:
            has_curve = True
        
        # Check for straight lines (which C shouldn't have)
        if abs(total_dx) < 0.1 and abs(total_dy) > 0.3:  # Vertical
            has_vertical = True
        elif abs(total_dx) > 0.3 and abs(total_dy) < 0.1:  # Horizontal
            has_horizontal = True
    
    # C should have curve, not straight lines
    if has_vertical or has_horizontal:
        return 0.1  # Very low score for straight lines
    
    if has_curve:
        return 0.8  # Good score for curved stroke
    
    return 0.2  # Low score for unclear shape

def validate_letter_d(strokes):
    """Validate letter D shape"""
    if len(strokes) < 2:
        return 0.1
    
    has_vertical = False
    has_curve = False
    has_horizontal = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        # Check for vertical stroke
        if dx < 0.1 and dy > 0.3:
            has_vertical = True
        # Check for horizontal stroke (which D shouldn't have)
        elif dx > 0.3 and dy < 0.1:
            has_horizontal = True
        # Check for curve
        elif len(stroke) > 5:
            # Verify it's actually curved
            mid = stroke[len(stroke)//2]
            dx1 = mid[0] - start[0]
            dy1 = mid[1] - start[1]
            dx2 = end[0] - mid[0]
            dy2 = end[1] - mid[1]
            
            if abs(dx1 - dx2) > 0.1 or abs(dy1 - dy2) > 0.1:
                has_curve = True
    
    # D should have vertical and curve, not horizontal
    if has_horizontal:
        return 0.1  # Very low score for horizontal lines
    
    score = 0.0
    if has_vertical:
        score += 0.5
    if has_curve:
        score += 0.5
    
    return score

def validate_letter_e(strokes):
    """Validate letter E shape"""
    if len(strokes) < 3:
        return 0.2
    
    has_vertical = False
    has_horizontals = 0
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
        elif dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_horizontals += 1
    
    score = 0.0
    if has_vertical:
        score += 0.4
    score += min(0.6, has_horizontals * 0.2)
    
    return score

# Add similar validation functions for other letters...
def validate_letter_f(strokes):
    return validate_letter_e(strokes) * 0.8  # Similar to E but with fewer horizontals

def validate_letter_g(strokes):
    return validate_letter_c(strokes) * 0.9  # Similar to C

def validate_letter_h(strokes):
    if len(strokes) < 3:
        return 0.2
    
    verticals = 0
    has_horizontal = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            verticals += 1
        elif dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_horizontal = True
    
    score = 0.0
    if verticals >= 2:
        score += 0.6
    if has_horizontal:
        score += 0.4
    
    return score

def validate_letter_i(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_vertical = False
    has_dots = 0
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
        elif dx < 0.1 and dy < 0.1:  # Dot
            has_dots += 1
    
    score = 0.0
    if has_vertical:
        score += 0.6
    score += min(0.4, has_dots * 0.2)
    
    return score

def validate_letter_j(strokes):
    return validate_letter_i(strokes) * 0.9  # Similar to I

def validate_letter_k(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_vertical = False
    has_diagonals = 0
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
        elif dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonals += 1
    
    score = 0.0
    if has_vertical:
        score += 0.5
    score += min(0.5, has_diagonals * 0.25)
    
    return score

def validate_letter_l(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_vertical = False
    has_horizontal = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
        elif dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_horizontal = True
    
    score = 0.0
    if has_vertical:
        score += 0.6
    if has_horizontal:
        score += 0.4
    
    return score

def validate_letter_m(strokes):
    if len(strokes) < 3:
        return 0.2
    
    # Check for mountain-like pattern
    return 0.6  # Simplified validation

def validate_letter_n(strokes):
    if len(strokes) < 3:
        return 0.2
    
    verticals = 0
    has_diagonal = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx < 0.1 and dy > 0.3:  # Vertical stroke
            verticals += 1
        elif dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonal = True
    
    score = 0.0
    if verticals >= 2:
        score += 0.6
    if has_diagonal:
        score += 0.4
    
    return score

def validate_letter_o(strokes):
    if len(strokes) < 1:
        return 0.2
    
    # Check for circular/oval shape
    for stroke in strokes:
        if len(stroke) > 8:  # Long stroke likely to be a curve
            return 0.7
    
    return 0.3

def validate_letter_p(strokes):
    return validate_letter_b(strokes) * 0.8  # Similar to B

def validate_letter_q(strokes):
    return validate_letter_o(strokes) * 0.9  # Similar to O

def validate_letter_r(strokes):
    return validate_letter_p(strokes) * 0.9  # Similar to P

def validate_letter_s(strokes):
    if len(strokes) < 1:
        return 0.2
    
    # Check for curved stroke
    for stroke in strokes:
        if len(stroke) > 6:
            return 0.6
    
    return 0.3

def validate_letter_t(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_horizontal = False
    has_vertical = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_horizontal = True
        elif dx < 0.1 and dy > 0.3:  # Vertical stroke
            has_vertical = True
    
    score = 0.0
    if has_horizontal:
        score += 0.5
    if has_vertical:
        score += 0.5
    
    return score

def validate_letter_u(strokes):
    if len(strokes) < 2:
        return 0.2
    
    # Check for U-like curve
    return 0.6  # Simplified validation

def validate_letter_v(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_diagonals = 0
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonals += 1
    
    return min(1.0, has_diagonals * 0.5)

def validate_letter_w(strokes):
    return validate_letter_v(strokes) * 0.9  # Similar to V

def validate_letter_x(strokes):
    if len(strokes) < 2:
        return 0.2
    
    has_diagonals = 0
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonals += 1
    
    return min(1.0, has_diagonals * 0.5)

def validate_letter_y(strokes):
    return validate_letter_v(strokes) * 0.9  # Similar to V

def validate_letter_z(strokes):
    if len(strokes) < 3:
        return 0.2
    
    has_horizontals = 0
    has_diagonal = False
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        start = stroke[0]
        end = stroke[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > 0.2 and dy < 0.1:  # Horizontal stroke
            has_horizontals += 1
        elif dx > 0.1 and dy > 0.1:  # Diagonal stroke
            has_diagonal = True
    
    score = 0.0
    score += min(0.6, has_horizontals * 0.3)
    if has_diagonal:
        score += 0.4
    
    return score

def get_improvement_suggestions(analysis, prediction):
    """Get specific improvement suggestions based on analysis and prediction"""
    suggestions = []
    
    # Base suggestions on prediction score
    if prediction < 0.4:
        suggestions.append("Take your time and go slowly. Focus on smooth strokes.")
        suggestions.append("Try to follow the letter shape more carefully.")
    
    # Specific suggestions based on analysis
    if analysis.get('smoothness', 1) < 0.6:
        suggestions.append(random.choice(improvement_tips['smoothness']))
    
    if analysis.get('consistency', 1) < 0.6:
        suggestions.append(random.choice(improvement_tips['consistency']))
    
    if analysis.get('spacing', 1) < 0.6:
        suggestions.append(random.choice(improvement_tips['spacing']))
    
    if analysis.get('pressure', 1) < 0.6:
        suggestions.append(random.choice(improvement_tips['pressure']))
    
    # Shape-specific suggestions
    shape_accuracy = analysis.get('shape_accuracy', 1)
    if shape_accuracy < 0.5:
        suggestions.append("Try to match the letter shape more closely. Look at the dotted outline and follow it exactly!")
        suggestions.append("Make sure you're drawing the right letter. Check the letter name at the top!")
    
    # If no specific issues, provide general encouragement
    if not suggestions:
        if prediction > 0.7:
            suggestions.append("Excellent work! Keep practicing to maintain your skills!")
        elif prediction > 0.5:
            suggestions.append("Good progress! Try to make your strokes even smoother!")
        else:
            suggestions.append("Keep practicing! Remember to take your time and be patient!")
    
    return suggestions[:3]  # Limit to 3 suggestions

@app.route("/letter-info/<letter>", methods=["GET"])
def get_letter_info(letter):
    """Get educational information about a specific letter"""
    if not letter or len(letter) != 1 or not letter.isalpha():
        return jsonify({"error": "Invalid letter"}), 400
    
    letter = letter.upper()
    
    # Example words starting with the letter
    example_words = {
        'A': ["Apple", "Ant", "Airplane", "Alligator", "Arrow"],
        'B': ["Ball", "Bear", "Butterfly", "Book", "Bird"],
        'C': ["Cat", "Car", "Cake", "Cloud", "Cup"],
        'D': ["Dog", "Duck", "Dinosaur", "Doll", "Door"],
        'E': ["Elephant", "Egg", "Eagle", "Earth", "Eye"],
        'F': ["Fish", "Flower", "Frog", "Fire", "Fan"],
        'G': ["Giraffe", "Girl", "Goat", "Gift", "Game"],
        'H': ["House", "Hat", "Heart", "Horse", "Hand"],
        'I': ["Ice", "Ice cream", "Igloo", "Insect", "Ink"],
        'J': ["Jump", "Jelly", "Jellyfish", "Jacket", "Jam"],
        'K': ["Kite", "King", "Kangaroo", "Key", "Kitten"],
        'L': ["Lion", "Leaf", "Lamp", "Lemon", "Light"],
        'M': ["Moon", "Mouse", "Monkey", "Milk", "Map"],
        'N': ["Nest", "Night", "Nose", "Nurse", "Name"],
        'O': ["Orange", "Owl", "Ocean", "Octopus", "Onion"],
        'P': ["Pig", "Pizza", "Pencil", "Panda", "Plant"],
        'Q': ["Queen", "Quilt", "Question", "Quiet", "Quick"],
        'R': ["Rainbow", "Rabbit", "Robot", "Rose", "Rain"],
        'S': ["Sun", "Star", "Snake", "Ship", "Shoe"],
        'T': ["Tree", "Tiger", "Train", "Turtle", "Table"],
        'U': ["Umbrella", "Unicorn", "Up", "Under", "Use"],
        'V': ["Violin", "Vase", "Van", "Vegetable", "Voice"],
        'W': ["Water", "Wolf", "Window", "Worm", "Watch"],
        'X': ["Xylophone", "X-ray", "Box", "Fox", "Six"],
        'Y': ["Yellow", "Yacht", "Yogurt", "Yawn", "Year"],
        'Z': ["Zebra", "Zoo", "Zipper", "Zero", "Zoo"]
    }
    
    return jsonify({
        "letter": letter,
        "tip": letter_tips.get(letter, "Practice this letter carefully!"),
        "examples": example_words.get(letter, []),
        "sound": f"The letter {letter} makes different sounds in different words.",
        "difficulty": get_letter_difficulty(letter)
    })

def get_letter_difficulty(letter):
    """Get difficulty level for each letter"""
    easy_letters = ['O', 'L', 'I', 'T', 'H']
    medium_letters = ['A', 'E', 'F', 'P', 'U', 'V', 'W', 'X', 'Y', 'Z']
    hard_letters = ['B', 'C', 'D', 'G', 'J', 'K', 'M', 'N', 'Q', 'R', 'S']
    
    if letter in easy_letters:
        return "Easy"
    elif letter in medium_letters:
        return "Medium"
    else:
        return "Hard"

@app.route("/progress-summary", methods=["POST"])
def get_progress_summary():
    """Analyze overall progress and provide recommendations"""
    progress_data = request.json.get("progress", {})
    
    if not progress_data:
        return jsonify({"error": "No progress data received"}), 400
    
    # Calculate statistics
    total_letters = len(progress_data)
    completed_letters = sum(1 for p in progress_data.values() if p.get("score", 0) > 0.55)
    average_score = sum(p.get("score", 0) for p in progress_data.values()) / total_letters if total_letters > 0 else 0
    
    # Analyze progress by difficulty
    easy_progress = []
    medium_progress = []
    hard_progress = []
    
    for letter, data in progress_data.items():
        difficulty = get_letter_difficulty(letter)
        score = data.get("score", 0)
        
        if difficulty == "Easy":
            easy_progress.append(score)
        elif difficulty == "Medium":
            medium_progress.append(score)
        else:
            hard_progress.append(score)
    
    # Generate recommendations
    recommendations = []
    
    if average_score < 0.4:
        recommendations.append("Try practicing letters more slowly and carefully. Focus on smooth strokes.")
    elif average_score < 0.6:
        recommendations.append("Good progress! Try to make your strokes more consistent and smooth.")
    elif average_score < 0.8:
        recommendations.append("Excellent work! Try practicing more challenging letters.")
    else:
        recommendations.append("Outstanding! You're ready for more advanced writing exercises.")
    
    if len(easy_progress) > 0 and np.mean(easy_progress) < 0.6:
        recommendations.append("Practice the basic letters (O, L, I, T, H) to build confidence.")
    
    if len(hard_progress) > 0 and np.mean(hard_progress) < 0.5:
        recommendations.append("Take your time with complex letters like B, C, D, G, J, K, M, N, Q, R, S.")
    
    return jsonify({
        "total_letters": total_letters,
        "completed_letters": completed_letters,
        "average_score": average_score,
        "completion_percentage": (completed_letters / total_letters * 100) if total_letters > 0 else 0,
        "recommendations": recommendations,
        "difficulty_breakdown": {
            "easy": {
                "count": len(easy_progress),
                "average": np.mean(easy_progress) if easy_progress else 0
            },
            "medium": {
                "count": len(medium_progress),
                "average": np.mean(medium_progress) if medium_progress else 0
            },
            "hard": {
                "count": len(hard_progress),
                "average": np.mean(hard_progress) if hard_progress else 0
            }
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": analyzer.model is not None,
        "scaler_loaded": analyzer.scaler is not None
    })

if __name__ == "__main__":
    print("🚀 Starting Handwriting Analysis Server...")
    print("📊 Model Status:", "Loaded" if analyzer.model is not None else "Not loaded")
    print("🔧 Scaler Status:", "Loaded" if analyzer.scaler is not None else "Not loaded")
    app.run(debug=True, host='0.0.0.0', port=5000)