# PSL Application Server Code
# This module contains the server-side logic for the PSL rating application

import anvil.server
import anvil.media
from anvil import tables
import numpy as np
import cv2
import requests
import base64
import json
from io import BytesIO
from PIL import Image
import face_recognition
import time
import os

# NVIDIA API Configuration
NVIDIA_API_URL = "https://build.nvidia.com/moonshotai/kimi-k2.6"
# Get API key from Anvil secrets
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')

# PSL Scoring Weights
SCORING_WEIGHTS = {
    'facial_geometry': 0.35,
    'feature_proportions': 0.25,
    'skin_quality': 0.20,
    'dimorphism': 0.20
}

# Reference dataset for percentile calculation (simulated)
REFERENCE_DATASET = []

def initialize_reference_dataset():
    """Initialize with some sample data for percentile calculation"""
    global REFERENCE_DATASET
    np.random.seed(42)
    REFERENCE_DATASET = list(np.random.normal(7.5, 1.5, 1000))
    REFERENCE_DATASET = [max(0, min(10, score)) for score in REFERENCE_DATASET]

def calculate_percentile(score):
    """Calculate percentile based on reference dataset"""
    if not REFERENCE_DATASET:
        initialize_reference_dataset()
    
    count_below = sum(1 for s in REFERENCE_DATASET if s < score)
    percentile = (count_below / len(REFERENCE_DATASET)) * 100
    return percentile

def analyze_face_with_nvidia(image_data):
    """Send image to NVIDIA Kimi K2.6 for analysis with AI-generated improvements"""
    try:
        # Check if API key is available
        if not NVIDIA_API_KEY or NVIDIA_API_KEY == '':
            print("NVIDIA API key not configured. Falling back to local analysis.")
            return analyze_face_locally(image_data)
        
        image = Image.open(BytesIO(image_data))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare the request payload - ENHANCED PROMPT FOR AI-GENERATED IMPROVEMENTS
        payload = {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert facial analysis AI for PSL (Photo Scoring League). 
                            Analyze this face and provide:
                            
                            1. RATINGS: Rate EACH of the following four categories on a scale of 0-10 (10 being perfect, 0 being poor):
                               - facial_geometry: Bone structure, jawline definition, cheekbone prominence, facial harmony, symmetry
                               - feature_proportions: Balance between eyes, nose, mouth, interocular distance, nose-to-mouth ratio
                               - skin_quality: Clarity, texture, pore visibility, wrinkles, blemishes, overall skin health
                               - dimorphism: How strongly the face expresses masculine/feminine traits
                            
                            2. IMPROVEMENTS: Provide EXACTLY 3 AI-GENERATED, SPECIFIC, PERSONALIZED improvements for THIS EXACT FACE.
                               Each improvement must be:
                               - Based on what you actually see in this specific face
                               - Specific and actionable (not generic advice)
                               - Ranked by priority (1 = most impactful improvement)
                               - Include the category it relates to
                               - Be honest and constructive
                               - Consider the person's unique facial features
                            
                            Return ONLY a JSON object with this EXACT structure:
                            {
                                "ratings": {
                                    "facial_geometry": <score_0_10>,
                                    "feature_proportions": <score_0_10>,
                                    "skin_quality": <score_0_10>,
                                    "dimorphism": <score_0_10>
                                },
                                "improvements": [
                                    {"category": "<specific_category>", "improvement": "<ai_generated_specific_improvement_for_this_face>", "priority": 1},
                                    {"category": "<specific_category>", "improvement": "<ai_generated_specific_improvement_for_this_face>", "priority": 2},
                                    {"category": "<specific_category>", "improvement": "<ai_generated_specific_improvement_for_this_face>", "priority": 3}
                                ]
                            }
                            
                            IMPORTANT: The improvements MUST be AI-generated based on THIS SPECIFIC FACE, not generic advice.
                            Be specific about what you see and what would improve it."""
                        },
                        {
                            "type": "image",
                            "image": img_str
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NVIDIA_API_KEY}"
        }
        
        # Make the request
        response = requests.post(
            f"{NVIDIA_API_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            
            # Parse the JSON response
            try:
                analysis = json.loads(content)
                # Validate the structure
                if 'ratings' in analysis and 'improvements' in analysis:
                    if len(analysis.get('improvements', [])) == 3:
                        return analysis
                    else:
                        # Fix if not exactly 3 improvements
                        analysis['improvements'] = analysis.get('improvements', [])[:3]
                        while len(analysis['improvements']) < 3:
                            analysis['improvements'].append({
                                "category": "General",
                                "improvement": "Use better lighting for clearer analysis",
                                "priority": len(analysis['improvements']) + 1
                            })
                        return analysis
                else:
                    return analyze_face_locally(image_data)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    # Ensure we have 3 improvements
                    if len(analysis.get('improvements', [])) < 3:
                        analysis['improvements'] = analysis.get('improvements', [])[:3]
                        while len(analysis['improvements']) < 3:
                            analysis['improvements'].append({
                                "category": "General",
                                "improvement": "Use better lighting for clearer analysis",
                                "priority": len(analysis['improvements']) + 1
                            })
                    return analysis
                else:
                    return analyze_face_locally(image_data)
        else:
            print(f"NVIDIA API Error: {response.status_code} - {response.text}")
            return analyze_face_locally(image_data)
            
    except Exception as e:
        print(f"Error with NVIDIA API: {e}")
        return analyze_face_locally(image_data)

def analyze_face_locally(image_data):
    """Local fallback analysis using face_recognition and OpenCV"""
    try:
        image = face_recognition.load_image_file(BytesIO(image_data))
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if not face_landmarks_list:
            return {
                "error": "No face detected in the image",
                "ratings": {
                    "facial_geometry": 0,
                    "feature_proportions": 0,
                    "skin_quality": 0,
                    "dimorphism": 0
                },
                "improvements": [
                    {"category": "General", "improvement": "Ensure face is clearly visible in the photo", "priority": 1},
                    {"category": "General", "improvement": "Use better lighting to help the AI analyze your features", "priority": 2},
                    {"category": "General", "improvement": "Remove obstructions from face for accurate analysis", "priority": 3}
                ]
            }
        
        face_landmarks = face_landmarks_list[0]
        
        # Basic facial geometry analysis (scale 0-10)
        facial_geometry = analyze_facial_geometry(face_landmarks, image.shape) / 10
        feature_proportions = analyze_feature_proportions(face_landmarks) / 10
        skin_quality = analyze_skin_quality(image, face_landmarks) / 10
        dimorphism = analyze_dimorphism(face_landmarks, image) / 10
        
        # Generate AI-style improvements based on scores
        improvements = generate_ai_improvements(
            facial_geometry, feature_proportions, skin_quality, dimorphism, face_landmarks, image
        )
        
        return {
            "ratings": {
                "facial_geometry": facial_geometry,
                "feature_proportions": feature_proportions,
                "skin_quality": skin_quality,
                "dimorphism": dimorphism
            },
            "improvements": improvements,
            "analysis_details": {
                "facial_geometry_notes": "Local analysis based on facial landmarks",
                "feature_proportions_notes": "Local analysis of feature ratios",
                "skin_quality_notes": "Local skin texture analysis",
                "dimorphism_notes": "Local dimorphism estimation"
            }
        }
        
    except Exception as e:
        print(f"Local analysis error: {e}")
        return {
            "ratings": {
                "facial_geometry": 5.0,
                "feature_proportions": 5.0,
                "skin_quality": 5.0,
                "dimorphism": 5.0
            },
            "improvements": [
                {"category": "General", "improvement": "Ensure clear, well-lit photo for accurate AI analysis", "priority": 1},
                {"category": "General", "improvement": "Face should be front-facing and clearly visible", "priority": 2},
                {"category": "General", "improvement": "Avoid heavy filters that obscure natural features", "priority": 3}
            ],
            "analysis_details": {
                "facial_geometry_notes": "Default score due to analysis error",
                "feature_proportions_notes": "Default score due to analysis error",
                "skin_quality_notes": "Default score due to analysis error",
                "dimorphism_notes": "Default score due to analysis error"
            }
        }

def generate_ai_improvements(fg, fp, sq, dim, landmarks, image):
    """Generate AI-style improvements based on actual facial analysis"""
    improvements = []
    
    # Analyze specific facial characteristics
    try:
        # Get facial measurements
        chin = landmarks.get('chin', [])
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        nose_tip = landmarks.get('nose_tip', [])
        
        # Calculate asymmetry
        if left_eye and right_eye:
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_midpoint = (left_eye_center + right_eye_center) / 2
            asymmetry = np.linalg.norm(left_eye_center - eye_midpoint) - np.linalg.norm(right_eye_center - eye_midpoint)
        else:
            asymmetry = 0
        
        # Jawline analysis
        if chin:
            chin_width = np.linalg.norm(np.array(chin[0]) - np.array(chin[-1]))
            jawline_defined = chin_width > 100  # Arbitrary threshold
        else:
            jawline_defined = False
        
        # Generate improvements based on actual analysis
        if fg < 7:
            if asymmetry > 5:
                improvements.append({
                    "category": "Facial Geometry",
                    "improvement": "Address facial asymmetry through strategic lighting and camera angles to create more balanced appearance",
                    "priority": 1
                })
            elif not jawline_defined:
                improvements.append({
                    "category": "Facial Geometry",
                    "improvement": "Enhance jawline definition through targeted exercises or contouring techniques to improve bone structure visibility",
                    "priority": 1
                })
            else:
                improvements.append({
                    "category": "Facial Geometry",
                    "improvement": "Refine cheekbone prominence with proper lighting to accentuate your natural bone structure",
                    "priority": 1
                })
        
        if fp < 7:
            if left_eye and right_eye:
                eye_width_left = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[3]))
                eye_width_right = np.linalg.norm(np.array(right_eye[0]) - np.array(right_eye[3]))
                if abs(eye_width_left - eye_width_right) > 5:
                    improvements.append({
                        "category": "Feature Proportions",
                        "improvement": "Balance eye proportions through strategic makeup or grooming to create more harmonious feature ratios",
                        "priority": 2 if len(improvements) > 0 else 1
                    })
                else:
                    improvements.append({
                        "category": "Feature Proportions",
                        "improvement": "Optimize nose-to-mouth proportion with contouring to achieve better facial feature balance",
                        "priority": 2 if len(improvements) > 0 else 1
                    })
            else:
                improvements.append({
                    "category": "Feature Proportions",
                    "improvement": "Improve overall feature harmony through strategic framing and angle selection",
                    "priority": 2 if len(improvements) > 0 else 1
                })
        
        if sq < 7:
            # Analyze skin texture
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if chin and left_eye:
                hull = cv2.convexHull(np.array(chin + landmarks.get('left_eyebrow', []) + landmarks.get('right_eyebrow', [])))
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [hull], -1, 255, -1)
                face_region = gray[mask == 255]
                if len(face_region) > 0:
                    std_intensity = np.std(face_region)
                    if std_intensity > 30:
                        improvements.append({
                            "category": "Skin Quality",
                            "improvement": "Improve skin texture through proper skincare routine and hydration to reduce visible pores and imperfections",
                            "priority": 3 if len(improvements) >= 2 else (2 if len(improvements) > 0 else 1)
                        })
                    else:
                        improvements.append({
                            "category": "Skin Quality",
                            "improvement": "Enhance skin clarity with targeted treatments to address blemishes and achieve smoother complexion",
                            "priority": 3 if len(improvements) >= 2 else (2 if len(improvements) > 0 else 1)
                        })
                else:
                    improvements.append({
                        "category": "Skin Quality",
                        "improvement": "Maintain skin health with consistent skincare to preserve current quality",
                        "priority": 3 if len(improvements) >= 2 else (2 if len(improvements) > 0 else 1)
                    })
            else:
                improvements.append({
                    "category": "Skin Quality",
                    "improvement": "Focus on skin clarity and texture improvement through professional skincare regimen",
                    "priority": 3 if len(improvements) >= 2 else (2 if len(improvements) > 0 else 1)
                })
        
        if dim < 7:
            improvements.append({
                "category": "Dimorphism",
                "improvement": "Enhance masculine/feminine features through strategic styling, grooming, and accessory choices",
                "priority": 3 if len(improvements) >= 2 else (2 if len(improvements) > 0 else 1)
            })
        
        # If we have fewer than 3 improvements, add more based on scores
        all_categories = [
            (fg, "Facial Geometry", [
                "Define jawline through targeted exercises or contouring",
                "Improve cheekbone prominence with proper lighting and angles",
                "Work on facial symmetry through posture and camera positioning"
            ]),
            (fp, "Feature Proportions", [
                "Balance eye-to-nose ratio with strategic makeup or grooming",
                "Improve nose-to-mouth proportion through contouring",
                "Enhance interocular distance perception with hairstyle framing"
            ]),
            (sq, "Skin Quality", [
                "Improve skin clarity with proper skincare routine",
                "Reduce pore visibility with professional treatments",
                "Enhance skin texture with hydration and exfoliation"
            ]),
            (dim, "Dimorphism", [
                "Enhance masculine/feminine features through styling",
                "Accentuate jawline definition for stronger dimorphism",
                "Use clothing and accessories to emphasize natural features"
            ])
        ]
        
        # Sort by score (lowest first)
        sorted_categories = sorted(all_categories, key=lambda x: x[0])
        
        # Add improvements until we have 3
        for score, category, suggestions in sorted_categories:
            if len(improvements) >= 3:
                break
            if score < 8:  # Only suggest improvements for scores below 8
                for suggestion in suggestions:
                    if len(improvements) >= 3:
                        break
                    if suggestion not in [imp["improvement"] for imp in improvements]:
                        improvements.append({
                            "category": category,
                            "improvement": suggestion,
                            "priority": len(improvements) + 1
                        })
        
        # If still not 3, add general improvements
        general_suggestions = [
            "Use better lighting to highlight facial features",
            "Experiment with different camera angles for optimal results",
            "Ensure consistent photo quality across analyses"
        ]
        while len(improvements) < 3:
            for suggestion in general_suggestions:
                if suggestion not in [imp["improvement"] for imp in improvements]:
                    improvements.append({
                        "category": "General",
                        "improvement": suggestion,
                        "priority": len(improvements) + 1
                    })
                    break
        
        # Sort by priority
        improvements = sorted(improvements, key=lambda x: x["priority"])
        
    except Exception as e:
        print(f"Error generating improvements: {e}")
        # Fallback to generic improvements
        improvements = [
            {"category": "General", "improvement": "Use better lighting for clearer facial analysis", "priority": 1},
            {"category": "General", "improvement": "Ensure face is front-facing and clearly visible", "priority": 2},
            {"category": "General", "improvement": "Avoid heavy filters that obscure natural features", "priority": 3}
        ]
    
    return improvements[:3]

def analyze_facial_geometry(landmarks, image_shape):
    """Analyze facial geometry using landmarks (0-10 scale)"""
    try:
        chin = landmarks['chin']
        left_eyebrow = landmarks['left_eyebrow']
        right_eyebrow = landmarks['right_eyebrow']
        nose_bridge = landmarks['nose_bridge']
        nose_tip = landmarks['nose_tip']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Calculate symmetry
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_midpoint = (left_eye_center + right_eye_center) / 2
        
        # Nose symmetry
        nose_points = np.array(nose_bridge + nose_tip)
        nose_center = np.mean(nose_points, axis=0)
        
        # Jawline definition
        chin_width = np.linalg.norm(np.array(chin[0]) - np.array(chin[-1]))
        chin_height = np.linalg.norm(np.array(chin[0]) - np.array(chin[len(chin)//2]))
        
        # Calculate geometry score (0-10)
        symmetry_score = 10 - min(10, abs(np.linalg.norm(left_eye_center - eye_midpoint) - np.linalg.norm(right_eye_center - eye_midpoint)) * 0.2)
        jawline_score = min(10, (chin_width / max(1, chin_height)) * 2)
        
        # Combine scores
        geometry_score = (symmetry_score * 0.6) + (jawline_score * 0.4)
        return max(0, min(10, geometry_score))
        
    except Exception as e:
        print(f"Error in facial geometry analysis: {e}")
        return 5.0

def analyze_feature_proportions(landmarks):
    """Analyze feature proportions (0-10 scale)"""
    try:
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose_tip = np.array(landmarks['nose_tip'])
        mouth = np.array(landmarks['top_lip'] + landmarks['bottom_lip'])
        
        # Calculate distances
        eye_width_left = np.linalg.norm(left_eye[0] - left_eye[3])
        eye_width_right = np.linalg.norm(right_eye[0] - right_eye[3])
        interocular_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        nose_to_mouth = np.linalg.norm(np.mean(nose_tip, axis=0) - np.mean(mouth, axis=0))
        
        # Ideal ratios
        ideal_interocular_ratio = 0.45
        ideal_eye_width_ratio = 0.1
        
        # Calculate proportion score (0-10)
        eye_symmetry = 10 - abs(eye_width_left - eye_width_right) * 0.2
        eye_width_ratio = (eye_width_left / max(1, interocular_distance)) * 10
        
        proportion_score = (eye_symmetry * 0.4) + (eye_width_ratio * 0.3) + 5
        return max(0, min(10, proportion_score))
        
    except Exception as e:
        print(f"Error in feature proportions analysis: {e}")
        return 5.0

def analyze_skin_quality(image, landmarks):
    """Analyze skin quality (0-10 scale)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        chin = landmarks['chin']
        left_eyebrow = landmarks['left_eyebrow']
        right_eyebrow = landmarks['right_eyebrow']
        
        hull = cv2.convexHull(np.array(chin + left_eyebrow + right_eyebrow))
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        
        face_region = gray[mask == 255]
        
        if len(face_region) == 0:
            return 5.0
            
        mean_intensity = np.mean(face_region)
        std_intensity = np.std(face_region)
        
        smoothness_score = 10 - (std_intensity * 0.2)
        brightness_score = min(10, (mean_intensity / 255) * 10)
        
        skin_score = (smoothness_score * 0.7) + (brightness_score * 0.3)
        return max(0, min(10, skin_score))
        
    except Exception as e:
        print(f"Error in skin quality analysis: {e}")
        return 5.0

def analyze_dimorphism(landmarks, image):
    """Analyze sexual dimorphism (0-10 scale)"""
    try:
        jawline = np.array(landmarks['jawline'] if 'jawline' in landmarks else landmarks['chin'])
        
        if len(jawline) >= 3:
            vectors = []
            for i in range(len(jawline) - 1):
                vectors.append(jawline[i+1] - jawline[i])
            
            angles = []
            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i+1]
                cosine = np.dot(v1, v2) / (max(1, np.linalg.norm(v1)) * max(1, np.linalg.norm(v2)))
                angles.append(np.arccos(cosine))
            
            avg_angle = np.mean(angles)
            jawline_angularity = np.degrees(avg_angle)
            
            dimorphism_score = min(10, jawline_angularity * 0.2)
            return dimorphism_score
        else:
            return 5.0
            
    except Exception as e:
        print(f"Error in dimorphism analysis: {e}")
        return 5.0

def calculate_overall_psl(facial_geometry, feature_proportions, skin_quality, dimorphism):
    """Calculate weighted overall PSL score (0-10)"""
    overall = (
        facial_geometry * SCORING_WEIGHTS['facial_geometry'] +
        feature_proportions * SCORING_WEIGHTS['feature_proportions'] +
        skin_quality * SCORING_WEIGHTS['skin_quality'] +
        dimorphism * SCORING_WEIGHTS['dimorphism']
    )
    return overall

# Database schema for storing ratings and user data
@tables.in_app
class UserRatings(tables.Table):
    user_id = tables.Column(tables.StringType(), unique=True)
    email = tables.Column(tables.StringType(), unique=True)
    created_at = tables.Column(tables.DateTimeType())
    total_ratings = tables.Column(tables.NumberType())
    avg_overall_score = tables.Column(tables.NumberType())

@tables.in_app  
class FaceRating(tables.Table):
    user_id = tables.Column(tables.StringType())
    image_data = tables.Column(tables.BlobType())
    facial_geometry_score = tables.Column(tables.NumberType())
    feature_proportions_score = tables.Column(tables.NumberType())
    skin_quality_score = tables.Column(tables.NumberType())
    dimorphism_score = tables.Column(tables.NumberType())
    overall_psl_score = tables.Column(tables.NumberType())
    percentile = tables.Column(tables.NumberType())
    timestamp = tables.Column(tables.DateTimeType())
    analysis_data = tables.Column(tables.JsonType())
    improvements = tables.Column(tables.JsonType())

@anvil.server.callable
@anvil.server.background_task
async def rate_face(image_data):
    """Main function to rate a face and return results with AI-generated improvements"""
    try:
        analysis = analyze_face_with_nvidia(image_data)
        
        if 'error' in analysis:
            return {
                'success': False,
                'error': analysis['error']
            }
        
        # Extract ratings (0-10)
        ratings = analysis.get('ratings', {})
        facial_geometry = float(ratings.get('facial_geometry', 5.0))
        feature_proportions = float(ratings.get('feature_proportions', 5.0))
        skin_quality = float(ratings.get('skin_quality', 5.0))
        dimorphism = float(ratings.get('dimorphism', 5.0))
        
        # Extract AI-generated improvements
        improvements = analysis.get('improvements', [
            {"category": "General", "improvement": "Use better lighting for clearer analysis", "priority": 1},
            {"category": "General", "improvement": "Ensure face is clearly visible", "priority": 2},
            {"category": "General", "improvement": "Avoid heavy filters", "priority": 3}
        ])
        
        # Ensure we have exactly 3 improvements
        if len(improvements) < 3:
            improvements = improvements[:3]
            while len(improvements) < 3:
                improvements.append({
                    "category": "General",
                    "improvement": f"Improvement suggestion {len(improvements) + 1}",
                    "priority": len(improvements) + 1
                })
        
        # Calculate overall score (0-10)
        overall_score = calculate_overall_psl(
            facial_geometry, feature_proportions, skin_quality, dimorphism
        )
        
        # Calculate percentile
        percentile = calculate_percentile(overall_score)
        
        # Store the rating
        rating = FaceRating(
            user_id="anonymous",
            image_data=image_data,
            facial_geometry_score=facial_geometry,
            feature_proportions_score=feature_proportions,
            skin_quality_score=skin_quality,
            dimorphism_score=dimorphism,
            overall_psl_score=overall_score,
            percentile=percentile,
            timestamp=time.time(),
            analysis_data=analysis,
            improvements=improvements
        )
        rating.save()
        
        # Update reference dataset
        REFERENCE_DATASET.append(overall_score)
        
        return {
            'success': True,
            'ratings': {
                'facial_geometry': facial_geometry,
                'feature_proportions': feature_proportions,
                'skin_quality': skin_quality,
                'dimorphism': dimorphism
            },
            'overall_psl_score': overall_score,
            'percentile': percentile,
            'improvements': improvements,
            'analysis_details': analysis.get('analysis_details', {})
        }
        
    except Exception as e:
        print(f"Error in rate_face: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@anvil.server.callable
def get_leaderboard(limit=10):
    """Get top ratings for leaderboard"""
    try:
        ratings = FaceRating.search(
            tables.sort_by('overall_psl_score', descending=True),
            tables.limit(limit)
        )
        
        leaderboard = []
        for rating in ratings:
            leaderboard.append({
                'overall_psl_score': rating['overall_psl_score'],
                'percentile': rating['percentile'],
                'timestamp': rating['timestamp']
            })
        
        return leaderboard
    except Exception as e:
        print(f"Error in get_leaderboard: {e}")
        return []

@anvil.server.callable
def get_statistics():
    """Get overall statistics"""
    try:
        all_ratings = FaceRating.search()
        
        if len(all_ratings) == 0:
            return {
                'total_ratings': 0,
                'avg_overall_score': 0,
                'highest_score': 0,
                'lowest_score': 0
            }
        
        scores = [r['overall_psl_score'] for r in all_ratings]
        
        return {
            'total_ratings': len(all_ratings),
            'avg_overall_score': float(np.mean(scores)),
            'highest_score': float(np.max(scores)),
            'lowest_score': float(np.min(scores))
        }
    except Exception as e:
        print(f"Error in get_statistics: {e}")
        return {
            'total_ratings': 0,
            'avg_overall_score': 0,
            'highest_score': 0,
            'lowest_score': 0
        }

# Initialize reference dataset on startup
initialize_reference_dataset()
