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

# Initialize Anvil server
anvil.server.connect("PSL-APP")

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

# NVIDIA API Configuration
NVIDIA_API_URL = "https://build.nvidia.com/moonshotai/kimi-k2.6"
NVIDIA_API_KEY = ""  # Will be configured in Anvil secrets

# PSL Scoring Weights
SCORING_WEIGHTS = {
    'facial_geometry': 0.35,
    'feature_proportions': 0.25,
    'skin_quality': 0.20,
    'dimorphism': 0.20
}

# Reference dataset for percentile calculation (simulated)
# In production, this would be populated with actual user data
REFERENCE_DATASET = []

def initialize_reference_dataset():
    """Initialize with some sample data for percentile calculation"""
    global REFERENCE_DATASET
    # Simulate a dataset of 1000 entries with normal distribution
    np.random.seed(42)
    REFERENCE_DATASET = list(np.random.normal(75, 15, 1000))
    # Ensure scores are between 0-100
    REFERENCE_DATASET = [max(0, min(100, score)) for score in REFERENCE_DATASET]

def calculate_percentile(score):
    """Calculate percentile based on reference dataset"""
    if not REFERENCE_DATASET:
        initialize_reference_dataset()
    
    count_below = sum(1 for s in REFERENCE_DATASET if s < score)
    percentile = (count_below / len(REFERENCE_DATASET)) * 100
    return percentile

def analyze_face_with_nvidia(image_data):
    """Send image to NVIDIA Kimi K2.6 for analysis"""
    try:
        # Prepare the image
        image = Image.open(BytesIO(image_data))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare the request payload
        payload = {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this face for PSL (Photo Scoring League) rating. 
                            Evaluate and score the following four categories on a scale of 0-100:
                            
                            1. Facial Geometry: Assess bone structure, jawline definition, cheekbone prominence, and overall facial harmony. Consider symmetry and proportions.
                            
                            2. Feature Proportions: Evaluate the balance and proportions between eyes, nose, mouth, and other facial features. Consider interocular distance, nose-to-mouth ratio, etc.
                            
                            3. Skin Quality: Assess skin clarity, texture, pore visibility, wrinkles, blemishes, and overall skin health.
                            
                            4. Dimorphism: Evaluate sexual dimorphism - how strongly the face expresses masculine or feminine traits. Higher scores indicate more pronounced dimorphic features.
                            
                            Return ONLY a JSON object with the following structure:
                            {
                                "facial_geometry": <score_0_100>,
                                "feature_proportions": <score_0_100>,
                                "skin_quality": <score_0_100>,
                                "dimorphism": <score_0_100>,
                                "analysis_details": {
                                    "facial_geometry_notes": "brief analysis",
                                    "feature_proportions_notes": "brief analysis", 
                                    "skin_quality_notes": "brief analysis",
                                    "dimorphism_notes": "brief analysis"
                                }
                            }
                            """
                        },
                        {
                            "type": "image",
                            "image": img_str
                        }
                    ]
                }
            ],
            "temperature": 0.1,
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
                return analysis
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
                else:
                    # Fallback to local analysis
                    return analyze_face_locally(image_data)
        else:
            print(f"NVIDIA API Error: {response.status_code} - {response.text}")
            # Fallback to local analysis
            return analyze_face_locally(image_data)
            
    except Exception as e:
        print(f"Error with NVIDIA API: {e}")
        # Fallback to local analysis
        return analyze_face_locally(image_data)

def analyze_face_locally(image_data):
    """Local fallback analysis using face_recognition and OpenCV"""
    try:
        # Load image
        image = face_recognition.load_image_file(BytesIO(image_data))
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if not face_landmarks_list:
            return {
                "error": "No face detected in the image",
                "facial_geometry": 0,
                "feature_proportions": 0,
                "skin_quality": 0,
                "dimorphism": 0
            }
        
        face_landmarks = face_landmarks_list[0]
        
        # Basic facial geometry analysis
        facial_geometry = analyze_facial_geometry(face_landmarks, image.shape)
        feature_proportions = analyze_feature_proportions(face_landmarks)
        skin_quality = analyze_skin_quality(image, face_landmarks)
        dimorphism = analyze_dimorphism(face_landmarks, image)
        
        return {
            "facial_geometry": facial_geometry,
            "feature_proportions": feature_proportions,
            "skin_quality": skin_quality,
            "dimorphism": dimorphism,
            "analysis_details": {
                "facial_geometry_notes": "Local analysis based on facial landmarks",
                "feature_proportions_notes": "Local analysis of feature ratios",
                "skin_quality_notes": "Local skin texture analysis",
                "dimorphism_notes": "Local dimorphism estimation"
            }
        }
        
    except Exception as e:
        print(f"Local analysis error: {e}")
        # Return default scores
        return {
            "facial_geometry": 50,
            "feature_proportions": 50,
            "skin_quality": 50,
            "dimorphism": 50,
            "analysis_details": {
                "facial_geometry_notes": "Default score due to analysis error",
                "feature_proportions_notes": "Default score due to analysis error",
                "skin_quality_notes": "Default score due to analysis error",
                "dimorphism_notes": "Default score due to analysis error"
            }
        }

def analyze_facial_geometry(landmarks, image_shape):
    """Analyze facial geometry using landmarks"""
    try:
        # Get key points
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
        
        # Calculate geometry score (0-100)
        symmetry_score = 100 - min(100, np.linalg.norm(left_eye_center - eye_midpoint) - np.linalg.norm(right_eye_center - eye_midpoint))
        jawline_score = min(100, (chin_width / max(1, chin_height)) * 20)
        
        # Combine scores
        geometry_score = (symmetry_score * 0.6) + (jawline_score * 0.4)
        return max(0, min(100, geometry_score))
        
    except:
        return 50

def analyze_feature_proportions(landmarks):
    """Analyze feature proportions"""
    try:
        # Get key points
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose_tip = np.array(landmarks['nose_tip'])
        mouth = np.array(landmarks['top_lip'] + landmarks['bottom_lip'])
        
        # Calculate distances
        eye_width_left = np.linalg.norm(left_eye[0] - left_eye[3])
        eye_width_right = np.linalg.norm(right_eye[0] - right_eye[3])
        interocular_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        nose_to_mouth = np.linalg.norm(np.mean(nose_tip, axis=0) - np.mean(mouth, axis=0))
        
        # Ideal ratios (approximate)
        ideal_interocular_ratio = 0.45  # of face width
        ideal_eye_width_ratio = 0.1  # of interocular distance
        
        # Calculate proportion score
        eye_symmetry = 100 - abs(eye_width_left - eye_width_right)
        eye_width_ratio = (eye_width_left / max(1, interocular_distance)) * 100
        
        proportion_score = (eye_symmetry * 0.4) + (eye_width_ratio * 0.3) + 50
        return max(0, min(100, proportion_score))
        
    except:
        return 50

def analyze_skin_quality(image, landmarks):
    """Analyze skin quality"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get face region
        chin = landmarks['chin']
        left_eyebrow = landmarks['left_eyebrow']
        right_eyebrow = landmarks['right_eyebrow']
        
        # Create mask from landmarks
        hull = cv2.convexHull(np.array(chin + left_eyebrow + right_eyebrow))
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        
        # Extract face region
        face_region = gray[mask == 255]
        
        if len(face_region) == 0:
            return 50
            
        # Calculate skin quality metrics
        mean_intensity = np.mean(face_region)
        std_intensity = np.std(face_region)
        
        # Lower std = smoother skin
        smoothness_score = 100 - (std_intensity * 2)
        brightness_score = min(100, (mean_intensity / 255) * 100)
        
        skin_score = (smoothness_score * 0.7) + (brightness_score * 0.3)
        return max(0, min(100, skin_score))
        
    except:
        return 50

def analyze_dimorphism(landmarks, image):
    """Analyze sexual dimorphism"""
    try:
        # Get key measurements
        chin = np.array(landmarks['chin'])
        jawline = np.array(landmarks['jawline'] if 'jawline' in landmarks else landmarks['chin'])
        
        # Calculate jawline angle (more angular = more masculine)
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
            
            # More angular jawline = higher dimorphism
            dimorphism_score = min(100, jawline_angularity * 2)
            return dimorphism_score
        else:
            return 50
            
    except:
        return 50

def calculate_overall_psl(facial_geometry, feature_proportions, skin_quality, dimorphism):
    """Calculate weighted overall PSL score"""
    overall = (
        facial_geometry * SCORING_WEIGHTS['facial_geometry'] +
        feature_proportions * SCORING_WEIGHTS['feature_proportions'] +
        skin_quality * SCORING_WEIGHTS['skin_quality'] +
        dimorphism * SCORING_WEIGHTS['dimorphism']
    )
    return overall

@anvil.server.callable
@anvil.server background_task
async def rate_face(image_data):
    """Main function to rate a face and return results"""
    try:
        # Analyze the face
        analysis = analyze_face_with_nvidia(image_data)
        
        if 'error' in analysis:
            return {
                'success': False,
                'error': analysis['error']
            }
        
        # Extract scores
        facial_geometry = float(analysis.get('facial_geometry', 50))
        feature_proportions = float(analysis.get('feature_proportions', 50))
        skin_quality = float(analysis.get('skin_quality', 50))
        dimorphism = float(analysis.get('dimorphism', 50))
        
        # Calculate overall score
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
            analysis_data=analysis
        )
        rating.save()
        
        # Update reference dataset
        REFERENCE_DATASET.append(overall_score)
        
        return {
            'success': True,
            'facial_geometry': facial_geometry,
            'feature_proportions': feature_proportions,
            'skin_quality': skin_quality,
            'dimorphism': dimorphism,
            'overall_psl_score': overall_score,
            'percentile': percentile,
            'analysis_details': analysis.get('analysis_details', {})
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@anvil.server.callable
def get_leaderboard(limit=10):
    """Get top ratings for leaderboard"""
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

@anvil.server.callable
def get_statistics():
    """Get overall statistics"""
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
        'avg_overall_score': np.mean(scores),
        'highest_score': np.max(scores),
        'lowest_score': np.min(scores)
    }

# Initialize reference dataset on startup
initialize_reference_dataset()
