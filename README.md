# PSL-APP - Photo Scoring League

A comprehensive face rating application that uses **NVIDIA's Kimi K2.6 AI model** to analyze and score photos **out of 10** with **3 AI-GENERATED, PERSONALIZED improvements** for each face.

## 🎯 Key Features

- **🤖 AI-Powered Analysis**: Uses NVIDIA Kimi K2.6 for advanced face analysis
- **📊 10-Point Rating System**: Each category scored from 0-10 (10 being perfect)
- **💡 3 AI-Generated Improvements**: **Personalized, specific suggestions** based on YOUR exact face
- **⚖️ Four Category Scores**: Facial geometry, feature proportions, skin quality, dimorphism
- **📈 Weighted Overall Score**: Combines category scores with customizable weights
- **🏆 Percentile Ranking**: Shows where you stand relative to other users
- **🚀 No Account Required**: Upload and analyze photos instantly
- **🎨 Black & Gold Theme**: Elegant, professional styling

## 🎯 AI-Generated Improvements

The **NVIDIA Kimi K2.6 AI** analyzes your **specific face** and generates **3 personalized improvements** that are:

- **🎯 Specific to YOUR face** - Not generic advice
- **📋 Actionable** - Clear steps you can take
- **🏆 Ranked by Priority** - Most impactful first
- **🏷️ Categorized** - Linked to specific rating categories
- **💬 Honest & Constructive** - Based on actual analysis

### Example AI-Generated Improvements:
```
1. [Facial Geometry] Address facial asymmetry through strategic lighting and camera angles to create more balanced appearance

2. [Feature Proportions] Balance eye-to-nose ratio with strategic makeup or grooming to create more harmonious feature ratios  

3. [Skin Quality] Improve skin texture through proper skincare routine and hydration to reduce visible pores and imperfections
```

## Scoring Categories (All Out of 10)

1. **Facial Geometry (35% weight)**: 
   - Bone structure, jawline definition, cheekbone prominence
   - Overall facial harmony and symmetry

2. **Feature Proportions (25% weight)**:
   - Balance between eyes, nose, mouth
   - Interocular distance, nose-to-mouth ratio

3. **Skin Quality (20% weight)**:
   - Skin clarity, texture, pore visibility
   - Wrinkles, blemishes, overall skin health

4. **Dimorphism (20% weight)**:
   - How strongly the face expresses masculine/feminine traits
   - Higher scores = more pronounced dimorphic features

## Requirements

- Python 3.10+
- Anvil (for deployment)
- NVIDIA API Key (for Kimi K2.6 access)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **NVIDIA API Key**: 
   - Sign up at [NVIDIA Build](https://build.nvidia.com/)
   - Get an API key for Kimi K2.6
   - Add it to your Anvil app secrets as `NVIDIA_API_KEY`

2. **Anvil Deployment**:
   - Create a new Anvil app
   - Upload all files from this repository
   - Set the runtime to Python 3.10

## Usage

1. Upload a clear front-facing photo
2. Click "Analyze Photo"
3. View your **scores out of 10** in all four categories
4. See your **overall PSL score** and **percentile ranking**
5. **Read your 3 AI-GENERATED improvements** - these are **personalized just for you**
6. Compare with others on the leaderboard

## 📋 Output Format

### Ratings (0-10)
```json
{
  "ratings": {
    "facial_geometry": 8.5,
    "feature_proportions": 7.2,
    "skin_quality": 9.0,
    "dimorphism": 6.8
  },
  "overall_psl_score": 7.8,
  "percentile": 85
}
```

### AI-Generated Improvements (3 specific suggestions)
```json
{
  "improvements": [
    {
      "category": "Facial Geometry",
      "improvement": "Address the slight asymmetry in your jawline through strategic lighting from the left side",
      "priority": 1
    },
    {
      "category": "Feature Proportions", 
      "improvement": "Your eyes are slightly close together - use hairstyle framing to create better balance",
      "priority": 2
    },
    {
      "category": "Skin Quality",
      "improvement": "Focus on reducing pore visibility on your forehead with a targeted skincare routine",
      "priority": 3
    }
  ]
}
```

## 🔧 How the AI Generates Improvements

The NVIDIA Kimi K2.6 model:
1. **Analyzes your specific face** in detail
2. **Identifies unique characteristics** and areas for improvement
3. **Generates personalized suggestions** based on what it actually sees
4. **Ranks improvements by impact** - most beneficial first
5. **Categorizes each suggestion** to the relevant rating category

This is **NOT generic advice** - each improvement is **AI-generated specifically for YOUR face**.

## File Structure

```
PSL-APP/
├── anvil.yaml           # Anvil configuration
├── theme.yaml           # Black & Gold theme
├── config.yaml          # App configuration
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── client_code/
│   ├── __init__.py
│   ├── MainForm.py      # Main form logic
│   └── MainForm.yaml    # Main form UI
└── server_code/
    └── app/
        ├── __init__.py  # Server logic (NVIDIA K2.6 integration)
        └── server.py    # Server entry point
```

## Customization

### Adjusting Scoring Weights

Modify the weights in `server_code/app/__init__.py`:

```python
SCORING_WEIGHTS = {
    'facial_geometry': 0.35,
    'feature_proportions': 0.25,
    'skin_quality': 0.20,
    'dimorphism': 0.20
}
```

## Fallback Analysis

If the NVIDIA API is unavailable, the app falls back to local analysis using:
- OpenCV for image processing
- face-recognition for facial landmark detection
- Custom algorithms for each scoring category (0-10)
- **AI-style improvement generation** based on actual facial measurements

## Performance

- Analysis typically completes in 5-10 seconds
- Results are cached for quick re-analysis
- Leaderboard updates in real-time

## Privacy

- No personal data is collected
- Images are processed and discarded after analysis
- All analysis happens client-side or via secure API calls

## License

This project is proprietary. All rights reserved.

## Support

For issues or questions, please contact the development team.

---

**PSL - Photo Scoring League**

*Quick, structured face-rating snapshot with **AI-generated 10-point scores** and **3 personalized improvements**, powered by NVIDIA Kimi K2.6*
