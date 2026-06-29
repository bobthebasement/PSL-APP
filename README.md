# PSL-APP - Photo Scoring League

A comprehensive face rating application that uses NVIDIA's Kimi K2.6 AI model to analyze and score photos **out of 10** with **3 specific improvements** for each face.

## Features

- **AI-Powered Analysis**: Uses NVIDIA Kimi K2.6 for advanced face analysis
- **10-Point Rating System**: Each category scored from 0-10 (10 being perfect)
- **Four Category Scores**: Detailed scoring in facial geometry, feature proportions, skin quality, and dimorphism
- **3 Personalized Improvements**: AI provides 3 specific, actionable improvements for your face
- **Weighted Overall Score**: Combines category scores with customizable weights
- **Percentile Ranking**: Shows where you stand relative to other users
- **No Account Required**: Upload and analyze photos instantly
- **Black & Gold Theme**: Elegant, professional styling
- **Responsive Design**: Works on desktop and mobile devices

## Scoring Categories (All Out of 10)

1. **Facial Geometry (35% weight)**: Assesses bone structure, jawline definition, cheekbone prominence, and overall facial harmony. Consider symmetry and proportions.

2. **Feature Proportions (25% weight)**: Evaluates balance and proportions between eyes, nose, mouth, and other facial features. Consider interocular distance, nose-to-mouth ratio, etc.

3. **Skin Quality (20% weight)**: Assesses skin clarity, texture, pore visibility, wrinkles, blemishes, and overall skin health.

4. **Dimorphism (20% weight)**: Evaluates sexual dimorphism - how strongly the face expresses masculine or feminine traits. Higher scores indicate more pronounced dimorphic features.

## 3 Key Improvements

For each analysis, the AI provides **exactly 3 specific improvements** ranked by priority:
- Each improvement targets a specific category
- Actionable suggestions for enhancing your appearance
- Ranked from most to least important

Example improvements:
- "Define jawline through targeted exercises or contouring"
- "Improve cheekbone prominence with proper lighting and angles"
- "Balance eye-to-nose ratio with strategic makeup or grooming"

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
   - Configure the NVIDIA API URL in `anvil.yaml`

## Usage

1. Upload a clear front-facing photo
2. Click "Analyze Photo"
3. View your **scores out of 10** in all four categories
4. See your **overall PSL score** and **percentile ranking**
5. Read your **3 personalized improvements**
6. Compare with others on the leaderboard

## Tips for Best Results

- Use a clear, well-lit photo
- Front-facing (selfie-style) works best
- Avoid heavy filters or editing
- Neutral facial expression recommended
- Ensure the face is clearly visible

## Output Format

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

### Improvements (3 specific suggestions)
```json
{
  "improvements": [
    {
      "category": "Facial Geometry",
      "improvement": "Define jawline through targeted exercises or contouring",
      "priority": 1
    },
    {
      "category": "Feature Proportions", 
      "improvement": "Balance eye-to-nose ratio with strategic makeup or grooming",
      "priority": 2
    },
    {
      "category": "Skin Quality",
      "improvement": "Improve skin clarity with proper skincare routine",
      "priority": 3
    }
  ]
}
```

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

### Changing Theme Colors

Edit `theme.yaml` to customize the black and gold color scheme.

## Fallback Analysis

If the NVIDIA API is unavailable, the app falls back to local analysis using:
- OpenCV for image processing
- face-recognition for facial landmark detection
- Custom algorithms for each scoring category (0-10)
- Automatic generation of 3 improvements

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

*Quick, structured face-rating snapshot with 10-point scores and 3 personalized improvements, powered by AI*
