# PSL-APP - Photo Scoring League

A comprehensive face rating application that uses NVIDIA's Kimi K2.6 AI model to analyze and score photos based on four key categories: Facial Geometry, Feature Proportions, Skin Quality, and Dimorphism.

## Features

- **AI-Powered Analysis**: Uses NVIDIA Kimi K2.6 for advanced face analysis
- **Four Category Scores**: Detailed scoring in facial geometry, feature proportions, skin quality, and dimorphism
- **Weighted Overall Score**: Combines category scores with customizable weights
- **Percentile Ranking**: Shows where you stand relative to other users
- **No Account Required**: Upload and analyze photos instantly
- **Black & Gold Theme**: Elegant, professional styling
- **Responsive Design**: Works on desktop and mobile devices

## Scoring Categories

1. **Facial Geometry (35% weight)**: Assesses bone structure, jawline definition, cheekbone prominence, and overall facial harmony
2. **Feature Proportions (25% weight)**: Evaluates balance and proportions between facial features
3. **Skin Quality (20% weight)**: Assesses skin clarity, texture, and health
4. **Dimorphism (20% weight)**: Evaluates sexual dimorphism - how strongly the face expresses masculine or feminine traits

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
3. View your scores in all four categories
4. See your overall PSL score and percentile ranking
5. Compare with others on the leaderboard

## Tips for Best Results

- Use a clear, well-lit photo
- Front-facing (selfie-style) works best
- Avoid heavy filters or editing
- Neutral facial expression recommended
- Ensure the face is clearly visible

## File Structure

```
PSL-APP/
├── anvil.yaml           # Anvil configuration
├── theme.yaml           # Black & Gold theme
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── client_code/
│   ├── __init__.py
│   ├── MainForm.py      # Main form logic
│   └── MainForm.yaml    # Main form UI
└── server_code/
    └── app/
        ├── __init__.py  # Server logic
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
- Custom algorithms for each scoring category

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

*Quick, structured face-rating snapshot powered by AI*
