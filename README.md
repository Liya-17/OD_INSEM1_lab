# Garbage Detection System

A web-based garbage classification system using MobileNetV2 transfer learning.

## Classes Detected
- Cardboard (Recyclable)
- Glass (Recyclable)
- Metal (Recyclable)
- Paper (Recyclable)
- Plastic (Recyclable)
- Trash (Non-recyclable)

## Project Structure
```
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com deployment config
├── model/
│   ├── train_model.py      # Training script (use with TrashNet dataset)
│   ├── create_demo_model.py # Creates demo model without dataset
│   └── class_indices.json  # Class label mapping
├── templates/
│   └── index.html          # Web UI
└── static/
    └── uploads/            # Uploaded images
```

## Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate demo model (skip if you have a trained model)
python model/create_demo_model.py

# 4. Run the app
python app.py
# Visit http://localhost:5000
```

## Training with Real Data

1. Download [TrashNet dataset](https://github.com/garythung/trashnet)
2. Run:
   ```bash
   python model/train_model.py
   ```
3. The trained `garbage_model.h5` will be saved in `model/`

## Deploy to Render.com

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Deploy!

> **Note**: The `.h5` model file is in `.gitignore`. For deployment, either:
> - Remove `*.h5` from `.gitignore` and push the model, OR
> - Add a startup script that generates the demo model automatically.
