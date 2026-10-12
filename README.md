# Twitter Mood Detection System

A Python sentiment-analysis project with tweet collection, text preprocessing, model comparison, and a Streamlit dashboard. Implemented methods include VADER, TextBlob, a pretrained transformer pipeline, and an optional TF-IDF/custom classifier trained on supplied labels. No validated accuracy or live deployment is claimed.

## Run locally

Use Python 3.11. The application is inside `twitter-mood-detector/`:

```bash
git clone https://github.com/Sourya-Prabaharan/Twitter-Mood-Detection-System.git
cd Twitter-Mood-Detection-System/twitter-mood-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp env_example.txt .env
python run_dashboard.py
```

Open http://localhost:8501. Put your own bearer token in the local `.env` only. Live collection requires current X/Twitter API access; account limits and paid-access requirements vary. NLP resources and transformer weights may download on first use.

## Architecture and contribution

The repository connects [collection](twitter-mood-detector/src/twitter_collector.py), [preprocessing](twitter-mood-detector/src/data_preprocessor.py), [sentiment methods](twitter-mood-detector/src/sentiment_analyzer.py), and [dashboard](twitter-mood-detector/src/dashboard.py). The custom-model path needs genuine labeled text; the repository does not establish its performance on an independent test set.

See the [usage guide](twitter-mood-detector/README.md) for analysis and export examples. Run a source syntax check without credentials:

```bash
python -m compileall -q src run_dashboard.py
```

Live API collection, external model downloads, and sentiment quality require separate validation. Social-media sentiment predictions can miss sarcasm, context, dialect, and domain shifts.
