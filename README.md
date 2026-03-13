# News Summary NLP

## Overview

News Summary NLP is a Korean news summarization project that compares multiple approaches including BERT-based summarization, KoBART summarization, and a custom Seq2Seq model served through a Flask web interface.

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- PyTorch
- Transformers
- KoNLPy
- BeautifulSoup

## Architecture

News URL -> Article crawling -> Text preprocessing -> Multiple summarizers -> Comparative summary output

## Project Structure

```text
news-summary-nlp/
├── src/news_summary_nlp/
├── data/
├── models/
├── notebooks/
├── myproject/templates/
├── train.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python train.py --help
python inference.py --help
python app.py
```

## Results

Reported metrics from the original project:

- ROUGE-1: `0.041`
- ROUGE-2: `0.017`
- ROUGE-L: `0.041`

## Demo

- Flask web app with article URL input
- Comparison across BERT, KoBART, and custom Seq2Seq summaries
- Project report: `4조_miniporject_NLP_News_Summary.pdf`
