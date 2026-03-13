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

Input article URL -> Crawling and preprocessing -> Summarization models -> Summary output

## Usage

```bash
pip install -r requirements.txt
python train.py
python inference.py
python app.py
```

## Results

Reported metrics from the original project:

- ROUGE-1: `0.041`
- ROUGE-2: `0.017`
- ROUGE-L: `0.041`
