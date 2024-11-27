# Multi-Language-Text-Summarization-

This project offers a comprehensive text summarization service that combines both extractive and abstractive summarization methods to create concise and informative summaries. It supports multiple languages, including English, Spanish, and French, making it accessible to a wide range of users. The service is built with Flask, a lightweight web framework, which allows you to easily integrate this functionality into any application or service.

The project utilizes advanced spaCy, NLTK, and transformers models to process and summarize text effectively. spaCy is used for tokenization, part-of-speech tagging, and other linguistic tasks, while NLTK handles text preprocessing, including stopword removal and sentence segmentation. The transformers library from Hugging Face provides powerful deep learning models that generate accurate and fluent abstractive summaries, making the summarization process more natural and human-like.

## Features

- Multi-Language Support: Supports summarization in English, Spanish, and French.
- Extractive Summarization: Extracts key sentences directly from the input text to create a summary.
- Abstractive Summarization: Generates new sentences based on the input text, creating a more natural and concise summary.
- Flask API: Exposes a simple API for easy integration with other applications or services.

## Installation Instructions
### 1. Prepare Your Ubuntu Environment
```python
sudo apt update
sudo apt upgrade -y
```

### 2. Install Python and Essential Development Tools
```python
sudo apt install python3 python3-pip python3-venv git build-essential -y
```

### 3. Set Up Project Environment
```python
mkdir multi-language-summarizer
cd multi-language-summarizer
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Dependencies
```python
pip install --upgrade pip
pip install numpy pandas scikit-learn torch spacy nltk transformers flask flask-cors networkx
```

### 5. Download SpaCy Language Models
```python
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_news_sm
python3 -m spacy download fr_core_news_sm
```

### 6. Create Project Files
```python
touch multi_language_summarizer.py
touch advanced_summarizer.py
touch app.py
```

### 7. Add Code to Project Files
Add the respective code to multi_language_summarizer.py, advanced_summarizer.py, and app.py.

### 8. Install Additional NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 9. Run the Flask Application
```python
python3 app.py
```

### 10. Test the Application
Use curl to test the summarization functionality:
- Extractive Summarization:
  ```python
  curl -X POST http://localhost:5000/summarize \
     -H "Content-Type: application/json" \
     -d '{
         "text": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
         "language": "en",
         "mode": "extractive"
     }'
     ```
- Abstractive Summarization:
  ```python
  curl -X POST http://localhost:5000/summarize \
     -H "Content-Type: application/json" \
     -d '{
         "text": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
         "mode": "abstractive"
     }'
     ```
  
  
