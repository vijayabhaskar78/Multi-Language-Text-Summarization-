import os
import json
import numpy as np
import pandas as pd
import nltk
import spacy
import torch
import transformers
from typing import List, Dict, Any

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class MultiLanguageSummarizer:
    def __init__(self, languages: List[str] = ['en', 'es', 'fr']):
        self.languages = languages
        self.nlp_models = {}
        self.load_language_models()

    def load_language_models(self):
        language_model_map = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm'
        }

        for lang in self.languages:
            try:
                self.nlp_models[lang] = spacy.load(language_model_map[lang])
            except OSError:
                print(f"Downloading language model for {lang}")
                os.system(f"python -m spacy download {language_model_map[lang]}")
                self.nlp_models[lang] = spacy.load(language_model_map[lang])

    def preprocess_text(self, text: str, language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(f"Language {language} is not supported.")

        nlp = self.nlp_models[language]
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 3]
    
        if not sentences:
            raise ValueError("Text does not contain enough valid sentences for summarization.")

        return sentences


    def summarize(self, text: str, language: str, num_sentences: int = 3) -> str:
        sentences = self.preprocess_text(text, language)
        if len(sentences) < num_sentences:
            raise ValueError(f"Not enough sentences for summarization. Found {len(sentences)}.")

        summary_sentences = self.extract_key_sentences(sentences, num_sentences)
        return ' '.join(summary_sentences)

    def extract_key_sentences(self, sentences: List[str], num_sentences: int = 3) -> List[str]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import networkx as nx
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)

            ranked_sentences = sorted(
                [(scores[i], sentence) for i, sentence in enumerate(sentences)],
                reverse=True
            )
            return [sent for _, sent in ranked_sentences[:num_sentences]]

        except Exception as e:
            raise RuntimeError(f"Error in key sentence extraction: {e}")


# Example usage
def main():
    summarizer = MultiLanguageSummarizer()
    
    # Example texts in different languages
    texts = {
        'en': "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        'es': "El procesamiento del lenguaje natural es un subcampo de la lingüística, la ciencia de la computación y la inteligencia artificial que se ocupa de las interacciones entre computadoras y el lenguaje humano.",
        'fr': "Le traitement du langage naturel est un sous-domaine de la linguistique, des sciences informatiques et de l'intelligence artificielle qui s'intéresse aux interactions entre les ordinateurs et le langage humain."
    }
    
    for lang, text in texts.items():
        summary = summarizer.summarize(text, lang)
        print(f"Summary ({lang}): {summary}")

if __name__ == "__main__":
    main()
