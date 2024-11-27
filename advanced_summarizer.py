import torch
from transformers import pipeline

class AbstractiveSummarizer:
    def __init__(self):
        from transformers import pipeline
        
        # Load summarization pipeline
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            raise RuntimeError(f"Error loading summarization model: {e}")

    def generate_summary(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        if len(text.split()) < min_length:
            raise ValueError(f"Input text is too short for summarization. Minimum words required: {min_length}.")

        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            raise RuntimeError(f"Error during summarization: {e}")



# Example usage
def test_abstractive_summarizer():
    abstractive_summarizer = AbstractiveSummarizer()
    
    sample_text = "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It involves developing algorithms and models that enable computers to understand, interpret, and generate human language in a valuable way."
    
    summary = abstractive_summarizer.generate_summary(sample_text)
    print(f"Abstractive Summary: {summary}")

if __name__ == "__main__":
    test_abstractive_summarizer()
