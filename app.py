from flask import Flask, request, jsonify
from flask_cors import CORS
from multi_language_summarizer import MultiLanguageSummarizer
from advanced_summarizer import AbstractiveSummarizer

app = Flask(__name__)
CORS(app)

# Initialize summarization models
extractive_summarizer = MultiLanguageSummarizer()
abstractive_summarizer = AbstractiveSummarizer()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        # Parse input JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request payload is missing."}), 400

        text = data.get('text', '')
        language = data.get('language', 'en')
        mode = data.get('mode', 'extractive').lower()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if mode not in ['extractive', 'abstractive']:
            return jsonify({"error": f"Unsupported mode: {mode}. Use 'extractive' or 'abstractive'."}), 400

        # Select summarizer based on mode
        if mode == 'extractive':
            summary = extractive_summarizer.summarize(text, language)
        else:
            summary = abstractive_summarizer.generate_summary(text)

        return jsonify({"summary": summary}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
