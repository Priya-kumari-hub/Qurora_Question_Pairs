
This project aims to detect whether two questions asked on Quora are duplicates or not — meaning they are essentially asking the same thing, even if worded differently. Duplicate detection helps improve search experience and avoids redundant answers.

@@What It Does
Your machine learning system takes in two user-submitted questions and analyzes how semantically similar they are. It outputs the probability that the questions are duplicates.

It handles misspellings, synonyms, rephrased queries, and partial matches.

It runs in real-time and is exposed via a Streamlit web interface so users can easily test question pairs.

@@Core ML Pipeline
Dataset
You use the official [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) from Kaggle which contains ~400,000 labeled question pairs. Each pair is marked as duplicate (1) or not (0).

Preprocessing

Lowercasing

Removing special characters

Stemming

Feature Extraction
You extract multiple features that measure the similarity between two questions:

Cosine similarity (using TF-IDF vectorizer)

Token overlap ratio

Longest common substring

SequenceMatcher ratio (Levenshtein-based)

Jaccard similarity

Model Training

You use XGBoost, a powerful gradient boosting model, trained on the extracted features and the binary duplicate labels.

@@Web Interface
Built using Streamlit

Users can input two questions and get an instant duplicate probability

Fast response (100+ question pairs/sec on CPU)

🗂️ Project Structure
plaintext
Copy
Edit
Qurora_Question_Pairs/
├── model/
│   ├── count_vectorizer.pkl       # Saved TF-IDF model
│   └── quora_xgb_model.pkl        # Trained XGBoost model
├── streamlit_app/
│   ├── app.py                     # Web interface logic
│   ├── helper.py                  # Similarity computation
│   ├── requirements.txt
│   └── setup.sh
