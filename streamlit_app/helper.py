
import re
import numpy as np
import pickle
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

cv = pickle.load(open('model/count_vectorizer.pkl','rb'))
STOP_WORDS = ENGLISH_STOP_WORDS

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest:x_longest]

def test_common_words(q1, q2):
    return len(set(q1.lower().split()) & set(q2.lower().split()))

def test_total_words(q1, q2):
    return len(set(q1.lower().split())) + len(set(q2.lower().split()))

def test_fetch_token_features(q1, q2, safe_div=0.0001):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if not q1_tokens or not q2_tokens:
        return [0.0] * 8
    
    q1_words = {w for w in q1_tokens if w not in STOP_WORDS}
    q2_words = {w for w in q2_tokens if w not in STOP_WORDS}
    q1_stops = set(q1_tokens) - q1_words
    q2_stops = set(q2_tokens) - q2_words
    
    common_word = len(q1_words & q2_words)
    common_stop = len(q1_stops & q2_stops)
    common_token = len(set(q1_tokens) & set(q2_tokens))
    
    return [
        common_word / (min(len(q1_words), len(q2_words)) + safe_div),
        common_word / (max(len(q1_words), len(q2_words)) + safe_div),
        common_stop / (min(len(q1_stops), len(q2_stops)) + safe_div),
        common_stop / (max(len(q1_stops), len(q2_stops)) + safe_div),
        common_token / (min(len(q1_tokens), len(q2_tokens)) + safe_div),
        common_token / (max(len(q1_tokens), len(q2_tokens)) + safe_div),
        int(q1_tokens[-1] == q2_tokens[-1]),
        int(q1_tokens[0] == q2_tokens[0])
    ]

def test_fetch_length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if not q1_tokens or not q2_tokens:
        return [0.0] * 3
    
    lcs = longest_common_substring(q1, q2)
    min_len = max(min(len(q1), len(q2)), 1)
    
    return [
        abs(len(q1_tokens) - len(q2_tokens)),
        (len(q1_tokens) + len(q2_tokens)) / 2,
        len(lcs) / min_len
    ]

def string_similarity(str1, str2):
    """More reliable similarity metrics"""
    str1 = preprocess(str1)
    str2 = preprocess(str2)
    
    # Basic string similarity
    seq_match = SequenceMatcher(None, str1, str2).ratio() * 100
    
    # Token-based similarities
    tokens1 = str1.split()
    tokens2 = str2.split()
    
    # Jaccard similarity (proper implementation)
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1 & set2)
    union = max(len(set1 | set2), 1)
    jaccard = intersection / union * 100
    
    return [
        seq_match,
        jaccard,
        len(set1 & set2),  # Common words count
        int(str1 == str2) * 100  # Exact match flag
    ]

def preprocess(q):
    """More conservative preprocessing"""
    q = str(q).lower().strip()
    q = BeautifulSoup(q, 'html.parser').get_text()
    # Only remove special characters, keep basic punctuation
    q = re.sub(r'[^a-z0-9\s]', ' ', q)
    return ' '.join(q.split()

def query_point_creator(q1, q2):
    q1_processed = preprocess(q1)
    q2_processed = preprocess(q2)
    
    features = [
        len(q1_processed),
        len(q2_processed),
        len(q1_processed.split()),
        len(q2_processed.split()),
        test_common_words(q1_processed, q2_processed),
        test_total_words(q1_processed, q2_processed),
        test_common_words(q1_processed, q2_processed) / (test_total_words(q1_processed, q2_processed) + 1e-6)
    ]
    
    features.extend(test_fetch_token_features(q1_processed, q2_processed))
    features.extend(test_fetch_length_features(q1_processed, q2_processed))
    features.extend(string_similarity(q1_processed, q2_processed))
    
    bow_features = np.hstack([
        cv.transform([q1_processed]).toarray(),
        cv.transform([q2_processed]).toarray()
    ])
    
    return np.hstack([np.array(features).reshape(1, -1), bow_features])
