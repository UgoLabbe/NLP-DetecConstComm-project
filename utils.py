from collections import Counter
import pandas as pd
import string
import warnings
import logging
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

def extract_original_text(file_path):
    """
    Load the .conllu file and group sentences by comment ID, returning a list of concatenated texts.
    """
    comments_list = []
    current_comment_id = None
    current_text = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('# sent_id ='):
                # Extract the comment ID (X) from 'sent_id = X_Y'
                sent_id = line.split('=')[1].strip()
                comment_id = sent_id.split('_')[0]

                # Check if we've moved to a new comment
                if current_comment_id is not None and comment_id != current_comment_id:
                    # Store the completed text for the previous comment
                    if current_text:
                        comments_list.append(" ".join(current_text))
                    current_text = []

                # Update the current comment ID
                current_comment_id = comment_id

            elif line.startswith('# text ='):
                # Extract the text for the current sentence
                sentence_text = line.split('=')[1].strip()
                current_text.append(sentence_text)

        # Add the last comment if any
        if current_comment_id is not None and current_text:
            comments_list.append(" ".join(current_text))

    return comments_list

def extract_preprocessed_text(file_path):
    """
    Load the .conllu file and group sentences by comment ID, returning a list of concatenated cleaned texts using lemmas.
    """
    comments_list = []
    current_comment_id = None
    current_text = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('# sent_id ='):
                # Extract the comment ID (X) from 'sent_id = X_Y'
                sent_id = line.split('=')[1].strip()
                comment_id = sent_id.split('_')[0]

                # Check if we've moved to a new comment
                if current_comment_id is not None and comment_id != current_comment_id:
                    # Store the completed text for the previous comment
                    if current_text:
                        comments_list.append(" ".join(current_text))
                    current_text = []

                current_comment_id = comment_id

            elif not line.startswith('#') and line:
                # Extract the lemma (3rd column)
                columns = line.split('\t')
                if len(columns) > 2:
                    lemma = columns[2].lower()  # Use the lemma column in lowercase
                    current_text.append(lemma)

        # Add the last comment if any
        if current_comment_id is not None and current_text:
            comments_list.append(" ".join(current_text))

    return comments_list

def load_conllu_data(file_path):
    comments = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_comment_id = None
        pos_counts = Counter()
        num_tokens = 0
        total_word_length = 0
        num_sentences = 0
        
        for line in f:
            line = line.strip()
            
            # Check if the line is a sentence ID line
            if line.startswith("# sent_id"):
                sent_id = line.split("= ")[1]
                current_comment_id = sent_id.split('_')[0]
                
                # Initialize a new comment entry if not already present
                if current_comment_id not in comments:
                    comments[current_comment_id] = {
                        'num_tokens': 0,
                        'total_word_length': 0,
                        'pos_counts': Counter(),
                        'num_sentences': 0
                    }
                    pos_counts = Counter()
                    num_tokens = 0
                    total_word_length = 0
                    num_sentences = 0

            # Check if the line is a text line
            elif line.startswith("# text"):
                text = line.split("= ")[1]

            # Process token lines
            elif line and not line.startswith("#"):
                columns = line.split("\t")
                if len(columns) >= 4:
                    token = columns[1]
                    pos_tag = columns[3]
                    
                    # Extract token-level features
                    word_length = len(token)
                    total_word_length += word_length
                    num_tokens += 1
                    pos_counts[pos_tag] += 1
            
            # End of a sentence block
            if line == "" and current_comment_id is not None:
                comments[current_comment_id]['num_tokens'] += num_tokens
                comments[current_comment_id]['total_word_length'] += total_word_length
                comments[current_comment_id]['pos_counts'].update(pos_counts)
                comments[current_comment_id]['num_sentences'] += 1

    # Convert the aggregated features to a DataFrame
    features = []
    for comment_id, data in comments.items():
        pos_counts = data['pos_counts']
        avg_word_length = data['total_word_length'] / data['num_tokens'] if data['num_tokens'] > 0 else 0
        
        feature_dict = {
            'comment_id': comment_id,
            'num_tokens': data['num_tokens'],
            'avg_word_length': avg_word_length,
            'num_sentences': data['num_sentences'],
            'num_nouns': pos_counts.get('NOUN', 0),
            'num_verbs': pos_counts.get('VERB', 0),
            'num_adjectives': pos_counts.get('ADJ', 0),
            'num_adverbs': pos_counts.get('ADV', 0),
            'num_pronouns': pos_counts.get('PRON', 0),
            'num_conjunctions': pos_counts.get('CCONJ', 0),
            'num_determiners': pos_counts.get('DET', 0),
        }
        features.append(feature_dict)
    
    df = pd.DataFrame(features)
    return df


