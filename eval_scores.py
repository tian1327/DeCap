import os
import subprocess
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
import statistics
import sys
import re
import string
# from pycocoevalcap.cider import Cider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('wordnet')

# calculate the scores between groundth and generated captions

def cal_cider_score(reference_tokens, candidate_tokens):

    # print('reference_tokens', reference_tokens)
    # print('candidate_tokens', candidate_tokens)

    # Convert tokens to string for TF-IDF vectorization
    generated_text = ' '.join(candidate_tokens)
    reference_texts = [' '.join(reference_tokens)]

    # print('generated_text', generated_text)
    # print('reference_texts', reference_texts)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([generated_text] + reference_texts)

    # print('type(tfidf_matrix)', type(tfidf_matrix))
    # print('tfidf_matrix:', tfidf_matrix)
    # print('tfidf_matrix[0]:', tfidf_matrix[0])
    # print('tfidf_matrix[1:]:', tfidf_matrix[1:])

    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    # print('similarities', similarities)

    # Compute consensus score
    consensus_score = similarities.mean()
    # print('consensus_score', consensus_score)

    # Compute sentence length penalty
    generated_length = len(candidate_tokens)
    # reference_lengths = [len(ref) for ref in reference_tokens] # this applies when there are multiple references
    reference_lengths = len(reference_tokens) # this applies when there is only one reference
    # print('generated_length', generated_length)
    # print('reference_lengths', reference_lengths)

    length_penalty = max(0, 1 - abs(generated_length - np.mean(reference_lengths)) / np.mean(reference_lengths))
    # print('length_penalty', length_penalty)

    # Compute CIDEr score
    cider_score = consensus_score * length_penalty
    # print('cider_score', cider_score)
    # stop

    return cider_score



def remove_punctuation(sentence):
    # Remove punctuation using regular expressions
    sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)

    # convert to lower case
    sentence = sentence.lower()
    # print('sentence', sentence)

    # stem words
    stemmer = nltk.stem.PorterStemmer()
    sentence = " ".join([stemmer.stem(word) for word in sentence.split()])
    # print('sentence', sentence)

    return sentence

def cal_scores(fn, true_cap, gen_cap):
    
    print(f"\nCalculating scores for {gen_cap}...")

    # load captions
    with open(fn, 'r') as f:
        data = json.load(f)
    
    reference_captions = []
    candidate_captions = []
    for vid, info in data.items():
        if info[gen_cap] != "_":
            
            # remove punctuation
            true_caption = remove_punctuation(info[true_cap])
            gen_caption = remove_punctuation(info[gen_cap])

            # tokenize
            reference_captions.append(true_caption.split())
            candidate_captions.append(gen_caption.split())

    # Calculate BLEU score
    cumulative_bleu_scores = []
    bleu_scores_list = []
    weight_list = [(1, 0, 0, 0),(0.5, 0.5, 0, 0),(0.33, 0.33, 0.33, 0),(0.25, 0.25, 0.25, 0.25)]
    for n in range(4):
        bleu_scores = [sentence_bleu(ref, cand, weights=weight_list[n]) for ref, cand in zip(reference_captions, candidate_captions)]        
        average_bleu_score = statistics.mean(bleu_scores)
        bleu_scores_list.append(bleu_scores)
        cumulative_bleu_scores.append(average_bleu_score)
        
    # Calculate METEOR score
    meteor_scores = [meteor_score([ref], cand) for ref, cand in zip(reference_captions, candidate_captions)]
    average_meteor_score = statistics.mean(meteor_scores)

    # Calculate CIDEr score
    cider_scores = [cal_cider_score(ref, cand) for ref, cand in zip(reference_captions, candidate_captions)]
    average_cider_score = statistics.mean(cider_scores)

    # assign the scores to the data
    idx = 0
    for vid, info in data.items():
        if info[gen_cap] != "_":
            info[f'{gen_cap}_bleu1'] = round(bleu_scores_list[0][idx], 6)
            info[f'{gen_cap}_bleu2'] = round(bleu_scores_list[1][idx], 6)
            info[f'{gen_cap}_bleu3'] = round(bleu_scores_list[2][idx], 6)
            info[f'{gen_cap}_bleu4'] = round(bleu_scores_list[3][idx], 6)
            info[f'{gen_cap}_meteor'] = round(meteor_scores[idx], 6)
            info[f'{gen_cap}_cider'] = round(cider_scores[idx], 6)
            idx += 1
        else:
            info[f'{gen_cap}_bleu1'] = -1
            info[f'{gen_cap}_bleu2'] = -1
            info[f'{gen_cap}_bleu3'] = -1
            info[f'{gen_cap}_bleu4'] = -1
            info[f'{gen_cap}_meteor'] = -1
            info[f'{gen_cap}_cider'] = -1

    # save the data to json file
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved scores to {fn}")
     
    # Print cumulative BLEU scores
    print("\nCumulative BLEU scores:")
    for i, score in enumerate(cumulative_bleu_scores):
        print(f"BLEU@{i+1}: {score:.4f}")
    
    print("\nMETEOR score:")
    print(f"{average_meteor_score:.4f}")

    print("\nCIDEr score:")
    print(f"{average_cider_score:.4f}")

if __name__ == '__main__':
    fn = sys.argv[1]

    cal_scores(fn, 'summary_cap', 'DeCap_caption')
    cal_scores(fn, 'summary_cap', 'generated_summary')
