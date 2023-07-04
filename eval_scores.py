import os
import subprocess
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import statistics
import sys
import re
import string

# calculate the scores between groundth and generated captions


# # file path of output_val_1/2_summary_id_caption.json
# file_path = "/home/grads/h/hasnat.md.abdullah/open_ended_activity_analysis/zero-shot-video-to-text/experiments/summary_study_activityNet/val_1 exp/output_val_1_summary_id_caption.json"

# with open(file_path,'r') as f: 
#     output_val_1_summary_id_caption = json.load(f)
# # for i in output_val_1_summary_id_caption:
# #     print(i)
# #     break
# # bleu score calculation
# reference_captions = []
# candidate_captions = []
# print(len(output_val_1_summary_id_caption))
# for k,v in output_val_1_summary_id_caption.items():
#     if v['generated_summary']!="_":
#         reference_captions.append( v['summary_cap'])
#         candidate_captions.append(v['generated_summary'])


# # Tokenize the captions
# reference_captions = [caption.split() for caption in reference_captions]
# candidate_captions = [caption.split() for caption in candidate_captions]

# print("len(reference_captions)",len(reference_captions))
# print("len(candidate_captions)",len(candidate_captions))
# # Calculate BLEU score
# cumulative_bleu_scores = []
# weight_list = [(1, 0, 0, 0),(0.5, 0.5, 0, 0),(0.33, 0.33, 0.33, 0),(0.25, 0.25, 0.25, 0.25)]
# for n in range(1, 5):
#     bleu_scores = [sentence_bleu(ref, cand, weights=weight_list[n-1]) for ref, cand in zip(reference_captions, candidate_captions)]
    
#     # print(type(bleu_scores))

#     average_bleu_score = statistics.mean(bleu_scores)
#     cumulative_bleu_scores.append(average_bleu_score)

# # Print cumulative BLEU scores
# print("cumulative BLEU scores:")
# for i, score in enumerate(cumulative_bleu_scores):
#     print(f"BLEU@{i+1}: {score:.4f}")



def remove_punctuation(sentence):
    # Remove punctuation using regular expressions
    sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
    return sentence

def cal_scores(fn, true_cap, gen_cap):
    
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
    weight_list = [(1, 0, 0, 0),(0.5, 0.5, 0, 0),(0.33, 0.33, 0.33, 0),(0.25, 0.25, 0.25, 0.25)]
    for n in range(4):
        bleu_scores = [sentence_bleu(ref, cand, weights=weight_list[n]) for ref, cand in zip(reference_captions, candidate_captions)]
        average_bleu_score = statistics.mean(bleu_scores)
        cumulative_bleu_scores.append(average_bleu_score)

    # Print cumulative BLEU scores
    print("cumulative BLEU scores:")
    for i, score in enumerate(cumulative_bleu_scores):
        print(f"BLEU@{i+1}: {score:.4f}")

if __name__ == '__main__':
    fn = sys.argv[1]

    cal_scores(fn, 'summary_cap', 'DeCap_caption')
