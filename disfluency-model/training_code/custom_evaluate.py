


BATCH_SIZE = 4

model_name = "/home/grads/r/rohan.chaudhury/Disfluency/models/checkpoint-2253"
print ("Model name: ", model_name)
output_csv_path= "."
input_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/disfluent.txt"
output_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/fluent.txt"

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0" 

NUM_GPU=4
SEQUENCE_LENGTH = 512

import csv
print ("Model name: ", model_name)
import torch_optimizer as optim

from tqdm import tqdm

from nltk.tokenize import TreebankWordTokenizer

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizers import SentencePieceBPETokenizer
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, TextDataset, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import torch

# from rouge import Rouge 
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import evaluate



print ("Input texts path: ", input_texts_path)
print ("Output texts path: ", output_texts_path)


input_texts = open(input_texts_path).read().split('\n')
output_texts = open(output_texts_path).read().split('\n')





all_texts= input_texts + output_texts
all_words = []
for i in all_texts:
    all_words.extend(i.split())

all_words = list(set(all_words))
# print(all_words)
print(len(all_words))





lengths=[]
print(len(all_texts))
print(len(input_texts))
print(len(output_texts))


for i in range(len(all_texts)):
    lengths.append(len(all_texts[i]))

print ("Max length: ", max(lengths))

length_words=[]
for i in range(len(all_texts)):
    length_words.append(len(all_texts[i].split()))
print("Max length in words: ", max(length_words))





tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=SEQUENCE_LENGTH)



config = T5Config.from_pretrained(model_name)
config.model_max_length = SEQUENCE_LENGTH

model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

def add_to_input_texts(input_texts):
    input_texts = ["Remove text disfluency: " + text.strip() + " [END]" for text in input_texts]
    return input_texts

def add_to_output_texts(input_texts):
    input_texts = [text.strip() + " [END]" for text in input_texts]
    return input_texts



def remove_from_texts(input_texts):
    input_texts = [text.replace("Remove text disfluency:", "").strip() for text in input_texts]
    input_texts = [text.replace("[END]", "").strip() for text in input_texts]
    return input_texts


test_inputs = input_texts
test_outputs = output_texts

test_inputs = add_to_input_texts(test_inputs)


print ("Number of test inputs: ", len(test_inputs))
print ("Number of test outputs: ", len(test_outputs))
print ("First test input: ", test_inputs[0])
print ("First test output: ", test_outputs[0])
print ("Last test input: ", test_inputs[-1])
print ("Last test output: ", test_outputs[-1])





def translate_batch(input_texts, model, tokenizer):
    device = model.device  # Get the device where the model is

    inputs = tokenizer(input_texts, max_length=SEQUENCE_LENGTH, return_tensors="pt", padding=True, truncation=True).to(device)  # Move input tensor to the model's device
    outputs = model.generate(inputs.input_ids, max_length=SEQUENCE_LENGTH, num_return_sequences=1)
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # text_outputs = [output.replace("Output: [start]", "").replace("Corrected Text: ", "") for output in translated_texts]
    text_outputs = [output.split("[END]")[0].strip() for output in translated_texts]
    return text_outputs





# In[ ]:

def lcs(s1, s2):
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + ' ' + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    try:
        cs = matrix[-1][-1]
    except IndexError:
        cs = ""
        print ("Error")
        print (s1)
        print (s2)
    return cs




def calculate_metrics(predicted_sentences, ground_truth_sentences, noisy_sentences):
    # Initialize counters
    true_positive = 0
    false_positive = 0
    false_negative = 0
    gold_number = 0
    predict_number = 0
    correct_number = 0

    for predicted, truth, noisy in zip(predicted_sentences, ground_truth_sentences, noisy_sentences):
        predicted_tokens = TreebankWordTokenizer().tokenize(predicted)
        # truth_tokens = TreebankWordTokenizer().tokenize(truth)
        noisy_tokens = TreebankWordTokenizer().tokenize(noisy)
        truth_tokens = TreebankWordTokenizer().tokenize(lcs(noisy.split(), truth.split()))


        # Initialize the binary masks
        ground_truth_mask = []
        predicted_mask = []
        
        # Indexes to track the positions in the sentences
        idx_noisy = 0
        idx_truth = 0
        idx_predicted = 0
        
        while idx_noisy < len(noisy_tokens):
            # If the word in the noisy sentence is also in the ground truth sentence at the current position,
            # mark as 0 (not removed), else mark as 1 (removed)
            if idx_truth < len(truth_tokens) and noisy_tokens[idx_noisy] == truth_tokens[idx_truth]:
                ground_truth_mask.append(0)
                idx_truth += 1
            else:
                ground_truth_mask.append(1)
            
            # Do the same for the predicted sentence
            if idx_predicted < len(predicted_tokens) and noisy_tokens[idx_noisy] == predicted_tokens[idx_predicted]:
                predicted_mask.append(0)
                idx_predicted += 1
            else:
                # If the word in the predicted sentence is not in the noisy sentence, skip it
                if idx_predicted < len(predicted_tokens) and predicted_tokens[idx_predicted] not in noisy_tokens:
                    idx_predicted += 1
                    continue
                predicted_mask.append(1)
            
            idx_noisy += 1


        gold_number += ground_truth_mask.count(1)
        predict_number += predicted_mask.count(1)
        sum_result = list(map(lambda x: x[0] + x[1], zip(ground_truth_mask, predicted_mask)))
        correct_number += sum_result.count(2)

    try:
        p_score = correct_number * 1.0 / predict_number
        r_score = correct_number * 1.0 / gold_number
        f_score = 2.0 * p_score * r_score / (p_score + r_score)
    except:
        p_score = 0
        r_score = 0
        f_score = 0
    return p_score, r_score, f_score



def translate_inputs(model, tokenizer, test_inputs, batch_size):
    translated_texts = []

    num_samples = len(test_inputs)
    for batch_start_idx in tqdm(range(0, num_samples, batch_size), desc="Translating"):
        batch_end_idx = min(batch_start_idx + batch_size, num_samples)
        input_batch = test_inputs[batch_start_idx:batch_end_idx]
        translated_batch = translate_batch(input_batch, model, tokenizer)
        translated_texts.extend(translated_batch)

    return translated_texts


translated_texts = translate_inputs(model, tokenizer, test_inputs, 10)


final_translated_texts = []
for i, j in zip(test_inputs, translated_texts):
    new_j = lcs(i.split(), j.split())
    final_translated_texts.append(new_j)


rouge = evaluate.load('rouge')



print ("Hugging Face rogue scores without doing LCS")
print (rouge.compute(predictions=translated_texts, references=test_outputs))
print ("Hugging Face rogue scores with doing LCS")
print (rouge.compute(predictions=final_translated_texts, references=test_outputs))


print ("Checking type of the outputs")
print (type(test_outputs))
print (type(test_outputs[0]))
print (type(translated_texts))
print (type(translated_texts[0]))


print ("token based scores without doing LCS")
# predicted_sentences, ground_truth_sentences, noisy_sentences
precision, recall, f1 = calculate_metrics(remove_from_texts(translated_texts), remove_from_texts(test_outputs), remove_from_texts(test_inputs))
print ("Precision: ", precision)
print ("Recall: ", recall)
print ("F1: ", f1)

print ("token based scores with doing LCS")
precision, recall, f1 = calculate_metrics(remove_from_texts(final_translated_texts), remove_from_texts(test_outputs), remove_from_texts(test_inputs))
print ("Precision: ", precision)
print ("Recall: ", recall)
print ("F1: ", f1)

# Write to CSV
with open(output_csv_path+"/"+'translation_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['input_text', 'translated_text', 'original_output_text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the Rouge scores at the top of the CSV
    writer.writerow({"input_text": "HuggingFace Rouge Scores", "translated_text": str(rouge.compute(predictions=translated_texts, references=test_outputs)), "original_output_text": ""})
    writer.writerow({"input_text": "Precision", "translated_text": str(precision), "original_output_text": ""})
    writer.writerow({"input_text": "Recall", "translated_text": str(recall), "original_output_text": ""})
    writer.writerow({"input_text": "F1", "translated_text": str(f1), "original_output_text": ""})

    # Write an empty row to separate the scores from the data
    writer.writerow({})

    # Write the column names
    writer.writeheader()

    # Write the data
    for input_text, translated_text, original_output in zip(test_inputs, translated_texts, test_outputs):
        writer.writerow({'input_text': input_text, 'translated_text': translated_text, 'original_output_text': original_output})

