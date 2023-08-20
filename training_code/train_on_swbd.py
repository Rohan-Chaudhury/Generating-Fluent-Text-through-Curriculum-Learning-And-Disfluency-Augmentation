


BATCH_SIZE = 4
# model_name = 't5-base'
model_name = "/home/grads/r/rohan.chaudhury/Disfluency/models/checkpoint-2253"
print ("Model name: ", model_name)
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0" 

NUM_GPU=4
SEQUENCE_LENGTH = 512
# In[3]:
import csv
print ("Model name: ", model_name)

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
# In[4]:


input_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/disfluent.txt"
output_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/fluent.txt"
test_input_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/fluent.txt"
print ("Input texts path: ", input_texts_path)
print ("Output texts path: ", output_texts_path)


input_texts = open(input_texts_path).read().split('\n')
output_texts = open(output_texts_path).read().split('\n')
test_input_texts = open(test_input_texts_path).read().split('\n')


# In[5]:


all_texts= input_texts + output_texts + test_input_texts
all_words = []
for i in all_texts:
    all_words.extend(i.split())

all_words = list(set(all_words))
# print(all_words)
print(len(all_words))


# In[6]:


lengths=[]
print(len(all_texts))
print(len(input_texts))
print(len(output_texts))
print(len(test_input_texts))

for i in range(len(all_texts)):
    lengths.append(len(all_texts[i]))
# print(all_texts[np.argmax(lengths)])
print ("Max length: ", max(lengths))

length_words=[]
for i in range(len(all_texts)):
    length_words.append(len(all_texts[i].split()))
print("Max length in words: ", max(length_words))




# In[7]:


tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=SEQUENCE_LENGTH)

# new_tokens = all_words
# # new_tokens.extend(['<pad>'])
# # check if the tokens are already in the vocabulary
# new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# print(len(list(new_tokens)))
# # add the tokens to the tokenizer vocabulary
# tokenizer.add_tokens(list(new_tokens))


# In[8]:


# add new, random embeddings for the new tokens

config = T5Config.from_pretrained(model_name)
config.model_max_length = SEQUENCE_LENGTH

model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)


train_inputs, temp_inputs, train_outputs, temp_outputs = train_test_split(input_texts, output_texts, shuffle=True,  test_size=0.3, random_state=42)
val_inputs, test_inputs, val_outputs, test_outputs = train_test_split(temp_inputs, temp_outputs,  shuffle=True, test_size=0.5, random_state=42)

# def add_to_input_texts(input_texts):
#     input_texts = ["Remove text disfluency: " + text for text in input_texts]
#     return input_texts

# def add_to_output_texts(input_texts):
#     # input_texts = ["Corrected Text: " + text for text in input_texts]
#     return input_texts

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

train_inputs = add_to_input_texts(train_inputs)
val_inputs= add_to_input_texts(val_inputs)
test_inputs = add_to_input_texts(test_inputs)

train_outputs = add_to_output_texts(train_outputs)
val_outputs = add_to_output_texts(val_outputs)

print(train_inputs[0])
print(train_outputs[0])
print(val_inputs[0])
print(val_outputs[0])



from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, tokenizer, inputs, outputs, max_length=SEQUENCE_LENGTH):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.outputs = outputs
        self.max_length = max_length

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        input_tokenized = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        output_tokenized = self.tokenizer(output_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": input_tokenized["input_ids"].squeeze(),
            "attention_mask": input_tokenized["attention_mask"].squeeze(),
            "labels": output_tokenized["input_ids"].squeeze(),
        }

    def __len__(self):
        return len(self.inputs)

train_dataset = TranslationDataset(tokenizer, train_inputs, train_outputs)
val_dataset = TranslationDataset(tokenizer, val_inputs, val_outputs)


# In[12]:


from typing import Dict, List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
import torch
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: PaddingStrategy = True,
        max_length: int = None,
        pad_to_multiple_of: int = None,
        model: T5ForConditionalGeneration = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        input_ids = [feature["input_ids"] for feature in features]

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Replace padding_token_id with -100 to ignore loss correctly
        labels = labels["input_ids"].masked_fill(labels["input_ids"] == self.tokenizer.pad_token_id, -100)

        return {"input_ids": input_ids["input_ids"], "labels": labels}

data_collator = CustomDataCollator(tokenizer)


# In[13]:


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=40,  
    early_stopping_threshold=0.0 
)


training_args = Seq2SeqTrainingArguments(
    output_dir="./results/t5baseNewwords/final_Sw_whole_50",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=800,
    save_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit = 5,
    predict_with_generate=True,
    gradient_accumulation_steps=16//(BATCH_SIZE*NUM_GPU),
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    seed=42,
    learning_rate=3e-5,
    weight_decay=0.001,
    report_to="none"
)

import torch_optimizer as optim





optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
optimizer = optim.Lookahead(optimizer)


# optimizer = AdamW



# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = Lookahead(optimizer)

num_training_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
num_warmup_steps = int(num_training_steps * 0.1)  

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# model = T5ForConditionalGeneration.from_pretrained(best_checkpoint)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
    optimizers=(optimizer, scheduler) 
)

print ("Training started now whole 50.")
trainer.train()




def translate_batch(input_texts, model, tokenizer):
    device = model.device  # Get the device where the model is
    # input_texts= add_to_input_texts(input_texts)
    # print(input_texts[0])
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



from nltk.tokenize import TreebankWordTokenizer

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



import csv

from tqdm import tqdm



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
with open('translation_results_brown.csv', 'w', newline='', encoding='utf-8') as csvfile:
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

