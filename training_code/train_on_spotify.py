


BATCH_SIZE = 2
model_name = 't5-base'
# model_name = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/false_starts/checkpoint-500"
import os
os.environ["WANDB_DISABLED"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
SEQUENCE_LENGTH = 256
# In[3]:


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


# In[4]:

input_texts_path = "0.txt"
output_texts_path = "0.txt"
test_input_texts_path = "5.txt"


# input_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/false_starts/10.txt"
# output_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/false_starts/0.txt"
# test_input_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/false_starts/5.txt"

# input_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/10.txt"
# output_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/0.txt"
# test_input_texts_path = "/home/rohan/multidoc2dial/disfluency/dataset_repeats/5.txt"

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

def add_to_input_texts(input_texts):
    input_texts = ["Remove text disfluency: " + text for text in input_texts]
    return input_texts

def add_to_output_texts(input_texts):
    # input_texts = ["Corrected Text: " + text for text in input_texts]
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
    early_stopping_patience=10,  
    early_stopping_threshold=0.0 
)


training_args = Seq2SeqTrainingArguments(
    output_dir="./results/t5baseNewwords/ablation_new",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=500,
    save_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit = 5,
    predict_with_generate=True,
    gradient_accumulation_steps=64//BATCH_SIZE,
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


# from torch_optimizer import Lookahead
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

print ("Training started now now.")
trainer.train()


# In[ ]:


# best_checkpoint = "./results/t5baseNewwords/checkpoint-1100"
# # print(best_checkpoint)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# best_model = T5ForConditionalGeneration.from_pretrained(best_checkpoint).to(device)
# best_tokenizer = T5Tokenizer.from_pretrained(best_checkpoint)
# best_model.eval()
# print ("Ready to go.")


# In[ ]:


def translate_batch(input_texts, model, tokenizer):
    device = model.device  # Get the device where the model is
    # input_texts= add_to_input_texts(input_texts)
    # print(input_texts[0])
    inputs = tokenizer(input_texts, max_length=SEQUENCE_LENGTH, return_tensors="pt", padding=True, truncation=True).to(device)  # Move input tensor to the model's device
    outputs = model.generate(inputs.input_ids, max_length=SEQUENCE_LENGTH, num_return_sequences=1)
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # text_outputs = [output.replace("Output: [start]", "").replace("Corrected Text: ", "") for output in translated_texts]
    return translated_texts





def word_accuracy(y_true,y_pred):
    total_acc_words = 0
    total_num_words = 0
    for true,pred in zip(y_true,y_pred):
        true_words = true.split(" ")
        pred_words = pred.split(" ")
        count=0
        length = min(len(true_words),len(pred_words))
        for i in range(length):
            if (true_words[i] == pred_words[i]):
                count+=1
        total_acc_words+= count
        total_num_words+= len(true_words)
    acc = np.float(total_acc_words)/(total_num_words)
    return acc


# In[ ]:



from tqdm import tqdm

def evaluate_word_accuracy(model, tokenizer, test_inputs, test_outputs, batch_size=BATCH_SIZE):
    total_accuracy = 0
    num_samples = len(test_inputs)

    # Wrap the range object with tqdm for progress tracking
    for batch_start_idx in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
        batch_end_idx = min(batch_start_idx + batch_size, num_samples)
        input_batch = test_inputs[batch_start_idx:batch_end_idx]
        ground_truth_batch = test_outputs[batch_start_idx:batch_end_idx]

        translated_texts = translate_batch(input_batch, model, tokenizer)

        for ground_truth, translated_text in zip(ground_truth_batch, translated_texts):
            # print(ground_truth)
            print ("------------------")
            print(translated_text)
            print ("------------------")
            accuracy = word_accuracy(ground_truth, translated_text)
            total_accuracy += accuracy

    average_accuracy = total_accuracy / num_samples
    return average_accuracy
test_outputs = [test_output.replace("<pad> ", "", 1) for test_output in test_outputs]

from rouge import Rouge 

def evaluate_rouge_scores(model, tokenizer, test_inputs, test_outputs, batch_size=BATCH_SIZE):
    rouge = Rouge()
    total_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0}, 'rouge-2': {'f': 0, 'p': 0, 'r': 0}, 'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
    num_samples = len(test_inputs)

    # Wrap the range object with tqdm for progress tracking
    for batch_start_idx in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
        batch_end_idx = min(batch_start_idx + batch_size, num_samples)
        input_batch = test_inputs[batch_start_idx:batch_end_idx]
        ground_truth_batch = test_outputs[batch_start_idx:batch_end_idx]

        translated_texts = translate_batch(input_batch, model, tokenizer)

        for ground_truth, translated_text in zip(ground_truth_batch, translated_texts):
            print ("------------------")
            print(ground_truth)
            print ("------------------")
            print(translated_text)
            print ("------------------")
            scores = rouge.get_scores(translated_text, ground_truth, avg=True)
            total_scores['rouge-1']['f'] += scores['rouge-1']['f']
            total_scores['rouge-1']['p'] += scores['rouge-1']['p']
            total_scores['rouge-1']['r'] += scores['rouge-1']['r']

            total_scores['rouge-2']['f'] += scores['rouge-2']['f']
            total_scores['rouge-2']['p'] += scores['rouge-2']['p']
            total_scores['rouge-2']['r'] += scores['rouge-2']['r']

            total_scores['rouge-l']['f'] += scores['rouge-l']['f']
            total_scores['rouge-l']['p'] += scores['rouge-l']['p']
            total_scores['rouge-l']['r'] += scores['rouge-l']['r']

    average_scores = {key: {subkey: total_scores[key][subkey] / num_samples for subkey in ['f', 'p', 'r']} for key in ['rouge-1', 'rouge-2', 'rouge-l']}
    return average_scores


# In[ ]:


# validation_accuracy = evaluate_word_accuracy(best_model, best_tokenizer, test_inputs, test_outputs,8)
# print(f'Word accuracy on the validation set: {validation_accuracy:.4f}')


# validation_accuracy = evaluate_word_accuracy(model, best_tokenizer, test_inputs, test_outputs,8)
# print(f'Word accuracy on the validation set: {validation_accuracy:.4f}')


# rouge_scores = evaluate_rouge_scores(best_model, best_tokenizer, test_inputs, test_outputs, 64)

# print("ROUGE scores:")
# for rouge_type, scores in rouge_scores.items():
#     print(f"{rouge_type.upper()}:")
#     for score_type, score in scores.items():
#         print(f"  {score_type.upper()}: {score:.4f}")



rouge_scores = evaluate_rouge_scores(model, tokenizer, test_inputs, test_outputs, 50)

print("ROUGE scores:")
for rouge_type, scores in rouge_scores.items():
    print(f"{rouge_type.upper()}:")
    for score_type, score in scores.items():
        print(f"  {score_type.upper()}: {score:.4f}")