


BATCH_SIZE = 4
# model_name = 't5-base'
model_name = "/home/grads/r/rohan.chaudhury/Disfluency/models/checkpoint-2253"
print ("Model name: ", model_name)
output_model_path= "./results/t5baseNewwords/final_Sw_whole_50"
input_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/disfluent.txt"
output_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/fluent.txt"
test_input_texts_path = "/home/grads/r/rohan.chaudhury/Disfluency/formatted_text/final_sw_whole/train/fluent.txt"
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0" 

NUM_GPU=4
SEQUENCE_LENGTH = 512
# In[3]:
import csv
print ("Model name: ", model_name)
import torch_optimizer as optim


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

print ("Max length: ", max(lengths))

length_words=[]
for i in range(len(all_texts)):
    length_words.append(len(all_texts[i].split()))
print("Max length in words: ", max(length_words))




# In[7]:


tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=SEQUENCE_LENGTH)



config = T5Config.from_pretrained(model_name)
config.model_max_length = SEQUENCE_LENGTH

model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)


train_inputs, temp_inputs, train_outputs, temp_outputs = train_test_split(input_texts, output_texts, shuffle=True,  test_size=0.3, random_state=42)
val_inputs, test_inputs, val_outputs, test_outputs = train_test_split(temp_inputs, temp_outputs,  shuffle=True, test_size=0.5, random_state=42)


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
    output_dir=output_model_path,
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





optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
optimizer = optim.Lookahead(optimizer)


num_training_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
num_warmup_steps = int(num_training_steps * 0.1)  

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


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

print ("Training started now.")
trainer.train()

trainer.evaluate()

print ("Training completed.")