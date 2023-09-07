import numpy as np
import random
import re
from nltk.tokenize import sent_tokenize

from collections import Counter

# util function used to create randomly-sized sublists for various transformations
def get_sublists(transcript_text, rng):
    
    # create sublists of random size (according to params below) of the transcript
    text_list = transcript_text.split()
    sublists = []
    start_index = 0
    while start_index < len(text_list):

        # get a random sublist
        mu = 15 
        sigma = 1
        rand_int = int(rng.normal(mu, sigma)) 
        
        # move the end_index of the list up by that rand_int number of spots
        end_index = start_index + rand_int

        # add it to the big list of random sublists 
        sublist = text_list[start_index:end_index]  # if end_index is too big, python just takes the last index
        sublists.append(sublist)

        # then move start_index up
        start_index = end_index
        
    return sublists

def get_transformed_text(N, transcript_text, list_to_use, rng):

    # get sublists of random size of the transcript text
    sublists = get_sublists(transcript_text, rng)

    substrings = []
    for i in range(len(sublists)):
        sublist = sublists[i]
        
        if i+1 < len(sublists)-1:
            next_sublist = sublists[i+1]
        else:
            next_sublist = None

        # append (possibly various) random interjections from the interjections list
        # in all of the cases, the interjections are appended to the END of the random sublist
        
        shift_period = False
        for j in range(0,N):
            
            if list_to_use == []:
                word = sublist[-1]
                word = word.replace(".","").replace("!","").replace("?","")
            else:
                word = rng.choice(list_to_use)
            
            # if this is the end of the transcript, do NOT append anything
            if next_sublist == None:
                continue
            
            elif (j == 0) and ("." in sublist[-1]) and (N == 1):
                last_word = sublist.pop()
                sublist.append(last_word.lower())
                sublist.append(word + ".")
                
            # if this is the first interjection being appended, and there's a 
            elif (j == 0) and ("." in sublist[-1]):
                last_word = sublist.pop()
                sublist.append(last_word.lower())
                sublist.append(word)
                
                shift_period = True
            
            elif (next_sublist) and (j == N-1) and (shift_period == True):
                sublist.append(word + ".")
                
            
            else:
                sublist.append(word)

        substring = " ".join(sublist)
        substrings.append(substring)

    new_text = " ".join(substrings)
    
    def capitalize_after_period(text):
        return re.sub(r'(?<=\. )\w', lambda m: m.group().upper(), text)
    
    new_text = capitalize_after_period(new_text)
    
    if (new_text.split(" ")[-1] != "<SEP>") and (new_text[-1] != "."):
        new_text = new_text + "."
    
    return new_text

def get_repeats_text(n, transcript_text, rng):
    return get_transformed_text(N=n, transcript_text=transcript_text, list_to_use=[], rng=rng)

def get_interjections_text(n, transcript_text, rng):
    return get_transformed_text(N=n, transcript_text=transcript_text, list_to_use=["uh", "um", "well", "like", "so", "okay", "you know", "I mean"], rng=rng)

def get_false_starts_text(n, transcript_text, rng):
    return get_EDITED_text(N=n, transcript_text=transcript_text, rng=rng)

# does the EDITED (false starts only) transformation on the transcript_text
def get_EDITED_text(N, transcript_text, rng):
    
    # break the transcript_text into sentences
    sentences = sent_tokenize(transcript_text)
    
    # and prepare to build the list of new sentences
    new_sentences = []
    
    # get the sentences with len >= 4
    sentences_len_gteq4 = [s for s in sentences if len(s.split(" ")) >= 4]
    
    # randomly determine which sentences to inject a false start into 
    sentences_len_gteq4_mask = rng.choice(1+1,size=len(sentences_len_gteq4), replace=True,p=[0.80,0.20]) #p=[0,1]
    sentences_len_gteq4_mask_index = 0
    
    # then iterate through the sentences and inject those false starts based on the mask
    for i in range(len(sentences)):
        
        # get the current sentence
        sentence = sentences[i]
        #print(sentence)
        
        # and then determine whether or not to inject a false start into that sentence
        if len(sentence.split(" ")) >= 4:
            
            # if the sentence gets a false start
            if sentences_len_gteq4_mask[sentences_len_gteq4_mask_index] == 1:
                
                try:
                    
                    sentence_list = []
                    for word in sentence.split(" "):
                        sentence_list.append(word)
                    
                    subsentence_list = sentence_list[0:2]
                    rest_of_sentence_list = sentence_list[2:]
                    
                    subsentence_list = [s.replace(".","") for s in subsentence_list]
                    subsentence = " ".join(subsentence_list).strip()
                    rest_of_sentence = " ".join(rest_of_sentence_list).strip()

                    # create a lowercased version of the subsentence so it can be appended multiple times
                    lowercased_subsentence = ""
                    if (not subsentence.startswith("I ")) and (not subsentence.startswith("I'")):
                        lowercased_subsentence = subsentence[0].lower() + subsentence[1:].strip()
                    else:
                        lowercased_subsentence = subsentence

                    # build the new sentence
                    new_sentence = ""
                    new_sentence += subsentence + " "  # append the regularly captialized version
                    for _ in range(0,N):
                        new_sentence += lowercased_subsentence + " "  # then, append the rest as lowercased versions   
                    
                    new_sentence += rest_of_sentence  # then, append the rest of the sentence!
                    new_sentences.append(new_sentence)
                
                except:
                    print(subsentence_list)
            
            # if the sentence does not get a false start
            else:
                
                # just append the original sentence
                new_sentences.append(sentence)
            
            # whether it gets appended or not, it's still greater than length 4, so the index counter gets advanced by 1
            sentences_len_gteq4_mask_index += 1
        
        # if the sentence is less than length 4, the original sentence gets appended
        else:
            new_sentences.append(sentence)

    # join all the sentences to build the new transcript
    new_text = " ".join(new_sentences)
    
    if (new_text.split(" ")[-1] != "<SEP>") and (new_text[-1] != "."):
        new_text = new_text + "."
    
    # and return the new transcript
    return new_text

