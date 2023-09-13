def get_repaired_transcript(transcript):
    
    sentences = transcript.split("\n")
    
    new_transcript = []
    for sentence in sentences:
        
        words = sentence.split()
        
        new_sentence = []
        for i in range(0,len(words)-1,2):
            
            if i == 0:
                w = words[i].capitalize()
            elif i == len(words)-2:
                w = words[i] + "."
            else:
                w = words[i]
            
            if words[i+1] == "E":
                # do not add these words to the new transcript, bc they are marked as disfluent
                pass
            elif words[i+1] == "_":
                # do add these words to the new transcript, bc they are marked as fluent
                new_sentence.append(w)
         
        new_transcript.append(" ".join(new_sentence))
    
    return " ".join(new_transcript)
