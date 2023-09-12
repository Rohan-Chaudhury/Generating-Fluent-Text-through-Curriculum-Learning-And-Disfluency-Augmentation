def get_repaired_transcript(transcript):
    new_transcript = []
    for i in range(0,len(transcript)-2,2):
        if transcript[i+1] == "E":
            # do not add these words to the new transcript, bc they are marked as disfluent
            pass
        elif transcript[i+1] == "_":
            # do add these words to the new transcript, bc they are marked as fluent
            new_transcript.append(transcript[i])
            
    new_transcript = " ".join(new_transcript)
    return new_transcript