import os
import shutil

# set these paths
PATH_TO_PROJECT = "/home/grads/m/mariateleki/Generating-Fluent-Text-through-Curriculum-Learning-And-Disfluency-Augmentation/disfluency-spotify-wikipedia"
PATH_TO_2020_TESTSET_DIR = "/data2/maria/Spotify/TREC/spotify-podcasts-2020/podcasts-transcripts-summarization-testset"
PATH_TO_2020_TESTSET_DF = "/data2/maria/Spotify/TREC/spotify-podcasts-2020/metadata-summarization-testset.tsv"


def write_file(new_filename, directory, text):
    new_filepath = os.path.join(directory, new_filename)
    with open(new_filepath, mode="w") as f:
        f.write(text)
        
def write_file_if_not_blank(file, text):
    if text:
        with open(file, mode="w") as f:
            f.write(text)
            
def read_file(filepath):
    text = ""
    with open(filepath) as f:
        text = f.read()
    return text

def delete_file_if_already_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def create_and_or_clear_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
    # if this dir does exist, clear it out and re-create it
    else:
        shutil.rmtree(d)
        os.mkdir(d)
        
def just_create_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)