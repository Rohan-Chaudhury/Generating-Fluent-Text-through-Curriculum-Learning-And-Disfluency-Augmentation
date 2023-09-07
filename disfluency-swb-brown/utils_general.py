import os
import shutil

PATH_TO_PROJECT = ""

# utils function to write a file to a specified directory
def write_file(new_filename, directory, text):
    new_filepath = os.path.join(directory, new_filename)
    with open(new_filepath, mode="w") as f:
        f.write(text)
        
# utils function to write a file if it's not blank
# this prevents podcast transcripts without text in that chunk of time
# ex: truncated to 30sec, but transcript starts at 40 seconds
# to not be passed to the summarization models
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

# utils function to create or clear the directory
def create_and_or_clear_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
    # if this dir does exist, clear it out and re-create it
    else:
        shutil.rmtree(d)
        os.mkdir(d)
        
# utils function to create the directory
def just_create_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)