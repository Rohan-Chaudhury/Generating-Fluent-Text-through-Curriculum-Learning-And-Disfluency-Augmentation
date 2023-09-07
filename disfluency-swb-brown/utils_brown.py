import os

import utils_general
import utils_trees

fluency_dir = {"fluent": False, "disfluent": True}

def get_trees_and_trees_tagged_files(BROWN_FILES_LIST):
    utils_general.create_and_or_clear_this_dir("./brown")
    utils_general.just_create_this_dir("./brown/trees")
    utils_general.just_create_this_dir("./brown/trees-tagged")
    utils_trees.copy_files_over(in_filepaths=BROWN_FILES_LIST, out_dir="./brown/trees", withTags=False, isBrown=True)
    utils_trees.copy_files_over(in_filepaths=BROWN_FILES_LIST, out_dir="./brown/trees-tagged", withTags=True, isBrown=True)
    
def get_fluent_and_disfluent_files():
    input_dir = os.path.join(".", "brown", "trees-tagged")
    output_dir = os.path.join(".", "brown")
    utils_general.just_create_this_dir(output_dir)

    for fluency, fluency_bool in fluency_dir.items():
        utils_general.create_and_or_clear_this_dir(os.path.join(output_dir, fluency))
        for file in [os.path.join(input_dir,f) for f in os.listdir(os.path.join(input_dir)) if (f.endswith(".mrg") and not f.endswith("-checkpoint.txt"))]:
            text = ""
            text = utils_trees.get_consistent_transcript(filepath=file, get_disfluent=fluency_bool, isBrown=True)
            utils_general.write_file(new_filename=file.split("/")[-1].replace(".mrg",".txt"), directory=os.path.join(output_dir, fluency), text=text)
            
            
def get_model_files():
    input_dir = os.path.join(".", "brown")
    output_dir = os.path.join(".", "brown", "model_files")
    utils_general.create_and_or_clear_this_dir(output_dir)

    for fluency, fluency_bool in fluency_dir.items():
        # write it out to the csv
        current_input_dir = os.path.join(input_dir, fluency)
        current_output_path = os.path.join(output_dir, fluency + ".txt")
        for file in [os.path.join(current_input_dir,f) for f in os.listdir(os.path.join(current_input_dir)) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]:
            new_file_name = file.split("/")[-1].replace(".mrg",".txt")
            text = ""
            text = utils_trees.get_consistent_transcript(filepath=file, get_disfluent=fluency_bool, isBrown=True)
            with open(current_output_path, mode="a") as f:
                f.write(new_file_name + ":" + text + "\n")
    
    