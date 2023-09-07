import os

import utils_general
import utils_trees

def get_trees_and_trees_tagged_files(train_files, dev_files, test_files):
    
    utils_general.create_and_or_clear_this_dir("./swb")
            
    utils_general.create_and_or_clear_this_dir("./swb/train")
    utils_general.create_and_or_clear_this_dir("./swb/dev")
    utils_general.create_and_or_clear_this_dir("./swb/test")

    utils_general.just_create_this_dir("./swb/train/trees")
    utils_general.just_create_this_dir("./swb/dev/trees")
    utils_general.just_create_this_dir("./swb/test/trees")

    utils_general.just_create_this_dir("./swb/train/trees-tagged")
    utils_general.just_create_this_dir("./swb/dev/trees-tagged")
    utils_general.just_create_this_dir("./swb/test/trees-tagged")

    utils_trees.copy_files_over(in_filepaths=train_files, out_dir="./swb/train/trees", withTags=False, isBrown=False)
    utils_trees.copy_files_over(in_filepaths=dev_files, out_dir="./swb/dev/trees", withTags=False, isBrown=False)
    utils_trees.copy_files_over(in_filepaths=test_files, out_dir="./swb/test/trees", withTags=False, isBrown=False)

    utils_trees.copy_files_over(in_filepaths=train_files, out_dir="./swb/train/trees-tagged", withTags=True, isBrown=False)
    utils_trees.copy_files_over(in_filepaths=dev_files, out_dir="./swb/dev/trees-tagged", withTags=True, isBrown=False)
    utils_trees.copy_files_over(in_filepaths=test_files, out_dir="./swb/test/trees-tagged", withTags=True, isBrown=False)

    tagged_train_files = [os.path.join("./swb/train/trees-tagged", file) for file in os.listdir("./swb/train/trees-tagged") if "sw" in file]
    tagged_dev_files = [os.path.join("./swb/dev/trees-tagged", file) for file in os.listdir("./swb/dev/trees-tagged") if "sw" in file]
    tagged_test_files = [os.path.join("./swb/test/trees-tagged", file) for file in os.listdir("./swb/test/trees-tagged") if "sw" in file]
    
    return tagged_train_files, tagged_dev_files, tagged_test_files
    
def get_fluent_and_disfluent_files(tagged_train_files, tagged_dev_files, tagged_test_files):
    split_dict = {"train": tagged_train_files,
                  "dev": tagged_dev_files,
                  "test": tagged_test_files}
    
    fluency_dict = {"fluent": False, 
                    "disfluent": True}
    
    for split_name, split_list in split_dict.items():

        split_dir = os.path.join(".", "swb", split_name)
        utils_general.just_create_this_dir(split_dir)

        for filepath in split_list:
            for fluency_name, fluency_bool in fluency_dict.items():

                fluency_dir = os.path.join(split_dir, fluency_name)
                utils_general.just_create_this_dir(fluency_dir)

                transcript = ""
                transcript = utils_trees.get_clean_transcript_from_tree_file(filepath=filepath, get_disfluent=fluency_bool)

                old_filename = filepath.split("/")[-1]
                new_filename = old_filename.replace(".mrg",".txt")
                utils_general.write_file(new_filename, fluency_dir, transcript)
