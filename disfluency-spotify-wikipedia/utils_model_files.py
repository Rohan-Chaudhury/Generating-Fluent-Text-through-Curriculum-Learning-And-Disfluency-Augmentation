import os
import utils_general


def get_model_files(dataset_name="spotify", trfs=["repeats", "interjections", "false-starts", "all-3"], includeNegativeOne=False):
    
    # create output dir
    output_dir = os.path.join(utils_general.PATH_TO_PROJECT, dataset_name+"-model-files")
    utils_general.create_and_or_clear_this_dir(output_dir)
    
    # get the zero dir
    zero_dir = os.path.join(utils_general.PATH_TO_PROJECT, dataset_name, "0")
    
    # write out the -1 files
    if includeNegativeOne:
        
        # clear out the model-file directory
        utils_general.create_and_or_clear_this_dir(os.path.join(output_dir, "-1"))
        
        # iterate through k = 0, then append to k = 1,2,3,... for the rest of them
        files_list = [f for f in os.listdir(zero_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
        for f in files_list:

            # write the zero file out to the zero csv
            text = ""  # initialize
            text = utils_general.read_file(os.path.join(utils_general.PATH_TO_PROJECT, dataset_name, "-1", f))
            
            with open(os.path.join(output_dir, "-1", "-1.txt"), mode="a") as open_file:
                open_file.write(f + ": " + text + "\n")
        
    
    # for the 1-10 dirs
    for t in trfs:
        
        utils_general.create_and_or_clear_this_dir(os.path.join(output_dir, t))

        # iterate through k = 0, then append to k = 1,2,3,... for the rest of them
        files_list = [f for f in os.listdir(zero_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
        for f in files_list:

            # write the zero file out to the zero csv
            text = ""  # initialize
            text = utils_general.read_file(os.path.join(zero_dir, f))
            
            with open(os.path.join(output_dir, t, "0.txt"), mode="a") as open_file:
                open_file.write(f + ": " + text + "\n")

            # get all of the rest of the k-versions of this file
            for k in range(1,10+1):

                # read the current file
                text = ""  # initialize
                text = utils_general.read_file(os.path.join(utils_general.PATH_TO_PROJECT, dataset_name, t, str(k) + "_" + f))

                # write it out to the csv
                with open(os.path.join(output_dir, t, str(k) + ".txt"), mode="a") as open_file:
                    open_file.write(f + ": " + text + "\n")
