import os
import utils_general


def get_model_files(input_dir):
    input_dir = "/data1/maria/2023_03_08/2023_02_10/test_synthetic/all-3"
    output_dir = os.path.join(".", "0_to_10")

    utils_general.create_and_or_clear_this_dir(output_dir)

    # create all of the output files
    for k in range(0,10+1):  
        output_path = os.path.join(output_dir, str(k) + ".txt")

    # iterate through k = 0, then append to k = 1,2,3,... for the rest of them
    files_list = [f for f in os.listdir(input_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt") and f.startswith("0"))]
    for f in files_list:

        # this gets the file_id
        file = f.replace("0_", "")

        # write it out to the csv
        text = ""  # initialize
        current_file = os.path.join(input_dir, "0_" + file)
        text = utils_general.read_file(os.path.join(input_dir, current_file))
        current_output_path = os.path.join(output_dir, "0.txt")
        with open(current_output_path, mode="a") as f:
            f.write(text + "\n")

        # get all of the rest of the k-versions of this file
        for k in range(1,10+1):

            # read the current file
            current_file = os.path.join(input_dir, str(k) + "_" + file)
            text = ""  # initialize
            text = utils_general.read_file(os.path.join(input_dir, current_file))

            # write it out to the csv
            current_output_path = os.path.join(output_dir, str(k) + ".txt")
            with open(current_output_path, mode="a") as f:
                f.write(text + "\n")
