import os
import random
import shutil

###############################################################################
#
# Change these 4 parameters if needed.

TRAIN = 0.8
VALIDATE = 0.1
TEST = 0.1
OUT_PATH = "datasets/mydataset" # Must start with datasets/.

FILES_PER_INPUT = 2 # 1 for the image and 1 for the ground truth.
DATA_PATH = "dataset-raw/agri_data/data"

#
###############################################################################


def get_input_filenames_without_extension(path):
    """
    Returns a list of the filenames for all the inputs, without the extension.
    In other words, if the directory has the following files:
    ```
        agri_0_0.jpeg
        agri_0_0.txt
        agri_0_1.jpeg
        agri_0_0.txt
    ```
    The output will be:
    ```
        ['agri_0_0', 'agri_0_1']
    ```
    """

    f = []
    for (_, _, filenames) in os.walk(DATA_PATH):
        f.extend([f.split(".")[0] for f in filenames])
        break
    return list(set(f))


def split_input_filenames(fnames, train_size, validate_size, test_size):
    random.shuffle(fnames)

    a = int(len(fnames) * train_size)
    b = int(len(fnames) * validate_size)
    c = int(len(fnames) * test_size)
    assert(a + b + c == len(fnames))

    train_set = fnames[0:a]
    validate_set = fnames[a:a+b]
    test_set = fnames[a+b:]

    return train_set, validate_set, test_set


def create_output_dirs_just_in_case(out_path):
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{out_path}/train/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{out_path}/validate/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{out_path}/test/"), exist_ok=True)


def copy_data(train, validate, test, out_path):
    for t in train:
        f1 = f"{DATA_PATH}/{t}.jpeg"
        f2 = f"{DATA_PATH}/{t}.txt"
        shutil.copy2(f1, f"./{out_path}/train/")
        shutil.copy2(f2, f"./{out_path}/train/")
    for t in validate:
        f1 = f"{DATA_PATH}/{t}.jpeg"
        f2 = f"{DATA_PATH}/{t}.txt"
        shutil.copy2(f1, f"./{out_path}/validate/")
        shutil.copy2(f2, f"./{out_path}/validate/")
    for t in test:
        f1 = f"{DATA_PATH}/{t}.jpeg"
        f2 = f"{DATA_PATH}/{t}.txt"
        shutil.copy2(f1, f"./{out_path}/test/")
        shutil.copy2(f2, f"./{out_path}/test/")


if __name__ == "__main__":
    assert(OUT_PATH.startswith("datasets/"))
    _, _, files = next(os.walk(DATA_PATH))
    assert(len(files) % FILES_PER_INPUT == 0) 

    input_names = get_input_filenames_without_extension(DATA_PATH)
    train, validate, test = split_input_filenames(input_names, TRAIN, VALIDATE, TEST)
    create_output_dirs_just_in_case(OUT_PATH)
    copy_data(train, validate, test, OUT_PATH)