from tqdm.auto import tqdm
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
import sys ; sys.path.append(f"{dir_path}\..")

from data.util import get_cocofake

if __name__ == "__main__":
    train, val, test, = get_cocofake("../data/coco-2014", "../data/cocofake", train_limit=1000, val_limit=500)
    breakpoint()
    for batch in tqdm(train):
        pass

    for batch in tqdm(val):
        pass

    for batch in tqdm(test):
        pass

    print("Successfully iterated through train, validation, and test sets.")
