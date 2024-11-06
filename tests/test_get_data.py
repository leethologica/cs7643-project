import sys ; sys.path.append("..")
from tqdm.auto import tqdm

from data.util.cocofake import get_cocofake

if __name__ == "__main__":
    train, val = get_cocofake("../data/coco-2014", "../data/cocofake", n=1000)
    for batch in tqdm(train):
        pass

    for batch in tqdm(val):
        pass

    print("Successfully iterated through train and validation sets.")
