import glob
import os
import random
import cv2
import torch


class ImageDirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, path, pattern):
        self.paths = list(glob.glob(os.path.join(path, pattern)))

    def __len__(self):
        return len(self.paths)

    def __item__(self):
        path = random.choice(self.paths)
        return cv2.imread(path, 1)