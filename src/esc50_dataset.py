from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import pandas as pd
import os
import torch.nn as nn
import torch

class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta','esc50.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        # self.df['filename'] = os.path.join(self.root, self.base_folder, self.audio_dir) + os.sep + self.df['filename']
        self.df['category'] = self.df['category'].str.replace('_',' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}

        for filename, category in zip(self.df[self.file_col], self.df[self.label_col]):
            class_name = category.replace('_',' ')
            patch_name = filename.split('/')[-1].split('.')[0]
            class_idx = patch_name.split('-')[-1]
            self.class_to_idx[class_name] = int(class_idx)
        
        self.classes = sorted(self.class_to_idx.keys(), key=lambda x: self.class_to_idx[x])
        self.class_to_idx = {k: v for k, v in sorted(self.class_to_idx.items(), key=lambda item: item[1])}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1,-1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        download_url(self.url, self.root, self.filename)

        # extract file
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)