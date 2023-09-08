# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import *
from utils.layerwise import save_pairwise_distances, save_layerwise_plots, save_layerwise_plots_faiss

import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='speech_commands_mini')
parser.add_argument('--ns_label', type=str, default='note_names')
parser.add_argument('--pcvc_label', type=str, default='vowels')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--class_limit', type=int, default=None)
parser.add_argument('--split_size', type=int, default=None)
parser.add_argument('--compute_pdist', action='store_true')

args = parser.parse_args()

# %%
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# %%
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module_activation_dict = {
    # Spectrograms
    'audio_encoder.base.logmel_extractor': nn.Identity(),
    # Conv blocks
    'audio_encoder.base.conv_block1': nn.Identity(),
    'audio_encoder.base.conv_block2': nn.Identity(),
    'audio_encoder.base.conv_block3': nn.Identity(),
    'audio_encoder.base.conv_block4': nn.Identity(),
    'audio_encoder.base.conv_block5': nn.Identity(),
    'audio_encoder.base.conv_block6': nn.Identity(),
    # Linear layers
    'audio_encoder.base.fc1': F.relu,
    'audio_encoder.projection.linear1': F.gelu,
    'audio_encoder.projection.linear2': nn.Identity(),
}

module_list = list(module_activation_dict.keys())

# %%
dataset = {
    'esc50': ESC50Dataset,
    'gtzan': GTZANDataset,
    'count_to_five': CountToFiveDataset,
    'speech_commands': SpeechCommandsDataset,
    'speech_commands_mini': SpeechCommandsMiniDataset,
    'nsynth': NSynthDataset,
    'medley_solos_db': MedleySolosDBDataset,
    'libricount': LibriCountDataset,
    'french_vowels': FrenchVowelsDataset,
    'pcvc': PCVCDataset,
    'durham_chord': DurhamChordDataset,
    'urban_acoustic_scenes': UrbanAcousticScenesDataset,
    'fma_small': FMASmallDataset,
}[args.dataset]

if dataset in [SpeechCommandsDataset, SpeechCommandsMiniDataset, DurhamChordDataset, FMASmallDataset]:
    dataset = dataset(seed=args.seed, device=DEVICE, class_limit=args.class_limit)
elif dataset in [MedleySolosDBDataset, LibriCountDataset, FrenchVowelsDataset, UrbanAcousticScenesDataset]:
    dataset = dataset(class_limit=args.class_limit, device=DEVICE)
elif dataset == NSynthDataset:
    dataset = dataset(classes=args.ns_label, class_limit=args.class_limit, device=DEVICE)
elif dataset == PCVCDataset:
    dataset = dataset(classes=args.pcvc_label, device=DEVICE)
else:
    dataset = dataset(device=DEVICE)
# %%
print(f'Dataset size: {len(dataset)}')

if args.compute_pdist:
    save_pairwise_distances(dataset, list(reversed(module_list)))

# %%
groups = isinstance(dataset, MedleySolosDBDataset)
save_layerwise_plots(dataset, module_list, groups=groups)
