# %%
from CLAPWrapper import CLAPWrapper
from utils.dataset import *
from utils.activations import precompute_audio_tensors, save_dataset_activations, save_dataset_activations_alt

import torch.nn as nn
import torch.nn.functional as F
import os
from icecream import ic

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='speech_commands_mini')
parser.add_argument('--ns_label', type=str, default='note_names')
parser.add_argument('--pcvc_label', type=str, default='vowels')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--class_limit', type=int, default=None)
parser.add_argument('--split_size', type=int, default=None)

args = parser.parse_args()

# %%
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

# %%
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
module_activation_dict = {
    # Conv blocks
    'audio_encoder.base.logmel_extractor': nn.Identity(),
    # 'audio_encoder.base.conv_block1': nn.Identity(),
    # 'audio_encoder.base.conv_block2': nn.Identity(),
    # 'audio_encoder.base.conv_block3': nn.Identity(),
    # 'audio_encoder.base.conv_block4': nn.Identity(),
    # 'audio_encoder.base.conv_block5': nn.Identity(),
    # 'audio_encoder.base.conv_block6': nn.Identity(),
    # 'audio_encoder.base.fc1': F.relu,
    # 'audio_encoder.projection.linear1': F.gelu,
    # 'audio_encoder.projection.linear2': nn.Identity(),
}

# %%
conv = lambda i: f'audio_encoder.base.conv_block{i}'
fc = 'audio_encoder.base.fc1'
proj = lambda i: f'audio_encoder.projection.linear{i}'

module_list = [
    ('audio_encoder.base.logmel_extractor', ),
]

# %%
weights_path = "/scratch/pratyaksh.g/clap/CLAP_weights_2022_microsoft.pth"
clap_model = CLAPWrapper(weights_path, use_cuda=True if DEVICE == "cuda" else False)

# %%
clap_model.clap.eval()

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
}[args.dataset]

if dataset in [SpeechCommandsDataset, SpeechCommandsMiniDataset, DurhamChordDataset]:
    dataset = dataset(seed=args.seed, device=DEVICE, class_limit=args.class_limit)
elif dataset in [MedleySolosDBDataset, LibriCountDataset, FrenchVowelsDataset, UrbanAcousticScenesDataset]:
    dataset = dataset(class_limit=args.class_limit, device=DEVICE)
elif dataset == NSynthDataset:
    dataset = dataset(classes=args.ns_label, class_limit=args.class_limit, device=DEVICE)
elif dataset == PCVCDataset:
    dataset = dataset(classes=args.pcvc_label, device=DEVICE)
else:
    dataset = dataset(device=DEVICE)

# precompute_audio_tensors(dataset, clap_model,
#                          save_path=f'/scratch/pratyaksh.g/{dataset.path_name}/audio-tensors/')

save_dataset_activations(dataset, clap_model, module_activation_dict, module_list)

# %%
# save_path = f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations1/'
# os.makedirs(save_path, exist_ok=True)
# save_dataset_activations(dataset, clap_model, module_activation_dict, module_list,
#                          start=0, end=3_000, save_path=save_path)

# save_path = f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations2/'
# os.makedirs(save_path, exist_ok=True)
# save_dataset_activations(dataset, clap_model, module_activation_dict, module_list,
#                          start=3_000, end=6_000, save_path=save_path)

# save_path = f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations3/'
# os.makedirs(save_path, exist_ok=True)
# save_dataset_activations(dataset, clap_model, module_activation_dict, module_list,
#                          start=6_000, end=9_000, save_path=save_path)

# save_path = f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations4/'
# os.makedirs(save_path, exist_ok=True)
# save_dataset_activations(dataset, clap_model, module_activation_dict, module_list,
#                          start=9_000, end=-1, save_path=save_path)

# for param in clap_model.clap.parameters():
#     param.to(DEVICE)

# os.makedirs(f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations/', exist_ok=True)
# ic()
# save_dataset_activations_alt(dataset, clap_model,
#                              module_activation_dict, module_list, batch_size=32,
#                              save_path=f'/scratch/pratyaksh.g/{dataset.path_name}/full-activations/',
#                              device=DEVICE)
