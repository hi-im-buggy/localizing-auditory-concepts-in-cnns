# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import numpy as np
import pandas as pd
import os
import plotly.express as px
from einops import rearrange
from torch_scatter import scatter

from utils.dataset import *
from utils.clustering import cluster_activations, get_cluster_embeddings
from utils.entropy import compute_entropies

import argparse

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='speech_commands_mini')
parser.add_argument('--ns_label', type=str, default='note_names')
parser.add_argument('--pcvc_label', type=str, default='vowels')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--class_limit', type=int, default=None)
parser.add_argument('--split_size', type=int, default=None)

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

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module_activation_dict = {
    # Conv blocks
    'audio_encoder.base.conv_block1': nn.Identity(),
    'audio_encoder.base.conv_block2': nn.Identity(),
    'audio_encoder.base.conv_block3': nn.Identity(),
    'audio_encoder.base.conv_block4': nn.Identity(),
    'audio_encoder.base.conv_block5': nn.Identity(),
    'audio_encoder.base.conv_block6': nn.Identity(),
    'audio_encoder.base.fc1': F.relu,
    # 'audio_encoder.projection.linear1': F.gelu,
    # 'audio_encoder.projection.linear2': nn.Identity(),
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
}[args.dataset]

if dataset == SpeechCommandsMiniDataset:
    dataset = dataset(seed=args.scm_seed)
elif dataset == MedleySolosDBDataset or dataset == LibriCountDataset:
    dataset = dataset(class_limit=args.class_limit)
elif dataset == NSynthDataset:
    dataset = dataset(classes=args.ns_label, class_limit=args.class_limit)
else:
    dataset = dataset()

# %%
# cluster_activations(dataset, module_list)
# %%

class_indices = torch.LongTensor([dataset.class_to_idx[label] for label in dataset.class_label])
num_classes = len(dataset.classes)

for layer_name in reversed(module_list):
    print(f'Layer {layer_name}')
    root = f'/scratch/pratyaksh.g/{dataset.path_name}/'
    activations = torch.load(root + f'activations/{layer_name}.pt')
    clusters = torch.load(root + f'clusters/{layer_name}.pt')
    n_clusters = torch.load(root + f'clusters/{layer_name}_n.pt')

    cluster_embeddings = get_cluster_embeddings(activations, clusters, n_clusters)
    cluster_activations = [rearrange(c_emb, 'c n -> n c') for c_emb in cluster_embeddings]
    cluster_entropies = [compute_entropies(c_act, class_indices, num_classes) for c_act in cluster_activations]

    data_dict = dict(
        layer_name=[],
        partition_idx=[],
        cluster_idx=[],
        entropy=[],
        neuron_count=[],
        top_class_idx=[],
        top_class=[],
    )

    for partition_idx in reversed(range(0, len(n_clusters))):
        num_clusters = n_clusters[partition_idx]
        if num_clusters > 1000:
            continue

        sorted_indices = torch.argsort(cluster_entropies[partition_idx], descending=False)

        # iterate through clusters, keeping track of the cluster's sorting rank/idx
        for sorting_idx, cluster_idx in tqdm(enumerate(sorted_indices), desc=f'Partition {partition_idx} / {len(n_clusters)}', total=len(sorted_indices)):
            # Just move on if it's a zero cluster
            current_cluster = cluster_embeddings[partition_idx][cluster_idx]

            zero = torch.zeros_like(current_cluster)
            if torch.all(torch.isclose(current_cluster, zero)):
                continue

            file_dir = f'/scratch/pratyaksh.g/{dataset.path_name}/cluster-plots/{layer_name}/partition-{partition_idx}/'
            os.makedirs(file_dir, exist_ok=True)
            num_digits = len(str(n_clusters[partition_idx].item()))
            padded_sorting_idx = str(sorting_idx).zfill(num_digits)

            # If it's not a zero cluster, then plot it's cluster embedding
            num_neurons = (clusters[:, partition_idx] == cluster_idx).sum().item()
            df = pd.DataFrame({
                'dataset_index': range(len(dataset)),
                'activations': current_cluster,
                'classes': dataset.class_label
            })

            title = f'Layer: {layer_name}, Partition: {partition_idx}, Cluster: {cluster_idx}'
            title += ', '
            title += f'Entropy: {cluster_entropies[partition_idx][cluster_idx]:.2f}'
            title += ', '
            title += f'Neuron Count: {(clusters[:, partition_idx] == cluster_idx).sum().item()}'

            fig = px.bar(df, x='classes', y='activations', title=title,
                        labels = {
                            'classes': f'classes ({dataset.name})',
                            'activations': 'average cluster activations',
                        },
                        hover_data='dataset_index')

            fig.write_image(f'{file_dir}/{padded_sorting_idx}-cluster-{cluster_idx}.pdf',
                            width=1920, height=1080, engine='kaleido')
            
            neurons_in_cluster = (clusters[:, partition_idx] == cluster_idx).nonzero().squeeze()

            if 'conv' in layer_name:
                n, c, w, h = activations.shape
                feature_map = torch.zeros((w, h))

                _, ts, fs = np.unravel_index(neurons_in_cluster.cpu().numpy(), (c, w, h))
                for t, f in zip(ts, fs):
                    feature_map[t, f] += 1

                fig = px.imshow(feature_map.cpu().numpy().T, origin='lower', color_continuous_scale='viridis', title=title)

                fig.write_image(f'{file_dir}/{padded_sorting_idx}-posmap-{cluster_idx}.pdf',
                                width=1920, height=1080, engine='kaleido')

            # Top class 
            classwise = scatter(current_cluster, class_indices)
            top_class_idx = torch.argmax(classwise)
            top_class = dataset.classes[top_class_idx]

            data_dict['layer_name'].append(layer_name)
            data_dict['partition_idx'].append(partition_idx)
            data_dict['cluster_idx'].append(cluster_idx.item())
            data_dict['entropy'].append(cluster_entropies[partition_idx][cluster_idx].item())
            data_dict['neuron_count'].append((clusters[:, partition_idx] == cluster_idx).sum().item())
            data_dict['top_class_idx'].append(top_class_idx.item())
            data_dict['top_class'].append(top_class)

    df = pd.DataFrame(data_dict)    
    os.makedirs(root + 'cluster-stats/', exist_ok=True)
    df.to_csv(root + f'cluster-stats/{layer_name}.csv', index=False)