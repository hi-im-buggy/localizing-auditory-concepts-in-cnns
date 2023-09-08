# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from einops import rearrange
from icecream import ic
from tqdm.autonotebook import tqdm

from CLAPWrapper import CLAPWrapper
from utils.dataset import *
from utils.interventions import Intervention

import argparse

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
    'audio_encoder.base.conv_block1': nn.Identity(),    # 0
    'audio_encoder.base.conv_block2': nn.Identity(),    # 1
    'audio_encoder.base.conv_block3': nn.Identity(),    # 2
    'audio_encoder.base.conv_block4': nn.Identity(),    # 3
    'audio_encoder.base.conv_block5': nn.Identity(),    # 4
    'audio_encoder.base.conv_block6': nn.Identity(),    # 5
    'audio_encoder.base.fc1': F.relu,                   # 6
    # 'audio_encoder.projection.linear1': F.gelu,         # 7
    # 'audio_encoder.projection.linear2': nn.Identity(),  # 8
}

module_list = list(module_activation_dict.keys())

# %%
weights_path = "/scratch/pratyaksh.g/clap/CLAP_weights_2022_microsoft.pth"
clap_model = CLAPWrapper(weights_path, use_cuda=True if DEVICE == "cuda" else False)

# %%
clap_model.clap.eval()

# %%
probing_dataset = ESC50Dataset()
testing_dataset = ESC50Dataset()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--layer_idx', type=int)
parser.add_argument('--treatment', type=str, choices=['intervened', 'random'])
parser.add_argument('--invert_mask', type=str, choices=['true', 'false'])
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=24)

args = parser.parse_args()
layer_name = module_list[args.layer_idx]

# %%
activations = torch.load(f'/scratch/pratyaksh.g/{testing_dataset.path_name}/activations/{layer_name}.pt')
clusters = torch.load(f'/scratch/pratyaksh.g/{probing_dataset.path_name}/clusters/{layer_name}.pt')
n_clusters = torch.load(f'/scratch/pratyaksh.g/{probing_dataset.path_name}/clusters/{layer_name}_n.pt')
audio_tensors = torch.load(f'/scratch/pratyaksh.g/{testing_dataset.path_name}/audio-tensors/{testing_dataset.path_name}-audio-tensors.pt')

# %%
if len(activations.shape) > 2:
    n, c, w, h = activations.shape

# Computing text embeddings
prompt = 'this is a sound of '
y = [prompt + x for x in testing_dataset.classes]
text_embeddings = clap_model.get_text_embeddings(y)

clap_model.clap.to(DEVICE)
audio_tensors.to(DEVICE)

df = pd.read_csv(f'/scratch/pratyaksh.g/esc50/cluster-stats/{layer_name}.csv')
threshold = 5.26
df = df[df['entropy'] < threshold]
df = df.sort_values('neuron_count', ascending=False)
df = df.head(100)

for idx, record in tqdm(enumerate(df.iterrows()), total=len(df)):
    ic(idx, len(df))
    seed_everything(42)
    partition_idx = record[1]['partition_idx']
    cluster_idx = record[1]['cluster_idx']

    # %%
    performance = pd.DataFrame(
        columns=['accuracy', 'treatment', 'label']
    )

    # %%
    intervention = Intervention(clap_model, module_activation_dict)

    # %%
    intervention.clear_handles()

    cluster_mask = (clusters[:, partition_idx] == cluster_idx).bool()
    if args.invert_mask == 'true':
        cluster_mask = ~cluster_mask

    if args.treatment == 'random':
        num_neurons = clusters.shape[0]
        num_neurons_in_cluster = cluster_mask.sum().item()
        random_state = np.random.RandomState(42)
        random_neurons = random_state.choice(num_neurons, num_neurons_in_cluster, replace=False)
        random_cluster_mask = torch.zeros_like(cluster_mask.flatten())
        random_cluster_mask[random_neurons] = 1
        random_cluster_mask = random_cluster_mask.bool()
        cluster_mask = random_cluster_mask

    if len(activations.shape) > 2 and len(cluster_mask.shape) < 2:
        cluster_mask = rearrange(cluster_mask, '(c w h) -> c w h', c=c, w=w, h=h)
    intervention.set_intervention(activations, cluster_mask, layer_name, replace_with='random')
    # replace_with='random' here means that the activations of random instances are used to 'remove information',
    # and has nothing to do with intervention_type. The alternative type is replace_with='zero', which would
    # zero out the activations to 'remove information'

    # %%

    loader = DataLoader(list(zip(audio_tensors, testing_dataset.one_hot)), batch_size=args.batch_size, shuffle=False)

    # Computing audio embeddings
    for run in range(args.repeats):
        y_preds, y_labels = [], []
        desc_string = f"Run {run + 1} / {args.repeats} | {layer_name}, P{partition_idx}, C{cluster_idx} | treatment={args.treatment} | invert_mask={args.invert_mask} "
        for batch in tqdm(loader, desc=desc_string):
            audio_tensor, one_hot_target = batch

            audio_embeddings = clap_model.clap.audio_encoder(audio_tensor.to(DEVICE))[0]
            audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=-1, keepdim=True)

            similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings.to(DEVICE))

            y_pred = F.softmax(similarity.detach().cpu(), dim=-1).numpy()
            y_pred = np.argmax(y_pred, axis=-1)
            y_preds.append(y_pred)

            y_label = np.argmax(one_hot_target.detach().cpu().numpy(), axis=-1)
            y_labels.append(y_label)

        # %%
        confusion = confusion_matrix(np.concatenate(y_labels), np.concatenate(y_preds))
        class_wise_acc = confusion.diagonal() / confusion.sum(axis=1)

        for label_idx, label in enumerate(testing_dataset.classes):
            performance = performance.append({
                'accuracy': class_wise_acc[label_idx],
                'treatment': args.treatment,
                'label': label,
                'run': run,
            }, ignore_index=True)

    intervention.clear_handles()
    del intervention
    torch.cuda.empty_cache()

    # %%
    csv_path = f"/scratch/pratyaksh.g/{testing_dataset.path_name}/intervened-performance/{layer_name}/"
    os.makedirs(csv_path, exist_ok=True)
    expt_id = f"partition-{partition_idx}-cluster-{cluster_idx}-{args.treatment}-invert_mask={str(args.invert_mask)}.csv"
    performance.to_csv(csv_path + expt_id, index=False)