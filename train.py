import argparse
from trainer import Trainer
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def save_embedding(solver, dataset, fname):
    latent, labels, depth = solver.encode_adv(dataset = dataset)
    result = latent.numpy()
    np.savetxt(fname, result, delimiter = '\t')
    print(f'Translator embedding saved to {fname}!')
    return result, labels

def plot_embedding(embedding, labels, fname, title = 'Translator Embedding'):
    reducer = umap.UMAP(random_state=123, n_neighbors=30, min_dist=0.3, n_components=2, metric='cosine')
    l = pd.DataFrame(labels, columns=['celltype'])
    X_embedded = reducer.fit_transform(embedding)
    plt.clf()
    plt.figure(figsize=(16, 12))
    for i, c in enumerate(np.unique(l)):
        mask = (l == c).values.flatten()
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=c, s=8, color=cmap(i), alpha=0.8)
        plt.xticks([], [])
        plt.yticks([], [])
    plt.legend(loc = 'upper right', fontsize = 16)
    plt.title(title, fontsize = 24)
    plt.savefig(fname, dpi = 600, bbox_inches = 'tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translator')
    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('-l', '--load_ckpt', default=False, type=str, help='path to ckpt loaded')
    parser.add_argument('-cuda', '--cuda_dev', default=[0], type=list, help='GPU want to use')
    parser.add_argument('--sample_batch', default=False, type=bool, help='Add batch effect correction')
    parser.add_argument('--max_epoch', default=400, type=int, help='maximum training epoch')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
    parser.add_argument('-b', '--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--start_save', default=350, type=int, help='epoch starting to save models')
    parser.add_argument('--conv', default=False, type=bool, help='use conv vae')
    parser.add_argument('--model_type', default='inv', type=str, help='model type')
    parser.add_argument('-d', '--data_type', type=str, help='dataset')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--pos_w', default=30, type=float, help='BCE positive weight')
    parser.add_argument('--weight_decay', default=5e-4, type=str, help='weight decay for adam')
    parser.add_argument('--z_dim', default=10, type=int, help='latent dim')
    parser.add_argument('--out_every', default=1, type=int, help='save ckpt every x epoch')
    parser.add_argument('--ckpt_dir', default='./models/', type=str, help='out directory')
    parser.add_argument('--LAMBDA', default=1, type=float, help='lambda value')
    parser.add_argument('--file_name' default = None, type = str, help = 'Path to the reference dataset (npz file)')
    parser.add_argument('--type_name' default = None, type = str, help = 'Path to the ground truth (label) of the reference dataset (tsv file)')
    parser.add_argument('--file_name2' default = None, type = str, help = 'Path to the target dataset (npz file)')
    parser.add_argument('--type_name2' default = None, type = str, help = 'Path to the ground truth (label) of the target dataset (tsv file)')
    args = parser.parse_args()

    solver = Trainer(args)
    solver.warm_up()
    solver.inv_train()

    if args.data_type == 'SimDataset':
        latent, labels = save_embedding(solver, solver.dataset, os.path.join('results', f"Translator_embedding_{name}.tsv"))

        plot_embedding(latent, labels, os.path.join('results', f'Translator_embedding_{name}_UMAP.png')) 

    else:
        latent, labels = save_embedding(solver, solver.dataset.hd_dataset, os.path.join('results', f"Translator_embedding_{name}_reference.tsv"))
        plot_embedding(latent, labels, os.path.join('results', f'Translator_embedding_{name}_reference_UMAP.png')) 

        latent, labels = save_embedding(solver, solver.dataset.ld_dataset, os.path.join('results', f"Translator_embedding_{name}_target.tsv"))
        plot_embedding(latent, labels, os.path.join('results', f'Translator_embedding_{name}_target_UMAP.png')) 