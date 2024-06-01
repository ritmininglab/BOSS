import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys, psutil
import argparse
import pickle
from datetime import datetime

from torchvision import models
import pdb
import numpy as np

from core.model_generator import wideresnet, preact_resnet, resnet
from core.model_generator.efficientnet import EfficientNet
from core.model_generator.vit import vit
from core.training import Trainer, TrainingDynamicsLogger
from core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, TinyDataset
from core.utils import print_training_info, StdRedirect

import matplotlib.pyplot as plt
from selection import get_distance, get_prune_idx

import scipy
from scipy.optimize import nnls
from numpy import inf

def get_his_den(P, R, x):
    for i, v in enumerate(P):
        if i != len(P) - 2:
            if R[i] <= x < R[i+1]:
                return v
        else:
            return v

class optimizer(object):
    def __init__(self, index, budget:int, already_selected=[], importance=None, diff_change_rate=0.01):
        # self.args = args
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected
        self.importance = importance
        self.diff_change_rate = diff_change_rate


class LazyGreedy(optimizer):
    def __init__(self, index, budget:int, already_selected=[], importance=None,):
        super(LazyGreedy, self).__init__(index, budget, already_selected, importance)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        # print(gain_function(~selected, selected, **kwargs))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs) * self.importance[~selected]
        # greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)

        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            best_gain = -np.inf
            last_max_element = -1
            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    # Select cur_max_element into the current subset
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0] * self.importance[cur_max_element]
                # new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element
        return self.index[selected]


class LazyGreedyDynamic(optimizer):
    def __init__(self, index, budget:int, already_selected=[], importance=None, diff_change_rate=0.01):
        super(LazyGreedyDynamic, self).__init__(index, budget, already_selected, importance)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        el2n = self.importance
        c_param = 0
        pi = np.pi
        SS = c_param
        alpha = 1
        difficulty_importance = ((np.sin(pi*el2n - (3/2)*pi - SS*pi) + 1)/2)**alpha
        diff = difficulty_importance.max() - difficulty_importance.min()
        if diff > 0.001:    
            difficulty_importance = (difficulty_importance - difficulty_importance.min())/(diff)

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs) * difficulty_importance[~selected]
        # greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)

        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            best_gain = -np.inf
            last_max_element = -1

            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    # Select cur_max_element into the current subset
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0] * difficulty_importance[cur_max_element]
                # new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element

            c_param += self.diff_change_rate
            pi = np.pi
            SS = c_param
            alpha = 1
            difficulty_importance = ((np.sin(pi*el2n - (3/2)*pi - SS*pi) + 1)/2)**alpha
            diff = difficulty_importance.max() - difficulty_importance.min()
            if diff > 0.001:    
                difficulty_importance = (difficulty_importance - difficulty_importance.min())/(diff)

            greedy_gain = np.zeros(len(self.index))
            greedy_gain[selected] = -np.inf
            greedy_gain[~selected] = gain_function(~selected, selected, **kwargs) * difficulty_importance[~selected]


        return self.index[selected]


class StochasticGreedy(optimizer):
    def __init__(self, index, budget:int, already_selected=[], epsilon: float=0.9):
        super(StochasticGreedy, self).__init__(index, budget, already_selected)
        self.epsilon = epsilon

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        sample_size = max(round(-np.log(self.epsilon) * self.n / self.budget), 1)

        greedy_gain = np.zeros(len(self.index))
        all_idx = np.arange(self.n)
        for i in range(sum(selected), self.budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, self.budget))

            # Uniformly select a subset from unselected samples with size sample_size
            subset = np.random.choice(all_idx[~selected], replace=False, size=min(sample_size, self.n - i))

            if subset.__len__() == 0:
                break

            greedy_gain[subset] = gain_function(subset, selected, **kwargs)
            current_selection = greedy_gain[subset].argmax()
            selected[subset[current_selection]] = True
            greedy_gain[subset[current_selection]] = -np.inf
            if update_state is not None:
                update_state(np.array([subset[current_selection]]), selected, **kwargs)
        return self.index[selected]
    

class NaiveGreedy(optimizer):
    def __init__(self, index, budget:int, already_selected=[], importance=None,):
        super(NaiveGreedy, self).__init__(index, budget, already_selected, importance)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        for i in range(sum(selected), self.budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
            current_selection = greedy_gain.argmax()
            selected[current_selection] = True
            greedy_gain[current_selection] = -np.inf
            if update_state is not None:
                update_state(np.array([current_selection]), selected, **kwargs)
        return self.index[selected]


class SubmodularFunction(object):
    def __init__(self, index, importance, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)
        self.importance = importance

        # start from the vacancy
        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
        # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
        # calculate similarities incrementally at a later time if required.

        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            # Previous method utilizes cosine similarity between each gradient
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


class FacilityLocation(SubmodularFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.already_selected.__len__()==0:
            self.cur_max = np.zeros(self.n, dtype=np.float32)
        else:
            self.cur_max = np.max(self.similarity_kernel(np.arange(self.n), self.already_selected), axis=1)

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):
        gains = np.maximum(0., self.similarity_kernel(self.all_idx, idx_gain) - self.cur_max.reshape(-1, 1)).sum(axis=0)

        # gains = gains * self.importance

        return gains

    def calc_gain_batch(self, idx_gain, selected, **kwargs):
        batch_idx = ~self.all_idx
        batch_idx[0:kwargs["batch"]] = True
        gains = np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=0)
        for i in range(kwargs["batch"], self.n, kwargs["batch"]):
            batch_idx = ~self.all_idx
            batch_idx[i * kwargs["batch"]:(i + 1) * kwargs["batch"]] = True
            gains += np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1,1)).sum(axis=0)
        return gains

    def update_state(self, new_selection, total_selected, **kwargs):
        self.cur_max = np.maximum(self.cur_max, np.max(self.similarity_kernel(self.all_idx, new_selection), axis=1))


def hessian_pick_var(hessians,K=100):
    dominant_theta = np.var(hessians,axis=0)
    sorted_var = np.sort(dominant_theta)[::-1]

    argsort = np.argsort(dominant_theta)[::-1]
    pick = argsort[:K]
    hessians_reduced = hessians[:,pick]
    return hessians_reduced, pick, sorted_var


def l2_norm_np(v1, v2):
    m = v1.shape[0]  # x has shape (m, d)
    n = v2.shape[0]  # y has shape (n, d)
    x2 = np.sum(v1 ** 2, axis=1).reshape((m, 1))
    y2 = np.sum(v2 ** 2, axis=1).reshape((1, n))
    xy = v1.dot(v2.T)  # shape is (m, n)
    dists = np.sqrt(x2 + y2 - 2 * xy)  # shape is (m, n)
    dists[np.isnan(dists)] = 0

    return dists


def l1_norm_np(v1, v2):
    l1_dist = np.abs(v1[:, None, :] - v2[None, :, :]).sum(axis=-1)
    return l1_dist


def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res


model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet34', 'eff', 'vit'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'cinic10','tiny-imagenet-200'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', 
                    type=str, 
                    choices=['random', 'coreset', 'stratified', 'submod', 
                             'boss', 'stratified_submod', 'moderate', 
                             'new', 'grad_match', 'adacore'])

parser.add_argument('--data-score-path', type=str)
parser.add_argument('--coreset-key', type=str)
parser.add_argument('--data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0,
                    help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)

#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str)
parser.add_argument('--mis-data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float,
                    help="Ratio for the coreset, not the whole dataset.")

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

######################### Setting for Future Use #########################
# parser.add_argument('--ckpt-name', type=str, default='model.ckpt',
#                     help='The name of the checkpoint.')
# parser.add_argument('--lr-scheduler', choices=['step', 'cosine'])
# parser.add_argument('--network', choices=model_names, default='resnet18')
# parser.add_argument('--pretrained', action='store_true')
# parser.add_argument('--augment', choices=['cifar10', 'rand'], default='cifar10')

parser.add_argument('--c_param', type=float, help='importance parameter C', dest="c_param", default=0)
parser.add_argument('--a_param', type=float, help='importance parameter alpha', dest="a_param", default=1)
parser.add_argument('--baseline', type=int, help='if running baseline', dest="baseline", default=0)

parser.add_argument('--initial_train', type=int, help='if initial training', dest="initial_train", default=0)

parser.add_argument('--el2n_epoch', type=int, help='el2n_epoch', dest="el2n_epoch", default=10)

args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"


print(f'Dataset: {args.dataset}')
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name) 
os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')

######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join(args.data_dir, args.dataset)
print(f'Data dir: {data_dir}')

if args.dataset == 'cifar10':
    td_path_0 = "model/cifar10/all-data/td-all-data.pickle"
    # td_path_0 = "model/eff-cifar10/eff-cifar10/td-eff-cifar10.pickle"
    trainset = CIFARDataset.get_cifar10_train(data_dir)
elif args.dataset == 'cifar100':
    # td_path_0 = "model/cifar100/all-data/td-all-data.pickle"
    td_path_0 = "model/eff-cifar100/eff-cifar100/td-eff-cifar100.pickle"
    trainset = CIFARDataset.get_cifar100_train(data_dir)
elif args.dataset == 'svhn':
    td_path_0 = "model/svhn/all-data/td-all-data.pickle"
    trainset = SVHNDataset.get_svhn_train(data_dir)
elif args.dataset == 'cinic10':
    td_path_0 = "model/cinic10/all-data/td-all-data.pickle"
    trainset = CINIC10Dataset.get_cinic10_train(data_dir)
elif args.dataset == 'tiny-imagenet-200':
    td_path_0 = "model/tiny/all-data/td-all-data.pickle"
    trainset = TinyDataset.get_tiny_train(data_dir)

if args.dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif args.dataset == 'cifar100':
    num_classes=100
else:
    num_classes=200

print("num_class: ", num_classes)

if args.network == 'resnet18':
    print('resnet18')
    model = resnet('resnet18', num_classes=num_classes, device=device, initial_train=args.initial_train)
if args.network == 'resnet50':
    print('resnet50')
    model = resnet('resnet50', num_classes=num_classes, device=device)
if args.network == 'resnet34':
    print('resnet34')
    model = resnet('resnet34', num_classes=num_classes, device=device, initial_train=args.initial_train)
if args.network == 'eff':
    print('EfficientNet')
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    model = EfficientNet(cfg, num_classes=num_classes, initial_train=args.initial_train)
if args.network == 'vit':
    print('ViT')
    model = vit(num_classes=num_classes, initial_train=args.initial_train)


def reduce_train_dynamics(td_path_0):
    training_dynamics_reduced = []
    with open(td_path_0, 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']
    for x in training_dynamics:
        
        x['hessian'] = None
        x['gradient'] = None
        if x['epoch'] == 10:
            training_dynamics_reduced.append(x)
        else:
            x = None
    pickled_data = {
            'data-name': "tiny-imagenet-200",
            'training_dynamics': training_dynamics_reduced,
        }
    filepath = td_path_0 + ".feature"
    with open(filepath, 'wb') as handle:
        pickle.dump(pickled_data, handle)
    # TD_logger.save_training_dynamics(td_path, data_name=args.dataset)
    return training_dynamics_reduced

def get_feature_target(total_num, training_dynamics, data_score, coreset_key, c_indx=None):
    # with open(td_path_0 + ".feature", 'rb') as f:  ## for imagenet200
    with open(td_path_0, 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']
    
    # features_all = np.zeros((total_num, 512))
    features_all = np.zeros((total_num, 320))
    target_all = data_score['targets'].numpy().astype(float)

    for x in training_dynamics:
        x['hessian'] = None
        x['gradient'] = None
        if x['epoch'] == 10:
            features_all[x['idx']] = x['feature']
    return features_all, target_all

def get_feature(total_num):
    with open(td_path_0 + ".feature", 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']
    features_all = np.zeros((total_num, 512))
    for x in training_dynamics:
        if x['epoch'] == 10:
            features_all[x['idx']] = x['feature']
    return features_all

def get_gradient(total_num, c_indx=None):
    with open(td_path_0 + ".gradient", 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']
        
    total_num = len(c_indx)
    gradients_all = np.zeros((total_num, 102600))
    # gradients_all = np.zeros((total_num, 5130))
    for x in training_dynamics:
        if x['epoch'] == 10:
            for i,c in enumerate(c_indx):
                if c in x['idx']:
                    gradients_all[i] = x['gradient'][np.isin(x['idx'], c)]

    return gradients_all

def get_hessian(total_num, c_indx=None):
    with open(td_path_0 + ".hessian", 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']
    
    total_num = len(c_indx)
    hessians_all = np.zeros((total_num, 102600))
    # hessians_all = np.zeros((total_num, 5130))
    for x in training_dynamics:
        if x['epoch'] == 10:
            for i,c in enumerate(c_indx):
                if c in x['idx']:
                    hessians_all[i] = x['hessian'][np.isin(x['idx'], c)]
    hessians_reduced_all, pick_idx_var, var_statistics = hessian_pick_var(hessians_all, 100)

    del hessians_all
    return hessians_reduced_all

def get_tr_dyn(total_num, training_dynamics, data_score, coreset_key, dataset, el2n_epoch=None, c_indx=None):
    # with open(td_path_0+".feature", 'rb') as f:
    with open(td_path_0, 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']

    print("Memory used by td: ")
    process = psutil.Process()
    print(process.memory_info().rss/(1024*1024*1024), "GB")

    if args.dataset == 'cifar100':
        gradients_all = np.zeros((total_num, 51300))
        hessians_all = np.zeros((total_num, 51300))
    else:
        gradients_all = np.zeros((total_num, 5130))
        hessians_all = np.zeros((total_num, 5130))
    # features_all = np.zeros((total_num, 512))
    # features_all = np.zeros((len(c_indx), 512))
    el2n_all = data_score[coreset_key].numpy().astype(float)

    # gradients_all = np.zeros((total_num, 102600))
    # hessians_all = np.zeros((total_num, 102600))

    # gradients_all = np.zeros((total_num, 5130))
    # hessians_all = np.zeros((total_num, 5130))

    features_all = np.zeros((total_num, 320))

    for x in training_dynamics:
        if x['epoch'] == 10:
            # hessians_all[x['idx']] = x['hessian']
            # x['hessian'] = None
            # gradients_all[x['idx']] = x['gradient']
            # x['gradient'] = None
            # pdb.set_trace()
            features_all[x['idx']] = x['feature']
    
    # inverse_hessians = np.reciprocal(hessians_all*total_num)
    # nan_indices = np.isnan(inverse_hessians)
    # inf_indices = np.isinf(inverse_hessians)
    # inverse_hessians[inverse_hessians == inf] = 1
    
    # inverse_hessians = np.where(inf_indices, 0, 1 / (hessians_all*total_num))
    
    # pdb.set_trace()
    # gradients_all = gradients_all * inverse_hessians
    
    # hessians_reduced_all, pick_idx_var, var_statistics = hessian_pick_var(hessians_all, 100)
    # del hessians_all
    # print(".",end="")

    # targets = []
    data_size = len(dataset)

    # for i in range(data_size):
    #     if ds == "tiny-imagenet-200":
    #         _, (_,_,y) = dataset[i]
    #     else:
    #         _, (_, y) = dataset[i]
    #     targets.append(y)
    # if args.dataset == "svhn":
    #     targets = torch.tensor(dataset.labels)
    # else:
    #     targets = torch.tensor(dataset.targets)
    # l2_loss = torch.nn.MSELoss(reduction='none')
    # el2n_list = torch.zeros(data_size).type(torch.float32)

    # for x in training_dynamics:
        # if x['epoch'] <= el2n_epoch: 
        #     output = torch.exp(x['output'].type(torch.float))
        #     index = x['idx'].type(torch.long)
        #     label = targets[index]
        #     label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        #     
        #     el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))
        #     el2n_list[index] += el2n_score
        # x['hessian'] = None
        # x['gradient'] = None
        # if x['epoch'] == 10:
            
            # hessians_all[x['idx']] = x['hessian']
            # x['hessian'] = None
            # gradients_all[x['idx']] = x['gradient']
            # x['gradient'] = None
            # features_all[x['idx']] = x['feature']

            # for i,c in enumerate(c_indx):
            #     if c in x['idx']:
            #         features_all[i] = x['feature'][np.isin(x['idx'], c)]

            # x_idx_list = x['idx'][np.isin(x['idx'], c_indx)]
            # c_idx_list = 
            # features_all[idx_list] = x['feature'][np.isin(x['idx'], c_indx)]
            # if x['idx'] in c_indx:
            #     features_all[x['idx']] = x['feature']
    gradients_all = None
    hessians_reduced_all = None
    # features_all = None
    return gradients_all, hessians_reduced_all, el2n_all, features_all

def orthogonal_matching_pursuit(A, b, budget: int, lam: float = 0.5):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Acknowlegement to:
    https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
    Args:
        A: design matrix of size (d, n)
        b: measurement vector of length d
        budget: selection budget
        lam: regularization coef. for the final output vector
    Returns:
        vector of length n
    '''
    with torch.no_grad():
        d, n = A.shape
        if budget <= 0:
            budget = 0
        elif budget > n:
            budget = n

        x = np.zeros(n, dtype=np.float32)
        resid = b.clone()
        indices = []
        boolean_mask = torch.ones(n, dtype=bool, device="cuda")
        all_idx = torch.arange(n, device='cuda')

        #몇 개나 고를 수 있는지 선택하기.

        for i in range(budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
            projections = torch.matmul(A.T, resid)
            # shape (n , 1)
            # print(projections.shape)
            index = torch.argmax(projections[boolean_mask])
            index = all_idx[boolean_mask][index]

            indices.append(index.item())
            boolean_mask[index] = False

            if indices.__len__() == 1:
                A_i = A[:, index]
                x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                A_i = A[:, index].view(1, -1)
            else:
                #A_i가 하나씩 추가하는 모듈
                A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device="cuda")
                
                x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
        if budget > 1:
            x_i = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())[0]
            x[indices] = x_i
        elif budget == 1:
            x[indices[0]] = 1.
    return x

def update_val_gradients(self, new_selection, selected_for_train):

    sum_selected_train_gradients = torch.mean(self.train_grads[selected_for_train], dim=0)

    new_outputs = self.init_out - self.eta * sum_selected_train_gradients[:self.args.num_classes].view(1,
                    -1).repeat(self.init_out.shape[0], 1) - self.eta * torch.matmul(self.init_emb,
                    sum_selected_train_gradients[self.args.num_classes:].view(self.args.num_classes, -1).T)

    sample_num = new_outputs.shape[0]
    gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)], requires_grad=False)
    i = 0
    while i * self.args.selection_batch < sample_num:
        batch_indx = np.arange(sample_num)[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch,
                                                                                sample_num)]
        new_out_puts_batch = new_outputs[batch_indx].clone().detach().requires_grad_(True)
        loss = self.criterion(torch.nn.functional.softmax(new_out_puts_batch, dim=1), self.init_y[batch_indx])
        batch_num = len(batch_indx)
        bias_parameters_grads = torch.autograd.grad(loss.sum(), new_out_puts_batch, retain_graph=True)[0]

        weight_parameters_grads = self.init_emb[batch_indx].view(batch_num, 1, self.embedding_dim).repeat(1,
                                    self.args.num_classes, 1) * bias_parameters_grads.view(batch_num,
                                    self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
        gradients[batch_indx] = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu()
        i += 1

    self.val_grads = torch.mean(gradients, dim=0)


model = model.to(device)

######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)

if args.coreset:
    with open(args.data_score_path, 'rb') as f:
        data_score = pickle.load(f)

    # import pandas as pd
    # el2n = data_score['el2n_total']
    # forget = data_score['forgetting']
    # aum = data_score['accumulated_margin']
    # df = pd.DataFrame({'el2n':el2n,'forget':forget,'aum':aum})  
    # df.corr()

    if args.coreset_mode == 'glister':
        train_indx = np.arange(total_num)
        selection_result = np.array([], dtype=np.int64)

        el2n_all = data_score[coreset_key].numpy().astype(float)
        gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key, trainset, args.el2n_epoch)  

        for c in range(num_classes):
            print(c, end=' ')
            if args.dataset == "svhn":
                c_indx_0 = train_indx[np.array(trainset.labels) == c]
            else:
                c_indx_0 = train_indx[np.array(trainset.targets) == c]   

            data_score_0 = data_score.copy()
            data_score_0[coreset_key] = data_score_0[coreset_key][c_indx_0]
            if coreset_key != 'accumulated_margin':
                data_score_0['accumulated_margin'] = data_score_0['accumulated_margin'][c_indx_0]
            
            mis_num = 0
            data_score_1, score_index_1 = CoresetSelection.mislabel_mask(data_score_0, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)
            del data_score_0, data_score_1

            c_indx = c_indx_0[score_index_1]
            c_sel_len = len(c_indx)

            cur_gradients = gradients_all[c_indx]

            # submod_function = FacilityLocation(index=c_indx, 
            #                                    importance = np.ones((c_sel_len,1)),
            #                                    similarity_kernel=lambda a,b: 10 - l2_norm_np(cur_gradients[a], cur_gradients[b]))
            submod_optimizer = LazyGreedy(index=c_indx, budget=round(args.coreset_ratio * len(c_indx_0)), already_selected=[], importance = np.ones(c_sel_len))

            c_selection_result = submod_optimizer.select(gain_function=lambda idx_gain, selected,
                                                             **kwargs: torch.matmul(cur_gradients[idx_gain],
                                                             cur_gradients.view(-1, 1)).detach().cpu().numpy().
                                                             flatten(), upadate_state=self.update_val_gradients)


    if args.coreset_mode == 'adacore':
        train_indx = np.arange(total_num)
        selection_result = np.array([], dtype=np.int64)

        el2n_all = data_score[coreset_key].numpy().astype(float)
        gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key, trainset, args.el2n_epoch)  
       
        ## save the gradients:
        # pickled_data = {
        #     'data-name': "cifar-100",
        #     'training_dynamics': gradients_all,
        # }
        # filepath = td_path_0 + "fix" + ".pregrad"
        # with open(filepath, 'wb') as handle:
        #     pickle.dump(pickled_data, handle, protocol=4) 
        # pdb.set_trace()

        ## CIFAR100:
        # filepath = td_path_0 + "fix" + ".pregrad"
        # with open(filepath, 'rb') as f:
        #     pickled_data = pickle.load(f)
        #     gradients_all = pickled_data['training_dynamics']

        for c in range(num_classes):
            print(c, end=' ')
            if args.dataset == "svhn":
                c_indx_0 = train_indx[np.array(trainset.labels) == c]
            else:
                c_indx_0 = train_indx[np.array(trainset.targets) == c]   

            data_score_0 = data_score.copy()
            data_score_0[coreset_key] = data_score_0[coreset_key][c_indx_0]
            if coreset_key != 'accumulated_margin':
                data_score_0['accumulated_margin'] = data_score_0['accumulated_margin'][c_indx_0]
            
            mis_num = 0
            data_score_1, score_index_1 = CoresetSelection.mislabel_mask(data_score_0, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)
            del data_score_0, data_score_1

            c_indx = c_indx_0[score_index_1]
            c_sel_len = len(c_indx)

            # save the gradients for current class c:
            # pickled_data = {
            #     'data-name': "tiny-imagenet-200",
            #     'training_dynamics': gradients_all[c_indx],
            # }
            # filepath = td_path_0 + str(c) + "fix.pregrad"
            # with open(filepath, 'wb') as handle:
            #     pickle.dump(pickled_data, handle) 
            # continue

            # print('load file')
            # filepath = td_path_0 + str(c) + "fix.pregrad"
            # with open(filepath, 'rb') as f:
            #     pickled_data = pickle.load(f)
            #     cur_gradients = pickled_data['training_dynamics']

            # cur_gradients = gradients_all[c_indx]

            # submod_function = FacilityLocation(index=c_indx, 
            #                                    importance = np.ones((c_sel_len,1)),
            #                                    similarity_kernel=lambda a,b: 10 - l2_norm_np(cur_gradients[a], cur_gradients[b]))
            
            submod_function = FacilityLocation(index=c_indx,
                                               importance = np.ones((c_sel_len,1)),
                                               similarity_kernel=lambda a,b: cossim_np(cur_gradients[a], cur_gradients[b]))


            submod_optimizer = LazyGreedy(index=c_indx, budget=round(args.coreset_ratio * len(c_indx_0)), already_selected=[], importance = np.ones(c_sel_len))

            # print('select')
            c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                            update_state=submod_function.update_state)
            selection_result = np.append(selection_result, c_selection_result)
            del c_selection_result
        coreset_index = selection_result

    # print("Coreset Size: ", coreset_index.shape)
    # pdb.set_trace()

    if args.coreset_mode == 'grad_match':
        train_indx = np.arange(total_num)
        selection_result = np.array([], dtype=np.int64)

        el2n_all = data_score[coreset_key].numpy().astype(float)
        gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key, trainset, args.el2n_epoch)  

        for c in range(num_classes):
            print(c, end=' ')
            if args.dataset == "svhn":
                c_indx_0 = train_indx[np.array(trainset.labels) == c]
            else:
                c_indx_0 = train_indx[np.array(trainset.targets) == c]   

            data_score_0 = data_score.copy()
            data_score_0[coreset_key] = data_score_0[coreset_key][c_indx_0]
            if coreset_key != 'accumulated_margin':
                data_score_0['accumulated_margin'] = data_score_0['accumulated_margin'][c_indx_0]
            
            mis_num = 0
            data_score_1, score_index_1 = CoresetSelection.mislabel_mask(data_score_0, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)
            del data_score_0, data_score_1

            c_indx = c_indx_0[score_index_1]

            # save the gradients for current class c:
            # pickled_data = {
            #     'data-name': "tiny-imagenet-200",
            #     'training_dynamics': gradients_all[c_indx],
            # }
            # filepath = td_path_0 + str(c) + ".pregrad"
            # with open(filepath, 'wb') as handle:
            #     pickle.dump(pickled_data, handle) 
            # continue

            filepath = td_path_0 + str(c) + ".feature"
            with open(filepath, 'rb') as f:
                pickled_data = pickle.load(f)
                cur_gradients = pickled_data['training_dynamics']

            # cur_gradients = gradients_all[c_indx]
            # cur_gradients = get_gradient(total_num, c_indx)
            cur_val_gradients = torch.mean(torch.tensor(cur_gradients), dim=0)

            cur_weights = orthogonal_matching_pursuit(torch.tensor(cur_gradients.T).cuda(), cur_val_gradients.cuda(),
                                                      budget=round(args.coreset_ratio * len(c_indx_0)))
            
            selection_result = np.append(selection_result, c_indx[np.nonzero(cur_weights)[0]])
        coreset_index = selection_result

    if args.coreset_mode == 'new':
        train_indx = np.arange(total_num)
        train_y = np.array(trainset.targets)
        y_onehot = np.eye(10)[train_y]
        
        selection_result = np.array([], dtype=np.int64)
        np.random.shuffle(train_indx)
        chunk_indx = train_indx
        
        chunk_list = np.split(chunk_indx, 2)
        el2n_all = data_score[coreset_key].numpy().astype(float)
        gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key)
        for c in chunk_list:
            c_indx_0 = c
            data_score_0 = data_score.copy()
            data_score_0[coreset_key] = data_score_0[coreset_key][c_indx_0]
            c_indx = c_indx_0
            features = features_all[c_indx]
            ys = train_y[c_indx]
            c_sel_len = len(c_indx)
            submod_function = FacilityLocation(index=c_indx, ##### OUR METHOD
                                            #    importance = difficulty_importance,
                                               importance = np.ones((c_sel_len,1)),
                                            #    similarity_kernel=lambda a,b: cossim_np(features[a], features[b])*difficulty_importance[b])
                                               similarity_kernel=lambda a,b: cossim_np(features[a], features[b])*(1 - 0.5*(np.not_equal(ys[a],ys[b]))))

            submod_optimizer = LazyGreedy(index=c_indx, budget=round(args.coreset_ratio * len(c_indx_0)), already_selected=[], importance = np.ones(c_sel_len))
            c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                            update_state=submod_function.update_state)
            selection_result = np.append(selection_result, c_selection_result)
            del c_selection_result
        coreset_index = selection_result

    if args.coreset_mode == 'boss':

        train_indx = np.arange(total_num)
        selection_result = np.array([], dtype=np.int64)

        mis_ratio = args.mis_ratio
        if args.dataset == 'cifar10':
            if args.coreset_ratio == 0.1: 
                mis_ratio = 0.3
            elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.5 or args.coreset_ratio == 0.7:
                mis_ratio = 0
        elif args.dataset == 'cifar100':
            if args.coreset_ratio == 0.1:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.2:
                mis_ratio = 0.4
            elif args.coreset_ratio == 0.3 or args.coreset_ratio == 0.5:
                mis_ratio = 0.2
            elif args.coreset_ratio == 0.7:
                mis_ratio = 0.1
        elif args.dataset == 'svhn':
            if args.coreset_ratio == 0.05:
                mis_ratio = 0.2
            elif args.coreset_ratio == 0.1:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.3:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.7 or args.coreset_ratio == 0.5:
                mis_ratio = 0
            elif args.coreset_ratio == 0.03 or args.coreset_ratio == 0.04:
                mis_ratio = 0.3
            elif args.coreset_ratio == 0.02:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.01:
                mis_ratio = 0.8
            elif args.coreset_ratio == 0.08:
                mis_ratio = 0.2
            elif args.coreset_ratio in [0.12, 0.16, 0.2]:
                mis_ratio = 0.1
        # elif args.dataset == 'tiny-imagenet-200':
        #     if args.coreset_ratio == 0.1:
        #         mis_ratio = 0.3
        #     elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
        #         mis_ratio = 0.2
        #     elif args.coreset_ratio == 0.5:
        #         mis_ratio = 0.1
        elif args.dataset == 'tiny-imagenet-200':
            if args.coreset_ratio == 0.1:
                mis_ratio = 0.7
            elif args.coreset_ratio == 0.2:
                mis_ratio = 0.6
            elif args.coreset_ratio == 0.3:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.5:
                mis_ratio = 0.4
        print(f"**** {args.coreset_ratio} -- {mis_ratio} ****")
        # mis_num = int(mis_ratio * total_num)
        # mis_num = 0
        el2n_all = data_score[coreset_key].numpy().astype(float)
        gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key, trainset, args.el2n_epoch)
        # print("Feature Dim: ", features_all.shape)
        # gradients_all = get_gradient(total_num)
        # print("After gradient: ")
        # process = psutil.Process()
        # print(process.memory_info().rss/(1024*1024*1024), "GB") 
        # hessians_reduced_all = get_hessian(total_num)
        # print("After Hessian: ")
        # process = psutil.Process()
        # print(process.memory_info().rss/(1024*1024*1024), "GB") 
        # training_dynamics = reduce_train_dynamics(td_path_0)
        # exit()
        # hessians_reduced_all, pick_idx_var, var_statistics = hessian_pick_var(hessians_all, 100)
        # del hessians_all
        # process = psutil.Process()
        # print(process.memory_info().rss/(1024*1024*1024))
        # exit()
        # data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)
        # 
        # after_mis = len(score_index)
        # train_indx = np.arange(after_mis)
        for c in range(num_classes):
        # for c in range(2):
            print(c, end=" ")
            if args.dataset == "svhn":
                c_indx_0 = train_indx[np.array(trainset.labels) == c]
            else:
                c_indx_0 = train_indx[np.array(trainset.targets) == c]

            # is_class = np.array(trainset.targets) == c
            # is_sel = np.isin(np.array(trainset.targets), score_index)
            data_score_0 = data_score.copy()
            data_score_0[coreset_key] = data_score_0[coreset_key][c_indx_0]
            if coreset_key != 'accumulated_margin':
                data_score_0['accumulated_margin'] = data_score_0['accumulated_margin'][c_indx_0]
            

            mis_num = int(mis_ratio * len(c_indx_0))
            # mis_num = 0
            data_score_1, score_index_1 = CoresetSelection.mislabel_mask(data_score_0, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)
            del data_score_0, data_score_1

            # c_indx = train_indx[is_class & is_sel]
            c_indx = c_indx_0[score_index_1]
            # print(c)
            # 
            # gradients_all, hessians_reduced_all, el2n_all, features_all = get_tr_dyn(total_num, td_path_0, data_score, coreset_key, c_indx)
            # gradients = gradients_all[c_indx]
            # hessians_reduced = hessians_reduced_all[c_indx]
            features = features_all[c_indx]
            # features = features_all
            # inverse_hessians = np.reciprocal(hessians)
            # precond_gradients = gradients * inverse_hessians

            # hessians_reduced, pick_idx_var, var_statistics = hessian_pick_var(hessians, 100)
            # del hessians

            # hessians_reduced = get_hessian(total_num, c_indx)
            # gradients = get_gradient(total_num, c_indx)

            #########################
            c_sel_len = len(c_indx)
            if True:
                el2n = el2n_all[c_indx]
                el2n_min = el2n.min()
                el2n_max = el2n.max()
                # unscaled_el2n = el2n_all[c_indx]
                el2n = (el2n - el2n_min)/(el2n_max-el2n_min)

                # P, R = np.histogram(el2n, bins=40, density=True)
                # mapper = lambda t: get_his_den(P, R, t)
                # vfunc = np.vectorize(mapper)
                # density_importance = vfunc(el2n)

                # _ = plt.hist(el2n, bins=40)
                # plt.savefig('forget_hist3.pdf')
                # np.histogram(Y, density=True)

                # x = el2n
                pi = np.pi
                SS = args.c_param
                alpha = args.a_param
                # difficulty_importance = ((np.sin(pi*el2n - (3/2)*pi - SS*pi) + 1)/2)**alpha

                difficulty_importance = scipy.stats.beta.pdf(el2n, SS, alpha)

                # fun_1 = np.sin(5*pi*x)
                # fun_2 = np.sin(5*pi*x + pi/2)
                # fun_3 = np.sin(5*pi*x - pi/2)
                # fun_4 = np.sin(5*pi*x + 3*pi)
                # difficulty_importance = np.maximum.reduce([fun_1,fun_2,fun_3,fun_4])

                # difficulty_importance = np.ones(x.shape)

                # diff = difficulty_importance.max() - difficulty_importance.min()
                # if diff > 0.001:    
                #     difficulty_importance = (difficulty_importance - difficulty_importance.min())/(diff)

            #######################################################
                # density_importance[density_importance < 0.001] = 1
            
            # difficulty_importance = difficulty_importance / density_importance


            # plt.scatter(el2n, difficulty_importance, c ="blue")
            # plt.savefig('forget_hist_density6.pdf')

            # l2_norm_np(gradients[0].reshape((1, 51300)), gradients[1].reshape((1, 51300)))
            # l1_norm_np(hessians_reduced[0].reshape((1, 100)),hessians_reduced[1].reshape((1, 100)))

            # gg = LA.norm(gradients_all, axis=1)
            # hh = LA.norm(hessians_reduced, axis=1)

            # submod_function = FacilityLocation(index=c_indx,
            #                                    importance = np.ones((x.shape[0],1)),
            #                                    similarity_kernel=lambda a,b: 5 - l2_norm_np(
            #                                         features[a], features[b]) - 1 * (
            #                                         unscaled_el2n[a])
            #                                    )
            
            # submod_function = FacilityLocation(index=c_indx,
            #                                    importance = np.ones((x.shape[0],1)),
            #                                    similarity_kernel=lambda a,b: 10 - l2_norm_np(
            #                                         gradients[a], gradients[b]) - 0.01 * l1_norm_np(
            #                                         hessians_reduced[a],hessians_reduced[b]) - 1 * (
            #                                         unscaled_el2n[a] - unscaled_el2n[b]) - 1 * (
            #                                         unscaled_el2n[a])
            #                                    )

            # submod_function = FacilityLocation(index=c_indx,  ### LCMAT
            #                                 #    importance = difficulty_importance,
            #                                    importance = np.ones((c_sel_len,1)),
            #                                    similarity_kernel=lambda a,b: 10 - l2_norm_np(
            #                                         gradients[a], gradients[b]) - 0.01 * l1_norm_np(
            #                                         hessians_reduced[a],hessians_reduced[b])
            #                                    )
            # del gradients
            # del hessians_reduced
            
            # submod_function = FacilityLocation(index=c_indx, 
            #                                    importance = difficulty_importance,
            #                                    similarity_kernel=lambda a,b: 10 - l2_norm_np(gradients[a], gradients[b]))
            
            # submod_function = FacilityLocation(index=c_indx, #### CRAIG
            #                                    importance = difficulty_importance,
            #                                    similarity_kernel=lambda a,b: cossim_np(gradients[a], gradients[b]))

            # submod_function = FacilityLocation(index=c_indx, ##### FEATURE MATCH
            #                                 #    importance = difficulty_importance,
            #                                    importance = np.ones((c_sel_len,1)),
            #                                    similarity_kernel=lambda a,b: cossim_np(features[a], features[b]))

            submod_function = FacilityLocation(index=c_indx, ##### OUR METHOD
                                            #    importance = difficulty_importance,
                                               importance = np.ones((c_sel_len,1)),
                                               similarity_kernel=lambda a,b: cossim_np(features[a], features[b])*difficulty_importance[b])
                                            # similarity_kernel=lambda a,b: cossim_np(features[a], features[b]))

            submod_optimizer = LazyGreedy(index=c_indx, budget=round(args.coreset_ratio * len(c_indx_0)), already_selected=[], importance = np.ones(c_sel_len))
            # submod_optimizer = LazyGreedy(index=c_indx, budget=round(args.coreset_ratio * len(c_indx_0)), already_selected=[], importance = difficulty_importance,)
            # del difficulty_importance

            c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                            update_state=submod_function.update_state)
            selection_result = np.append(selection_result, c_selection_result)
            del c_selection_result
        
        # del c_selection_result, gradients, hessians, hessians_reduced, gradients_all, hessians_all, training_dynamics, pickled_data
        # del gradients
        del features
        # del hessians_reduced
        # del gradients_all
        # del hessians_all

        coreset_index = selection_result
        print("subset selection complete....")

    if args.coreset_mode == 'submod':

        trainset_for_label = IndexDataset(trainset)
        # labels = torch.zeros(total_num, dtype=torch.int)
        labels = trainset_for_label.dataset.targets

        B = int(args.coreset_ratio * len(trainset))
        td_task_path = os.path.join(args.base_dir, "all-data")
        
        # td_path_0 = os.path.join(td_task_path, "td-all-data.pickle")
        with open(td_path_0, 'rb') as f:
            pickled_data = pickle.load(f)
            training_dynamics = pickled_data['training_dynamics']
        
        features = np.zeros((total_num, 512))
        gradients_all = np.zeros((total_num, 51300))
        hessians_all = np.zeros((total_num, 51300))
        # features = training_dynamics[9]['feature'].cpu().data.numpy()

        for x in training_dynamics:
            if x['epoch'] == 10:
                features[x['idx']] = x['feature']
                # gradients_all[x['idx']] = x['gradient']
                # hessians_all[x['idx']] = x['hessian']
                # features[x['idx']] = x['feature'].cpu().data.numpy()

        # training_dynamics_ep10 = [x for x in training_dynamics if x['epoch'] == 9]

        # inverse_hessians = np.reciprocal(hessians_all)
        # precond_gradients = gradients_all * inverse_hessians

        mis_ratio = args.mis_ratio
        if args.dataset == 'cifar10':
            if args.coreset_ratio == 0.1: 
                mis_ratio = 0.3
            elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.5 or args.coreset_ratio == 0.7:
                mis_ratio = 0
        elif args.dataset == 'cifar100':
            if args.coreset_ratio == 0.1:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.2:
                mis_ratio = 0.4
            elif args.coreset_ratio == 0.3 or args.coreset_ratio == 0.5:
                mis_ratio = 0.2
            elif args.coreset_ratio == 0.7:
                mis_ratio = 0.1
        print(f"**** {args.coreset_ratio} -- {mis_ratio} ****")
        mis_num = int(mis_ratio * total_num)
        # mis_num = 0
        data_score, score_index, features, labels = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key, features=features, labels=labels)
        coreset_index = CoresetSelection.representative_difficulty(data_score, coreset_key, B, features, labels, args.c_param, args.a_param)
        coreset_index = score_index[coreset_index]
    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset), num=args.coreset_ratio * len(trainset))

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_ratio = args.mis_ratio
        if args.dataset == 'cifar10':
            if args.coreset_ratio == 0.1: 
                mis_ratio = 0.3
            elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.5 or args.coreset_ratio == 0.7:
                mis_ratio = 0
        elif args.dataset == 'cifar100':
            if args.coreset_ratio == 0.1:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.2:
                mis_ratio = 0.4
            elif args.coreset_ratio == 0.3 or args.coreset_ratio == 0.5:
                mis_ratio = 0.2
            elif args.coreset_ratio == 0.7:
                mis_ratio = 0.1
        # elif args.dataset == 'svhn':
        #     if args.coreset_ratio == 0.05:
        #         mis_ratio = 0.2
        #     elif args.coreset_ratio == 0.1:
        #         mis_ratio = 0.1
        #     elif args.coreset_ratio == 0.3:
        #         mis_ratio = 0.1
        #     elif args.coreset_ratio == 0.7 or args.coreset_ratio == 0.5:
        #         mis_ratio = 0
        # elif args.dataset == 'tiny-imagenet-200':
        #     if args.coreset_ratio == 0.1:
        #         mis_ratio = 0.3
        #     elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
        #         mis_ratio = 0.2
        #     elif args.coreset_ratio == 0.5:
        #         mis_ratio = 0.1
        print(f"**** {args.coreset_ratio} -- {mis_ratio} ****")
        mis_num = int(mis_ratio * total_num)
        # mis_num = 0
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)

        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'stratified_submod':

        trainset_for_label = IndexDataset(trainset)
        # labels = torch.zeros(total_num, dtype=torch.int)
        labels = trainset_for_label.dataset.targets

        B = args.coreset_ratio * len(trainset)
        td_task_path = os.path.join(args.base_dir, "all-data")
        
        # td_path_0 = os.path.join(td_task_path, "td-all-data.pickle")
        with open(td_path_0, 'rb') as f:
            pickled_data = pickle.load(f)
            training_dynamics = pickled_data['training_dynamics']
        
        features = np.zeros((total_num, 512))
        gradients_all = np.zeros((total_num, 51300))
        hessians_all = np.zeros((total_num, 51300))
        # features = training_dynamics[9]['feature'].cpu().data.numpy()

        for x in training_dynamics:
            if x['epoch'] == 10:
                features[x['idx']] = x['feature']
                gradients_all[x['idx']] = x['gradient']
                # hessians_all[x['idx']] = x['hessian']
                # features[x['idx']] = x['feature'].cpu().data.numpy()

        mis_ratio = args.mis_ratio
        if args.dataset == 'cifar10':
            if args.coreset_ratio == 0.1: 
                mis_ratio = 0.3
            elif args.coreset_ratio == 0.2 or args.coreset_ratio == 0.3:
                mis_ratio = 0.1
            elif args.coreset_ratio == 0.5 or args.coreset_ratio == 0.7:
                mis_ratio = 0
        elif args.dataset == 'cifar100':
            if args.coreset_ratio == 0.1:
                mis_ratio = 0.5
            elif args.coreset_ratio == 0.2:
                mis_ratio = 0.4
            elif args.coreset_ratio == 0.3 or args.coreset_ratio == 0.5:
                mis_ratio = 0.2
            elif args.coreset_ratio == 0.7:
                mis_ratio = 0.1
        print(f"**** {args.coreset_ratio} -- {mis_ratio} ****")
        mis_num = int(mis_ratio * total_num)
        grad_norm = np.linalg.norm(gradients_all, axis=1)
        residual = data_score['el2n_total']
        

        data_score, score_index, features, labels = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key, features=features, labels=labels)

        coreset_num = int(args.coreset_ratio * total_num)
        # coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index, _ = CoresetSelection.stratified_sampling_with_rep(data_score, coreset_key, coreset_num, features, labels, args.c_param, args.a_param)
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'moderate':
        features, targets = get_feature_target(total_num, td_path_0, data_score, coreset_key)
        distance = get_distance(features, targets)
        coreset_index = get_prune_idx(1-args.coreset_ratio, distance)


    trainset = torch.utils.data.Subset(trainset, coreset_index)
    print(len(trainset))
######################### Coreset Selection end #########################

trainset = IndexDataset(trainset)
print(len(trainset), len(np.unique(coreset_index)))

if args.dataset == 'cifar10':
    testset = CIFARDataset.get_cifar10_test(data_dir)
elif args.dataset == 'cifar100':
    testset = CIFARDataset.get_cifar100_test(data_dir)
elif args.dataset == 'svhn':
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'cinic10':
    testset = CINIC10Dataset.get_cinic10_test(data_dir)
elif args.dataset == 'tiny-imagenet-200':
    testset = TinyDataset.get_tiny_test(data_dir)

print(len(testset))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=4)

iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

# if args.dataset in ['cifar10', 'svhn', 'cinic10']:
#     num_classes=10
# else:
#     num_classes=100

# if args.network == 'resnet18':
#     print('resnet18')
#     model = resnet('resnet18', num_classes=num_classes, device=device)
# if args.network == 'resnet50':
#     print('resnet50')
#     model = resnet('resnet50', num_classes=num_classes, device=device)

# model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)

epoch_per_testing = args.iterations_per_testing // iterations_per_epoch

if args.initial_train:
    epoch_per_testing = 20
else:
    epoch_per_testing = 20


print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

trainer = Trainer()
TD_logger = TrainingDynamicsLogger()

best_acc = 0
best_epoch = -1

current_epoch = 0
while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device, TD_logger=TD_logger, log_interval=60, printlog=True, initial_train=args.initial_train, dataset=args.dataset)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True, dataset=args.dataset)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            if args.initial_train:
                torch.save(state, best_ckpt_path)

    current_epoch += 1
    # scheduler.step()

# last ckpt testing

test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True, dataset=args.dataset)
if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            if args.initial_train:
                torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)
######################### Save #########################

if args.initial_train:
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': current_epoch - 1
    }
    torch.save(state, last_ckpt_path)
    TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')

# python train.py --dataset svhn --gpuid 0 --epochs 100 --task-name lcmat_test --base-dir ./model/svhn --coreset --coreset-mode lcmat --data-score-path ./model/svhn/all-data/data-score-all-data.pickle --coreset-key el2n_total --coreset-ratio 0.2 --mis-ratio 0 --c_param 0 --a_param 1
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 100 --network resnet34 --task-name lcmat_test --base-dir ./model/tiny --coreset --coreset-mode lcmat --data-score-path ./model/tiny/all-data/data-score-all-data.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 2 --a_param 2


# python train.py --dataset cifar10 --gpuid 0 --epochs 200 --task-name lcmat_test --base-dir ./model/cifar10 --coreset --coreset-mode lcmat --data-score-path ./model/cifar10/all-data/data-score-all-data.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0.1 --a_param 2

# initial training: 
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 100 --lr 0.1 --network resnet34 --batch-size 256 --task-name all-data-tiny --base-dir ./model/tiny --initial_train 1

# importance score:
# python generate_importance_score.py --gpuid 0 --base-dir ./model/tiny --task-name all-data --data-dir ../data --dataset tiny-imagenet-200


# 96.00 -> svhn full set acc
# 62.79 -> TinyImageNet full set


# TI-CCS-mr:0.1-ss:0.1-27.33, 27.94,  26.01

# 85.91, 85.49, 83.82, 81.52, 83.52
# 87.45, 87.85, 89.48, 85.86, 86.85
# 90.95, 90.04, 89.28, 90.32, 90.22
# 91.46, 90.75, 91.43, 91.21, 91.93

# GradMatch
# python train.py --dataset cifar10 --gpuid 0 --epochs 200 --task-name lcmat_test --base-dir ./model/cifar10 --coreset --coreset-mode adacore --data-score-path ./model/cifar10/all-data/data-score-all-data.pickle --coreset-key forgetting --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0 --a_param 1
# python train.py --dataset cifar100 --gpuid 0 --epochs 200 --task-name lcmat_test --base-dir ./model/cifar100 --coreset --coreset-mode adacore --data-score-path ./model/cifar100/all-data/data-score-all-data.pickle --coreset-key forgetting --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0 --a_param 1

# EfficientNet-B0
# Initial Train:
# python train.py --dataset cifar100 --gpuid 0 --epochs 200 --lr 0.1 --network eff --batch-size 256 --task-name eff-cifar100 --base-dir ./model/eff-cifar100 --initial_train 1
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 100 --lr 0.1 --network eff --batch-size 256 --task-name eff-tiny --base-dir ./model/eff-tiny --initial_train 1
# Generate Scores:
# python generate_importance_score.py --gpuid 0 --base-dir ./model/eff-cifar100 --task-name eff-cifar100 --data-dir ../data --dataset cifar100
# python generate_importance_score.py --gpuid 0 --base-dir ./model/eff-tiny --task-name eff-tiny --data-dir ../data --dataset tiny-imagenet-200
# Subset Training:
# python train.py --dataset cifar100 --gpuid 0 --epochs 200 --network eff --task-name lcmat_test --base-dir ./model/eff-cifar100 --coreset --coreset-mode lcmat --data-score-path ./model/eff-cifar100/eff-cifar100/data-score-eff-cifar100.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 1.6 --a_param 1.6
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 200 --network eff --task-name lcmat_test --base-dir ./model/eff-tiny --coreset --coreset-mode lcmat --data-score-path ./model/eff-tiny/eff-tiny/data-score-eff-tiny.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0 --a_param 1

# ViT
# Initial Train:
# python train.py --dataset cifar100 --gpuid 0 --epochs 11 --lr 0.1 --network vit --batch-size 256 --task-name vit-cifar100 --base-dir ./model/vit-cifar100 --initial_train 1
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 100 --lr 0.1 --network eff --batch-size 256 --task-name eff-tiny --base-dir ./model/eff-tiny --initial_train 1
# Generate Scores:
# python generate_importance_score.py --gpuid 0 --base-dir ./model/eff-cifar100 --task-name eff-cifar100 --data-dir ../data --dataset cifar100
# python generate_importance_score.py --gpuid 0 --base-dir ./model/eff-tiny --task-name eff-tiny --data-dir ../data --dataset tiny-imagenet-200
# Subset Training:
# python train.py --dataset cifar100 --gpuid 0 --epochs 200 --network eff --task-name lcmat_test --base-dir ./model/eff-cifar100 --coreset --coreset-mode lcmat --data-score-path ./model/eff-cifar100/eff-cifar100/data-score-eff-cifar100.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 2 --a_param 2
# python train.py --dataset tiny-imagenet-200 --gpuid 0 --epochs 200 --network eff --task-name lcmat_test --base-dir ./model/eff-tiny --coreset --coreset-mode lcmat --data-score-path ./model/eff-tiny/eff-tiny/data-score-eff-tiny.pickle --coreset-key el2n_total --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0 --a_param 1

# python vit_train.py --dataset cifar100 --gpuid 0 --epochs 200 --network vit --task-name $job_name --base-dir ./model/vit2-cifar100 --coreset --coreset-mode stratified --data-score-path ./model/vit2-cifar100/vit2-cifar100/data-score-vit2-cifar100.pickle --coreset-key el2n_total --coreset-ratio $subset --mis-ratio $b_val --c_param $c_val --a_param $a_val --el2n_epoch $b_val
# python train.py --dataset cifar10 --gpuid 0 --epochs 200 --task-name lcmat_test --base-dir ./model/cifar10 --coreset --coreset-mode lcmat --data-score-path ./model/cifar10/all-data/data-score-all-data.pickle --coreset-key forgetting --coreset-ratio 0.1 --mis-ratio 0.1 --c_param 0 --a_param 1