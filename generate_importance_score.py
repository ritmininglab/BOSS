import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle

import pdb

from core.model_generator import wideresnet, preact_resnet, resnet
from core.model_generator.efficientnet import EfficientNet
from core.model_generator.vit import vit
from core.model_generator.vit_small import ViT
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, TinyDataset
from core.utils import print_training_info, StdRedirect

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny', 'svhn', 'cinic10', 'tiny-imagenet-200'])

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str,
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

args = parser.parse_args()

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
# ckpt_path = os.path.join(task_dir, f'ckpt-best_copy.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')

######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
if dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif dataset == 'cifar100':
    num_classes=100
elif dataset == 'tiny-imagenet-200':
    num_classes=200


######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device, dataset):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    # for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
    for x in enumerate(dataloader):
        if dataset == "tiny-imagenet-200":
            batch_idx, (_,(idx, inputs, targets)) = x
        else:
            batch_idx, (idx, (inputs, targets)) = x
        inputs, targets = inputs.to(device), targets.to(device)

        logits, _ = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance, ds):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        # pdb.set_trace()
        if ds == "tiny-imagenet-200":
            _, (_,_,y) = dataset[i]
        else:
            _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)

"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, max_epoch=10, drop=False, ds=None):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        if ds == "tiny-imagenet-200":
            _, (_,_,y) = dataset[i]
        else:
            _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        if drop:
            output = torch.exp(td_log['output_drop'].type(torch.float))
        else:
            output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        if item['epoch'] > max_epoch:
            return
        record_training_dynamics(item)
#########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),     
        ])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}')
if dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
elif dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)
elif dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir, transform = transform_identical)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir, transform = transform_identical)
elif args.dataset == 'tiny-imagenet-200':
    # td_path_0 = "model/tiny/all-data/td-all-data.pickle"
    trainset = TinyDataset.get_tiny_train(data_dir)

trainset = IndexDataset(trainset)
print(len(trainset))

data_importance = {}

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

# model = resnet('resnet34', num_classes=num_classes, device=device)

# cfg = {
#         'num_blocks': [1, 2, 2, 3, 3, 4, 1],
#         'expansion': [1, 6, 6, 6, 6, 6, 6],
#         'out_channels': [16, 24, 40, 80, 112, 192, 320],
#         'kernel_size': [3, 3, 5, 3, 5, 5, 3],
#         'stride': [1, 2, 2, 2, 1, 2, 1],
#         'dropout_rate': 0.2,
#         'drop_connect_rate': 0.2,
#     }
# model = EfficientNet(cfg, num_classes=num_classes)

# model = ViT(
#             3, 
#             num_classes, 
#             img_size=32, 
#             patch=8, 
#             dropout=0.5, 
#             mlp_hidden=384,
#             num_layers=7,
#             hidden=384,
#             head=12,
#             is_cls_token=True
#         )
model = vit(num_classes=num_classes)
model = model.to(device)

print(f'Ckpt path: {ckpt_path}.')
checkpoint = torch.load(ckpt_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)

training_dynamics = pickled_data['training_dynamics']

# post_training_metrics(model, trainloader, data_importance, device, dataset)
training_dynamics_metrics(training_dynamics, trainset, data_importance, dataset)
EL2N(training_dynamics, trainset, data_importance, max_epoch=5, ds=dataset)

# get 90% dropout el2n score:
data_importance['el2n_total'] = data_importance['el2n']
# model.get_score = True
# training_dynamics_metrics(training_dynamics, trainset, data_importance)
EL2N(training_dynamics, trainset, data_importance, max_epoch=5, drop=True, ds=dataset)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)