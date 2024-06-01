import torch
import torch.nn.functional as F
from datetime import datetime
import pdb
import numpy as np
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian


def calc_gradient_and_hess(self, index, criterion, batch_loader, n_train, model_optimizer, model, dataset):
    '''
    Calculate gradients matrix on current network for specified training dataset.
    '''
    criterion = extend(criterion)

    # batch_loader = torch.utils.data.DataLoader(
    #         dst_train if index is None else torch.utils.data.Subset(dst_train, index),
    #         batch_size=256,
    #         num_workers=4)
    # sample_num = n_train if index is None else len(index)
    # self.embedding_dim = self.model.get_last_layer().in_features

    # Initialize a matrix to save gradients. (on cpu)
    gradients = []
    hessians = []
    features = []

    # for batch_idx, (idx, (inputs, targets)) in enumerate(batch_loader):
    for x in enumerate(batch_loader):
        if dataset == "tiny-imagenet-200":
            batch_idx, (_,(idx, inputs, targets)) = x
        else:
            batch_idx, (idx, (inputs, targets)) = x
        model_optimizer.zero_grad()
        outputs, feature = model(inputs.cuda())
        loss = criterion(outputs,targets.cuda())
        batch_num = targets.shape[0]

        with backpack(BatchGrad(),BatchDiagHessian()):
            loss.backward()

        for name, param in model.named_parameters():
            if 'linear.weight' in name or 'classifier.weight' in name:
                weight_parameters_grads = param.grad_batch
                weight_parameters_hesses = param.diag_h_batch
                # print(weight_parameters_hesses.shape)
            elif 'linear.bias' in name or 'classifier.bias' in name:
                bias_parameters_grads = param.grad_batch
                bias_parameters_hesses = param.diag_h_batch
                # print(bias_parameters_hesses.shape)

        gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                    dim=1).cpu().numpy())
        hessians.append(torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
                                    dim=1).cpu().numpy())
        features.append(feature.detach().cpu().numpy())

    gradients = np.concatenate(gradients, axis=0)
    hessians = np.concatenate(hessians, axis=0)
    features = np.concatenate(features, axis=0)
    return gradients, hessians, features


class Trainer(object):
    """
    Helper class for training.
    """

    def __init__(self):
        pass

    """
    Dataset need to be an index dataset.
    Set remaining_iterations to -1 to ignore this argument.
    """
    def train(self, epoch, remaining_iterations, model, dataloader, optimizer, criterion, scheduler, device, TD_logger=None, log_interval=None, printlog=False, initial_train=0, dataset=None):
        # if initial_train:
        #     criterion = extend(criterion)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()
        if printlog:
            print('*' * 26)

        # if epoch == 10 and initial_train:
        #     gradients_all, hessians_all, features_all = calc_gradient_and_hess(self, None, criterion, dataloader, len(dataloader.dataset), optimizer, model, dataset)

        # pdb.set_trace()

        # for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        for x in enumerate(dataloader):
            if dataset == "tiny-imagenet-200":
                batch_idx, (_,(idx, inputs, targets)) = x
            else:
                batch_idx, (idx, (inputs, targets)) = x
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            gradients = []
            hessians = []
            features = []
            # if epoch == 10 and initial_train:
            #     gradients, hessians, features = gradients_all[idx], hessians_all[idx], features_all[idx]
            
            # pdb.set_trace()

            optimizer.zero_grad()
            outputs, f = model(inputs)
            loss = criterion(outputs, targets)
            if epoch == 10 and initial_train:
                features = f

            # if initial_train:
            #     model.get_score = True
            #     outputs_drop, _ = model(inputs)
            #     model.get_score = False
            # else:
            # outputs_drop = outputs

            # if initial_train:
            #     with backpack(BatchGrad(),BatchDiagHessian()):
            #         loss.backward()

            #     if epoch == 10:
            #         for name, param in model.named_parameters():
            #             if 'linear.weight' in name or 'classifier.weight' in name:
            #                 weight_parameters_grads = param.grad_batch
            #                 weight_parameters_hesses = param.diag_h_batch
            #                 # print(weight_parameters_hesses.shape)
            #             elif 'linear.bias' in name or 'classifier.bias' in name:
            #                 bias_parameters_grads = param.grad_batch
            #                 bias_parameters_hesses = param.diag_h_batch

            #         gradients = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
            #                                     dim=1).cpu().numpy()
                    
            #         hessians = torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
            #                                     dim=1).cpu().numpy()
            #     else:
            #         gradients = []
            #         hessians = []
            # else:
            #     loss.backward()
            #     gradients = []
            #     hessians = []
            #   
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.detach().cpu().max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets.detach().cpu()).sum().item()

            # features = features.detach().cpu().numpy()
            

            if TD_logger and initial_train:
                log_tuple = {
                'epoch': epoch,
                'iteration': batch_idx,
                'idx': idx.type(torch.long).clone(),
                'output': F.log_softmax(outputs.detach().cpu(), dim=1).detach().cpu().type(torch.half),
                'output_drop': F.log_softmax(outputs.detach().cpu(), dim=1).detach().cpu().type(torch.half),
                'feature': features,
                'gradient': gradients,
                'hessian': hessians
                }
                TD_logger.log_tuple(log_tuple)

            # pdb.set_trace()

            if printlog and log_interval and batch_idx % log_interval == 0:
                print(f"{batch_idx}/{len(dataloader)}")
                print(f'>> batch_idx [{batch_idx}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')


            # if epoch == 1:
            
            # calculate gradient and hessian:
            # n_train = len(dataloader.dataset)
            # train_indx = np.arrange(n_train)
            # pdb.set_trace()

            remaining_iterations -= 1
            if remaining_iterations == 0:
                if printlog: print("Exit early in epoch training.")
                break

        if printlog:
            print(f'>> Epoch [{epoch}]: Loss: {train_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'>> Epoch [{epoch}]: Training Accuracy: {correct/total * 100:.2f}')
            print(f'>> Epoch [{epoch}]: Time consumed: {(datetime.now() - start_time).total_seconds():.2f}')

    def test(self, model, dataloader, criterion, device, log_interval=None,  printlog=False, dataset=None):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = datetime.now()

        if printlog: print('======= Testing... =======')
        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for x in enumerate(dataloader):
                if dataset == "tiny-imagenet-200":
                    batch_idx, (idx, inputs, targets) = x
                else:
                    batch_idx, (inputs, targets) = x
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.detach().cpu().max(1)
                total += targets.shape[0]
                correct += predicted.eq(targets.detach().cpu()).sum().item()

                # if printlog and log_interval and batch_idx % log_interval == 0:
                #     print(batch_idx)

        if printlog:
            print(f'Loss: {test_loss:.2f}')
            # print(f'Correct/Total: {correct}/{total}')
            print(f'Test Accuracy: {correct/total * 100:.2f}')

        print(f'>> Test time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
        return test_loss, correct / total