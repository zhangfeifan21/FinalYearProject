from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import WideResNet
from sparselearning.utils import get_cifar10_dataloaders, get_mnist_dataloaders, plot_class_feature_histograms

# modified part
from extensions import gradient_redistribution, gradient_growth
from dataloader.cifar100 import get_cifar100_dataloaders
from models import ResNet18

cudnn.benchmark = True  # 为每个卷积层搜索最适合的卷积实现算法，实现网络加速。网络结构固定，输入形状不变时适用。
cudnn.deterministic = True  # 保证结果可复现，固定随机源

if not os.path.exists('./models'):
    os.mkdir('./models')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
logger = None

models = {'wrn-28-2': (WideResNet, [28, 2, 100, 0.3]),
          'wrn-22-8': (WideResNet, [22, 8, 100, 0.3]),
          'wrn-16-8': (WideResNet, [16, 8, 100, 0.3]),
          'wrn-16-10': (WideResNet, [16, 10, 100, 0.3]),
          'resnet-18': (ResNet18, [2, 100])}


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density,
                                               hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if lr_scheduler is not None:
            lr_scheduler.step()
        data, target = data.to(device), target.to(device)
        if args.fp16:
            data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    # training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 test')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S',
                        help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='The optimizer to use. Default: adam. Options: sgd, adam.')
    parser.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true',
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true',
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine',
                        help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

        if args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                               max_threads=args.max_threads)
        else:
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            # print(cls, cls_args)
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)

            print_and_log('=' * 60)
            print_and_log('Prune mode: {0}'.format(args.prune))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('=' * 60)
        # custom prune/growth/redistribution here
        if args.redistribution == 'gradient':
            args.redistribution = gradient_redistribution
        if args.growth == 'gradient':
            args.growth = gradient_growth

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,
                                  nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.1)

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                              .format(args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=None,
                                       dynamic_loss_scale=True,
                                       dynamic_loss_args={'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if not args.dense:
            if args.decay_schedule == 'cosine':
                decay = CosineDecay(args.prune_rate, len(train_loader) * args.epochs)
            elif args.decay_schedule == 'linear':
                decay = LinearDecay(args.prune_rate, len(train_loader) * args.epochs)
            mask = Masking(optimizer, decay, prune_rate=args.prune_rate, prune_mode=args.prune, growth_mode=args.growth,
                           redistribution_mode=args.redistribution,
                           verbose=args.verbose, fp16=args.fp16)
            mask.add_module(model, density=args.density)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask)

            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best=False, filename=args.save_model)

            if not args.dense and epoch < args.epochs:
                mask.at_end_of_epoch()

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))

        evaluate(args, model, device, test_loader, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))


if __name__ == "__main__":
    main()
