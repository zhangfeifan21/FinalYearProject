import math
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, save_features=False, bench=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.in_planes = in_planes

    def forward(self, x):
        residual = x

        if self.bench:
            out = self.bench.forward(self.conv1, x, str(self.in_planes) + '.conv1')
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item() / out.numel())

        if self.bench:
            out = self.bench.forward(self.conv2, out, str(self.in_planes) + '.conv2')
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item() / out.numel())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, save_features=False, bench=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.in_planes = in_planes

    def forward(self, x):
        residual = x

        if self.bench:
            out = self.bench.forward(self.conv1, x, str(self.in_planes) + '.conv1')
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item() / out.numel())

        if self.bench:
            out = self.bench.forward(self.conv2, out, str(self.in_planes) + '.conv2')
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item() / out.numel())

        if self.bench:
            out = self.bench.forward(self.conv3, out, str(self.in_planes) + '.conv3')
        else:
            out = self.conv3(out)

        out = self.bn3(out)
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item() / out.numel())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, depth, num_classes=100, save_features=False, bench_model=False):
        self.in_planes = 64
        super(ResNet18, self).__init__()
        block = BasicBlock
        layers = [depth, depth, depth, depth]
        self.save_features = save_features

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.bench = None if not bench_model else SparseSpeedupBench()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.feats = []
        self.densities = []

        # TODO: difference between initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_planes, blocks, stride=1):
        # TODO: things about save_features
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample,
                            save_features=self.save_features, bench=self.bench))
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, out_planes,
                                save_features=self.save_features, bench=self.bench))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bench is not None:
            x = self.bench.forward(self.conv1, x, 'conv1')
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # TODO: things about save_features
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# unmodified
class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

        Basic usage:
        1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
        2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
            self.bench = SparseSpeedupBench()
            self.conv_layer1 = nn.Conv2(3, 96, 3)

            if self.bench is not None:
                outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
            else:
                outputs = self.conv_layer1(inputs)
        3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """
    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data != 0.0).sum().item() / x.numel()

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print('\n')
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_channel_sparse += t_channel_sparse
            total_time_sparse += t_sparse

            print('Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}'.format(layer_id, t_dense,
                                                                                                      t_channel_sparse,
                                                                                                      t_sparse))
        self.total_timings.append(total_time_dense)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)
        self.total_timings_sparse.append(total_time_sparse)

        print('Speedups for this segment:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense,
                                                                                              total_time_channel_sparse,
                                                                                              total_time_dense / total_time_channel_sparse))
        print(
            'Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense,total_time_sparse,
                                                                                    total_time_dense / total_time_sparse))
        print('\n')

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print('Speedups for entire training:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense,
                                                                                              total_channel_sparse,
                                                                                              total_dense / total_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_sparse,
                                                                                      total_dense / total_sparse))
        print('\n')

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)

        # TODO: meaning of channel_sparse and sparse
    def forward(self, layer, x, layer_id):
        if self.layer_0_idx is None:
            self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx:
            self.iter_idx += 1

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in / float(num_channels_in * batch_size)
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end) / 1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        channel_sparsity_weight = sparse_channels / float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.total_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(
            time_taken_s * (1.0 - channel_sparsity_weight) * (1.0 - channel_sparsity_input)
        )
        self.layer_timings_sparse[layer_id].append(time_taken_s * input_sparsity * weight_sparsity)

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x
