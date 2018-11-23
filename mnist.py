# coding:utf-8

from chainer.datasets import mnist
from chainer.datasets import split_dataset_random
from chainer import iterators

import chainer.links as L
import chainer.functions as F

import random
import numpy as np
import chainer

from chainer import optimizers

from chainer.dataset import concat_examples
from chainer.cuda import to_cpu

from chainer import serializers

train_val, test = mnist.get_mnist(withlabel=True, ndim=1)
train, valid = split_dataset_random(train_val, 50000, seed=0)

batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(
    valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

reset_seed(0)

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

gpu_id = -1

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)

optimizer = optimizers.SGD(lr=0.01).setup(net)

max_epoch = 10

while train_iter.epoch < max_epoch:

    train_batch = train_iter.next()
    x, t = concat_examples(train_batch, gpu_id)

    y = net(x)

    loss = F.softmax_cross_entropy(y, t)

    net.cleargrads()
    loss.backward()

    optimizer.update()

    if train_iter.is_new_epoch:
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')

        valid_losses = []
        valid_accuracies = []

        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = concat_examples(valid_batch, gpu_id)

            with chainer.using_config('train', False), \
                    chainer.using_config('enable_backprop', False):
                y_valid = net(x_valid)

            loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))

            accuracy = F.accuracy(y_valid, t_valid)
            accuracy.to_cpu()
            valid_accuracies.append(accuracy.array)

            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(valid_losses), np.mean(valid_accuracies)))

test_accuracies = []
while True:
    test_batch = test_iter.next()
    x_test, t_test = concat_examples(test_batch, gpu_id)

    with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
        y_test = net(x_test)

    accuracy = F.accuracy(y_test, t_test)
    accuracy.to_cpu()
    test_accuracies.append(accuracy.array)

    if test_iter.is_new_epoch:
        test_iter.reset()
        break

serializers.save_npz('my_mnist.model', net)

print('test_accuracy:{:.04f}'.format(np.mean(test_accuracies)))

