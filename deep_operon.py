#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# CreateTime: 2016-09-21 16:51:48

import numpy as np
from Bio import SeqIO, Seq, SeqUtils
#from Bio.SeqUtils.CodonUsage import CodonAdaptationIndex
from Bio.SeqUtils import GC
from Bio.SeqUtils.CodonUsage import SynonymousCodons
import math
from math import log, sqrt
from collections import Counter
import pickle
import gc

# from sklearn import cross_validation, metrics  # Additional scklearn
# functions
from sklearn import metrics  # Additional scklearn functions
#from sklearn.metrics import f1_score, recall_score, precision_score
# from sklearn.grid_search import GridSearchCV  # Perforing grid search
from sklearn.svm import LinearSVC as SVC

#from tensorflow.keras_adabound import AdaBound

import os
from copy import deepcopy

if 1:
    # set backend
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from tensorflow.keras import backend as K
    # K.set_image_data_format('channels_first')


    try:
        import theano.ifelse
        from theano.ifelse import IfElse, ifelse
    except:
        pass

    try:
        from tensorflow import set_random_seed
    except:
        set_random_seed = np.random.seed


    import torch
    from torch import nn
    from torch.autograd import Variable
    import torch.nn.functional as F
    import torch.utils.data as Data


import chainer
import chainer.functions as cF
import chainer.links as cL
from chainer import training
from chainer import Variable as Var
from chainer.training import extensions
from chainer import serializers

#from chainer_sklearn.links import SklearnWrapperClassifier


try:
    import lightgbm as lgb
except:
    pass



from tensorflow.python.keras import backend as K_tf
from tensorflow.python.keras.optimizers import Optimizer as Optimizer_tf


class AdaBound(Optimizer_tf):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    Arguments:
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        final_lr: float >= 0. Final learning rate.
        gamma: float >= 0. Convergence speed of the bound functions.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsbound: boolean. Whether to use the AMSBound variant of this algorithm
            from the paper "Adaptive Gradient Methods with Dynamic Bound of Learning Rate".
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 final_lr=0.1,
                 gamma=0.001,
                 epsilon=1e-8,
                 decay=0.,
                 amsbound=False,
                 **kwargs):
        super(AdaBound, self).__init__(**kwargs)
        with K_tf.name_scope(self.__class__.__name__):
            self.iterations = K_tf.variable(0, dtype='int64', name='iterations')
            self.lr = K_tf.variable(lr, name='lr')
            self.beta_1 = K_tf.variable(beta_1, name='beta_1')
            self.beta_2 = K_tf.variable(beta_2, name='beta_2')
            self.final_lr = K_tf.variable(final_lr, name='final_lr')
            self.gamma = K_tf.variable(gamma, name='gamma')
            self.decay = K_tf.variable(decay, name='decay')
            self.amsbound = K_tf.variable(amsbound, name='amsbound')
        if epsilon is None:
            epsilon = K_tf.epsilon()
        self.epsilon = K_tf.variable(epsilon)
        self.initial_decay = decay
        self.base_lr = lr

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (
                    1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                          K_tf.dtype(self.decay))))

        t = math_ops.cast(self.iterations, K_tf.floatx()) + 1
        lr_t = lr * (
                K_tf.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                (1. - math_ops.pow(self.beta_1, t)))

        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1))
        upper_bound = final_lr * (1. + 1. / (self.gamma * t))

        ms = [K_tf.zeros(K_tf.int_shape(p), dtype=K_tf.dtype(p)) for p in params]
        vs = [K_tf.zeros(K_tf.int_shape(p), dtype=K_tf.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K_tf.zeros(K_tf.int_shape(p), dtype=K_tf.dtype(p)) for p in params]
        else:
            vhats = [K_tf.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsbound:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - m_t * K_tf.clip(lr_t / (K_tf.sqrt(vhat_t) + self.epsilon), lower_bound, upper_bound)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - m_t * K_tf.clip(lr_t / (K_tf.sqrt(v_t) + self.epsilon), lower_bound, upper_bound)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K_tf.get_value(self.lr)),
            'beta1': float(K_tf.get_value(self.beta_1)),
            'beta2': float(K_tf.get_value(self.beta_2)),
            'final_lr': float(K_tf.get_value(self.final_lr)),
            'gamma': float(K_tf.get_value(self.gamma)),
            'decay': float(K_tf.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsbound': self.amsbound
        }
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from pickle import load, dump
from time import time


# global parameters:
global_kmr = 5
global_len = 256
global_split_rate = 1./10
global_epoch = 256

#global_kmr = 5
#global_len = 256

###############################################################################
# multiple-gpu support
###############################################################################


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def f1_score_keras(y_true, y_pred):
#def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    #print type(precision)
    #if float(precision) > .87 and float(recall) > .87:
    if 1:
        return 2 * ((precision * recall) / (precision + recall))
    #else:
    #    return 0

fbeta_score = f1_score_keras
#fbeta_score = f1_score


##########################################################################
# overlap of two gene
##########################################################################
overlap = lambda s0, e0, s1, e1: min(e0, e1) - max(s0, s1) + 1


##########################################################################
# share kmer between 2 sequence
##########################################################################
def pearson(x, y):
    N, M = len(x), len(y)
    assert N == M
    x_m, y_m = sum(x) * 1. / N, sum(y) * 1. / M
    a, b, c = 0., 0., 0.
    for i in range(N):
        xi, yi = x[i] - x_m, y[i] - y_m
        a += xi * yi
        b += xi ** 2
        c += yi ** 2
    try:
        return a / sqrt(b * c)
    except:
        return 0


def sharekmer(s1, s2):
    # SynonymousCodons
    n1, n2 = list(map(len, [s1, s2]))
    k1 = [s1[elem: elem + 3] for elem in range(0, n1, 3)]
    k2 = [s1[elem: elem + 3] for elem in range(0, n2, 3)]
    fq1 = Counter(k1)
    fq2 = Counter(k2)
    flag = 0

    kmers = []
    for i in SynonymousCodons:
        j = SynonymousCodons[i]
        if len(j) < 2:
            continue
        c1 = [[fq1[elem], elem] for elem in j]
        best1 = max(c1, key=lambda x: x[0])
        c2 = [[fq2[elem], elem] for elem in j]
        best2 = max(c2, key=lambda x: x[0])

        #c1.sort(key = lambda x: x[0], reverse = True)
        #c2.sort(key = lambda x: x[0], reverse = True)

        if best1[1] == best2[1]:
            kmers.append(best1[1])

    # for val in SynonymousCodons.values():
    #	if len(val) > 5:
    #		kmers.extend(val)

    # print 'the kmer', kmers, len(kmers)
    vec1 = [fq1[elem] for elem in kmers]
    vec2 = [fq2[elem] for elem in kmers]

    return pearson(vec1, vec2)


##########################################################################
# the motif found
##########################################################################
box_up10 = ['TATAAT', [77, 76, 60, 61, 56, 82]]
box_up35 = ['TTGACA', [69, 79, 61, 56, 54, 54]]
# find the best region that may be a candidate of a motif


def find_motif(seq, motif, bg=None):
    if bg is None:
        bg = {}
    l = len(motif[0])
    #best = float('-inf')
    best = -100
    idx = -1
    for i in range(0, len(seq) - l + 1):
        lmer = seq[i: i + l]
        score = 0
        for a, b, c in zip(lmer, motif[0], motif[1]):
            if a == b:
                score += log(float(c) / bg.get(a, 1.))
            else:
                score += log((100. - c) / bg.get(a, 1.))
                # try:
                #       score += log((100. - c) / bg.get(a, 1.))
                # except:
                #       print c, bg.get(a, 1.)

        if score >= best:
            idx = i
            best = score

    return [seq[idx: idx + l], len(seq) - idx, best]


##########################################################################
# cai, from biopython
##########################################################################
index = Counter({'GCT': 1, 'CGT': 1, 'AAC': 1, 'GAC': 1, 'TGC': 1, 'CAG': 1, 'GAA': 1, 'GGT': 1, 'CAC': 1, 'ATC': 1, 'CTG': 1, 'AAA': 1, 'ATG': 1, 'TTC': 1, 'CCG': 1, 'TCT': 1, 'ACC': 1, 'TGG': 1, 'TAC': 1, 'GTT': 1, 'ACT': 0.965, 'TCC': 0.744, 'GGC': 0.724, 'GCA': 0.586, 'TGT': 0.5, 'GTA': 0.495, 'GAT': 0.434, 'GCG': 0.424, 'AGC': 0.41, 'CGC': 0.356, 'TTT': 0.296, 'CAT': 0.291, 'GAG': 0.259,
                 'AAG': 0.253, 'TAT': 0.239, 'GTG': 0.221, 'ATT': 0.185, 'CCA': 0.135, 'CAA': 0.124, 'GCC': 0.122, 'ACG': 0.099, 'AGT': 0.085, 'TCA': 0.077, 'ACA': 0.076, 'CCT': 0.07, 'GTC': 0.066, 'AAT': 0.051, 'CTT': 0.042, 'CTC': 0.037, 'TTA': 0.02, 'TTG': 0.02, 'GGG': 0.019, 'TCG': 0.017, 'CCC': 0.012, 'GGA': 0.01, 'CTA': 0.007, 'AGA': 0.004, 'CGA': 0.004, 'CGG': 0.004, 'ATA': 0.003, 'AGG': 0.002})


def cai(seq):
    if seq.islower():
        seq = seq.upper()

    N = len(seq)
    cai_value, cai_length = 0, 0
    for i in range(0, N, 3):
        codon = seq[i: i + 3]
        if codon in index:
            if codon not in ['ATG', 'TGG']:
                cai_value += math.log(index[codon])
                cai_length += 1
        elif codon not in ['TGA', 'TAA', 'TAG']:
            continue
        else:
            continue

    if cai_length > 0:
        return math.exp(cai_value / cai_length)
    else:
        return 0


##########################################################################
# get the features
##########################################################################
# convert ATCG based kmer number
#code = {'A': 1, 'a': 1, 'T': 2, 't': 2, 'G': 3, 'g': 3, 'C': 4, 'c': 4}
code = [0] * 256
code5 = [0] * 256
flag = 0
for i in 'ATGC':
    code[ord(i.lower())] = code[ord(i)] = flag
    code5[ord(i.lower())] = code5[ord(i)] = flag + 1
    flag += 1

# convert string to number


def s2n(s, code=code, scale=None):
    if scale == None:
        scale = max(code) + 1
    N = 0
    output = 0
    for i in s[::-1]:
        #output += code.get(i, 0) * scale ** N
        output += code[ord(i)] * scale ** N
        N += 1

    return output

# reverse of s2n


def n2s(n, length, alpha='ATGC', scale=None):
    if scale == None:
        scale = max(code) + 1
    N = n
    s = []
    for i in range(length):
        s.append(alpha[N % scale])
        N /= scale

    return ''.join(s[::-1])


# convert the dna sequence to kmer-position matrix.
# if length of dna < given, then add NNN in the center of the sequence.
# else if length of dna > given, then trim the center of the sequence.

# the new kpm, reshape
def kpm(S, d=64, k=3, code=code, scale=None):
    if scale == None:
        scale = max(code) + 1

    N = scale ** k
    assert isinstance(d, int)
    L = len(S)
    if d < L:
        F = d // 2
        R = d - F
        seq = ''.join([S[: F], S[-R:]])
    elif d > L:
        F = L // 2
        R = L - F
        seq = ''.join([S[: F], 'N' * (d - L), S[-R:]])
    else:
        seq = S

    mat = [[0] * (d // k) for elem in range(N * k)]
    for i in range(0, d - k + 1):
        kmer = seq[i: i + k]
        if 'N' in kmer or 'n' in kmer:
            continue
        R = s2n(kmer, code=code, scale=scale)
        mat[R + i % k * N][i // k] = 1

    mat = np.asarray(mat, 'int8')
    return mat


def kpm0(S, d=64, k=3, code=code, scale=None):
    if scale == None:
        scale = max(code) + 1

    N = scale ** k
    assert isinstance(d, int)
    L = len(S)
    if d < L:
        F = d // 2
        R = d - F
        seq = ''.join([S[: F], S[-R:]])
    elif d > L:
        F = L // 2
        R = L - F
        seq = ''.join([S[: F], 'N' * (d - L), S[-R:]])
    else:
        seq = S

    mat = [[0] * (d // 3) for elem in range(N * 3)]
    for i in range(0, d - k + 1):
        kmer = seq[i: i + k]
        if 'N' in kmer or 'n' in kmer:
            continue
        R = s2n(kmer, code=code, scale=scale)
        mat[R + i % 3 * N][i // 3] = 1

    mat = np.asarray(mat, 'int8')
    return mat


# get features by give loc1, start and end:
# get xx
def get_xx(j, seq_dict, kmer=2, dim=128, mode='train', context=False):
    loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = j[: 10]
    if scf1 != scf2 or std1 != std2:
        if context:
            X0 = np.ones((4 ** kmer * kmer, dim // kmer * kmer))
        else:
            X0 = np.ones((4 ** kmer * kmer, dim // kmer))
        X1 = [10**4] * 11
        X2 = [127] * dim
        return [X0], X1, X2

    # get the sequence
    st1, ed1, st2, ed2 = list(map(int, [st1, ed1, st2, ed2]))
    st1 -= 1
    st2 -= 1

    if st1 > st2:
        loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = loc2, scf2, std2, st2, ed2, loc1, scf1, std1, st1, ed1

    seq1 = seq_dict[scf1][st1: ed1]
    seq1 = std1 == '+' and seq1 or seq1.reverse_complement()
    seq2 = seq_dict[scf2][st2: ed2]
    seq2 = std1 == '+' and seq2 or seq2.reverse_complement()

    start, end = ed1, st2
    seq12 = seq_dict[scf1][start: end]

    seq12 = std1 == '+' and seq12 or seq12.reverse_complement()
    seq1, seq2, seq12 = list(map(str, [seq1.seq, seq2.seq, seq12.seq]))
    seq1, seq2, seq12 = seq1.upper(), seq2.upper(), seq12.upper()

    # 1D features such as gc, dist
    cai1, cai2, cai12 = list(map(cai, [seq1, seq2, seq12]))
    dist = st2 - ed1
    distn = (st2 - ed1) * 1. / (ed2 - st1)
    ratio = math.log((ed1 - st1) * 1. / (ed2 - st2))
    ratio = std1 == '+' and ratio or -ratio
    idx = -100
    bgs = Counter(seq12[idx:])
    up10, up35 = find_motif(seq12[idx:], box_up10, bgs), find_motif(
        seq12[idx:], box_up35, bgs)
    if seq12[idx:]:
        gc = SeqUtils.GC(seq12[idx:])
        try:
            skew = SeqUtils.GC_skew(seq12[idx:])[0]
        except:
            skew = 0.
    else:
        gc = skew = 0.

    bias = sharekmer(seq1, seq2)
    if st1 == st2 == '+':
        X1 = [cai1, cai2, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]
    else:
        X1 = [cai2, cai1, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]

    # 2D features of kmer matrix
    if context:
        seqmat12 = kpm(seq12, d=dim, k=kmer, scale=4)
        seqmat1 = kpm(seq1, d=dim, k=kmer, scale=4)
        seqmat2 = kpm(seq2, d=dim, k=kmer, scale=4)
        seqmat = np.concatenate((seqmat1, seqmat12, seqmat2), 1)
    else:
        seqmat = kpm(seq12, d=dim, k=kmer, scale=4)

    if ed1 > st2:
        seqmat[:] = 0
    X0 = [seqmat]
    n12 = len(seq12)
    X2 = [s2n(seq12[elem: elem + kmer], code5)
          for elem in range(n12 - kmer + 1)]

    return X0, X1, X2


def get_xx0(j, seq_dict, kmer=2, dim=128, mode='train', context=False):
    loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = j[: 10]
    if scf1 != scf2 or std1 != std2:
        if context:
            X0 = np.ones((4 ** kmer * 3, dim // 3 * 3))
        else:
            X0 = np.ones((4 ** kmer * 3, dim // 3))
        X1 = [10**4] * 11
        X2 = [127] * dim
        return [X0], X1, X2

    # get the sequence
    st1, ed1, st2, ed2 = list(map(int, [st1, ed1, st2, ed2]))
    st1 -= 1
    st2 -= 1

    if st1 > st2:
        loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = loc2, scf2, std2, st2, ed2, loc1, scf1, std1, st1, ed1

    seq1 = seq_dict[scf1][st1: ed1]
    seq1 = std1 == '+' and seq1 or seq1.reverse_complement()
    seq2 = seq_dict[scf2][st2: ed2]
    seq2 = std1 == '+' and seq2 or seq2.reverse_complement()

    start, end = ed1, st2
    seq12 = seq_dict[scf1][start: end]

    seq12 = std1 == '+' and seq12 or seq12.reverse_complement()
    seq1, seq2, seq12 = list(map(str, [seq1.seq, seq2.seq, seq12.seq]))
    seq1, seq2, seq12 = seq1.upper(), seq2.upper(), seq12.upper()

    # 1D features such as gc, dist
    cai1, cai2, cai12 = list(map(cai, [seq1, seq2, seq12]))
    dist = st2 - ed1
    distn = (st2 - ed1) * 1. / (ed2 - st1)
    ratio = math.log((ed1 - st1) * 1. / (ed2 - st2))
    ratio = std1 == '+' and ratio or -ratio
    idx = -100
    bgs = Counter(seq12[idx:])
    up10, up35 = find_motif(seq12[idx:], box_up10, bgs), find_motif(
        seq12[idx:], box_up35, bgs)
    if seq12[idx:]:
        gc = SeqUtils.GC(seq12[idx:])
        try:
            skew = SeqUtils.GC_skew(seq12[idx:])[0]
        except:
            skew = 0.
    else:
        gc = skew = 0.

    bias = sharekmer(seq1, seq2)
    if st1 == st2 == '+':
        X1 = [cai1, cai2, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]
    else:
        X1 = [cai2, cai1, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]

    # 2D features of kmer matrix
    if context:
        seqmat12 = kpm(seq12, d=dim, k=kmer, scale=4)
        seqmat1 = kpm(seq1, d=dim, k=kmer, scale=4)
        seqmat2 = kpm(seq2, d=dim, k=kmer, scale=4)
        seqmat = np.concatenate((seqmat1, seqmat12, seqmat2), 1)
    else:
        seqmat = kpm(seq12, d=dim, k=kmer, scale=4)

    if ed1 > st2:
        seqmat[:] = 0
    X0 = [seqmat]
    n12 = len(seq12)
    X2 = [s2n(seq12[elem: elem + kmer], code5)
          for elem in range(n12 - kmer + 1)]

    return X0, X1, X2


# get single line of features
def get_xx_one(j, seq_dict, kmer=2, dim=128, mode='train'):
    X0, X1, X2 = get_xx(j, seq_dict, kmer, dim, mode)
    x0, x1, x2 = list(map(np.asarray, [[X0], [X1], [X2]]))
    return x0, x1, X2

# generate training and testing data


def get_xxy(f, seq_dict, kmer=2, dim=128):
    # get the training data
    X0, X1, X2, y = [], [], [], []

    for i in f:
        j = i[:-1].split('\t')
        x0, x1, x2 = get_xx(j, seq_dict, kmer, dim)
        X0.append(x0)
        X1.append(x1)
        X2.append(x2)
        y.append(j[-1] == 'True' and 1 or 0)

    X0 = np.asarray(X0, 'int8')
    X1 = np.asarray(X1, 'float32')
    X2 = np.asarray(X2)
    y = np.asarray(y, 'int8')
    return X0, X1, X2, y

# split the X0, X1, y data to training and testing


def split_xxy(X0, X1, X2, y, train_size=1. / 3, seed=42):
    N = X0.shape[0]
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    start = int(train_size * N)
    idx_train, idx_test = idx[: start], idx[start:]
    X0_train, X1_train, X2_train, y_train = X0[
        idx_train], X1[idx_train], X2[idx_train], y[idx_train]
    X0_test, X1_test, X2_test, y_test = X0[idx_test], X1[
        idx_test], X2[idx_test], y[idx_test]

    return X0_train, X1_train, X2_train, y_train, X0_test, X1_test, X2_test, y_test




##########################################################################
# the lstm based
##########################################################################

# get lstm features by give loc1, start and end:
def get_lstm_xx(j, seq_dict, kmer=2, dim=128, mode='train'):
    loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = j[: 10]
    if scf1 != scf2 or std1 != std2:
        #X0 = np.ones((4 ** kmer, dim))
        X0 = [127] * dim
        #X0 = None
        X1 = [10**4] * 11
        return X0, X1

    # get the sequence
    st1, ed1, st2, ed2 = list(map(int, [st1, ed1, st2, ed2]))
    st1 -= 1
    st2 -= 1

    if st1 > st2:
        loc1, scf1, std1, st1, ed1, loc2, scf2, std2, st2, ed2 = loc2, scf2, std2, st2, ed2, loc1, scf1, std1, st1, ed1
    seq1 = seq_dict[scf1][st1: ed1]
    seq1 = std1 == '+' and seq1 or seq1.reverse_complement()
    seq2 = seq_dict[scf2][st2: ed2]
    seq2 = std1 == '+' and seq2 or seq2.reverse_complement()

    start, end = ed1, st2
    seq12 = seq_dict[scf1][start: end]

    # if len(seq12) > dim:
    #    seq12 = seq12[: dim // 2] + seq12[-dim // 2: ]

    seq12 = std1 == '+' and seq12 or seq12.reverse_complement()
    seq1, seq2, seq12 = list(map(str, [seq1.seq, seq2.seq, seq12.seq]))
    seq1, seq2, seq12 = seq1.upper(), seq2.upper(), seq12.upper()

    # 1D features such as gc, dist
    cai1, cai2, cai12 = list(map(cai, [seq1, seq2, seq12]))
    dist = st2 - ed1
    distn = (st2 - ed1) * 1. / (ed2 - st1)
    ratio = math.log((ed1 - st1) * 1. / (ed2 - st2))
    ratio = std1 == '+' and ratio or -ratio
    idx = -100
    bgs = Counter(seq12[idx:])
    up10, up35 = find_motif(seq12[idx:], box_up10, bgs), find_motif(
        seq12[idx:], box_up35, bgs)
    if seq12[idx:]:
        gc = SeqUtils.GC(seq12[idx:])
        try:
            skew = SeqUtils.GC_skew(seq12[idx:])[0]
        except:
            skew = 0.
    else:
        gc = skew = 0.

    bias = sharekmer(seq1, seq2)
    if st1 == st2 == '+':
        X1 = [cai1, cai2, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]
    else:
        X1 = [cai2, cai1, bias, distn, ratio, gc, skew] + up10[1:] + up35[1:]
        #X1 = [cai1, cai2, bias, distn, ratio, gc, skew] + up10[1: ] + up35[1: ]

    # 1D features of lstm
    n12 = len(seq12)

    '''
    L = dim // 2
    R = dim - L
    if n12 > dim:
        seq12 = seq12[: L] + seq12[-R: ]
    else:
        seq12 = seq12[: L] +'N' * (dim - n12)  + seq12[-R: ]
    '''
    lstm_seq = [s2n(seq12[elem: elem + kmer], code5)
                for elem in range(n12 - kmer + 1)]
                
    #X0 = lstm_seq[::kmer]
    #for i in xrange(kmer):
    #    X0.extend(lstm_seq[i::kmer])
    #X0 = lstm_seq
    X0 = [-1] * dim
    ndim = len(lstm_seq)
    if ndim == dim:
        X0 = lstm_seq
        #print 'X0 0', len(X0)

    elif 2 <= ndim < dim:
        ndim = ndim // 2
        X0[:ndim] = lstm_seq[:ndim]
        X0[-ndim:] = lstm_seq[-ndim:]
        #print 'X0 1', len(X0)
    elif ndim > dim:
        ndim = dim // 2
        X0[:ndim] = lstm_seq[:ndim]
        X0[-ndim:] = lstm_seq[-ndim:]
        #print 'X0 2', len(X0)
    else:
        pass

    return X0, X1

# get single line of lstm features


def get_lstm_xx_one(j, seq_dict, kmer=2, dim=128, mode='train'):
    X0, X1 = get_lstm_xx(j, seq_dict, kmer, dim, mode)
    x0, x1 = list(map(np.asarray, [[X0], [X1]]))
    return x0, x1

# generate training and testing data


def get_lstm_xxy(f, seq_dict, kmer=2, dim=128):
    # get the training data
    X0, X1, y = [], [], []
    for i in f:
        j = i[:-1].split('\t')
        #print 'line', j
        x0, x1 = get_lstm_xx(j, seq_dict, kmer, dim)
        #print 'X0 is', len(x0)
        # if len(x0) < 1 or x0[0] == -1:
        #    continue
        cat = j[-1]
        X0.append(x0)
        X1.append(x1)
        y.append(cat == 'True' and 1 or 0)

    #print max(map(len, X0)), min(map(len, X0)), X0[: 2]

    X0 = np.asarray(X0)
    print(('X0 shape', X0.shape))
    #print 'cat1, cat2, cai12, gc, skew, dist, distn, ratio, up10, up35'
    #print 'X1', X1[0]
    X1 = np.asarray(X1, 'float32')
    y = np.asarray(y, 'int8')

    #print 'X0', X0, X1, y
    return X0, X1, y


# split lstm xxy
def split_lstm_xxy(X0, X1, y, train_size=1. / 3, seed=42):
    #X = np.asarray(X)
    #y = np.asarray(y)
    N = X0.shape[0]
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    start = int(train_size * N)
    idx_train, idx_test = idx[: start], idx[start:]
    X0_train, X1_train, y_train = X0[idx_train], X1[idx_train], y[idx_train]
    X0_test, X1_test, y_test = X0[idx_test], X1[idx_test], y[idx_test]

    return X0_train, X1_train, y_train, X0_test, X1_test, y_test

##########################################################################
# cnn based on pytorch
##########################################################################
class Net_torch(nn.Module):
#class Net_torch:

    def __init__(self, shape=(-1,-1,-1), nb_filter=32, nb_conv=3, nb_pool=2, adaptive=True):
        super(Net_torch, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        channel, row, col = shape
        self.nb_filter = nb_filter
        self.nb_conv = nb_conv
        self.nb_pool = nb_pool
        self.conv2d0 = nn.Conv2d(channel, nb_filter, nb_conv)

        if adaptive:
            N, D = row//2, col//2
            self.pool = nn.AdaptiveMaxPool2d((N, D))
        else:
            #N = (row-nb_conv+1) / nb_pool * (col-nb_conv+1) / nb_pool * nb_filter
            N, D = (row-nb_conv+1) / nb_pool, (col-nb_conv+1) / nb_pool
            #N = nb_filter * a * b
            self.pool = nn.MaxPool2d(nb_pool, stride=(2, 2))
            #print 'N size', N,  nb_filter, a, b

        self.relu = nn.ReLU()

        #self.fc0 = nn.Linear(N, 128)
        self.fc0 = nn.Linear(nb_filter*N*D, 128)

        self.drop = nn.Dropout(.85)
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Sigmoid()
        # an affine operation: y = Wx + b

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv(x)), (self.nb_pool, self.nb_pool))
        x = self.conv2d0(x)
        x = self.relu(x)
        x = self.pool(x)
        #print 'pool shape', x.shape
        #x = self.drop(x)

        # If the size is a square you can only specify a single number
        #print 'x0', x.size(), np.prod(x.size()[1:])
        x = x.view(x.size(0), -1)
        N, D = x.shape
        #print 'x0 shape', D
        #x = F.linear(x, (128, D))
        x = self.fc0(x)
        #x = F.dropout(x, training=self.training)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        #6for s in size:
        num_features *= s
        return num_features

# bceloss for chainer
def BCEloss(y, y_p, eps=1e-9):
    #X = np.asarray(x.data, 'float32')
    #T = np.asarray(t.data, 'float32')

    try:
        y.to_cpu()
    except:
        pass

    Y = Var(np.asarray(y.data, 'float32'))
    try:
        y.to_gpu()
    except:
        pass

    try:
        y_p.to_cpu()
    except:
        pass

    Y_p = Var(np.asarray(y_p.data, 'float32'))

    try:
        y_p.to_gpu()
    except:
        pass

    N = y.shape[0]

    print('Y_p', cF.log(1 - Y_p))
    hs = Y*cF.log(Y_p+eps) + (1.-Y)*cF.log(1.-Y_p)
    h = hs.data.sum() / -N
    return h

class Net_ch(chainer.Chain):

    def __init__(self, nb_filter=32, nb_conv=3, nb_pool=2, n_out=2):

        super(Net_ch, self).__init__()
        with self.init_scope():
            self.nb_pool = nb_pool

            self.conv1=cL.Convolution2D(None, nb_filter, ksize=nb_conv)
            self.bn1 = cL.BatchNormalization(nb_filter)

            self.incept0=cL.Inception(None,  64,  96, 128, 16,  32,  32)
            self.incept1=cL.Inception(None, 128, 128, 192, 32,  96,  64)

            self.conv2=cL.Convolution2D(None, nb_filter*2, ksize=nb_conv)
            self.bn2 = cL.BatchNormalization(nb_filter*2)

            #self.vgg = cL.VGG16Layers()
            #self.google = cL.GoogLeNet()
            self.fc0 = cL.Linear(None, 256)
            self.fc1=cL.Linear(None, 128)
            self.fc2=cL.Linear(None, n_out)

        para = [nb_filter, nb_conv, nb_pool, n_out]
        #self.add_persistent("n_out", n_out)
        self.add_persistent("para", para)

    def __call__(self, X, t=None):
        #x = self.google(X)
        x = self.conv1(X)
        #x = self.incept0(x)
        #x = self.incept1(x)
        #x = self.bn1(x)
        x = cF.relu(x)

        #x = self.conv2(X)
        #x = self.bn2(x)
        #x = cF.relu(x)

        x = cF.max_pooling_2d(x, ksize=self.nb_pool)
        x = cF.dropout(x, .85)
        x = self.fc0(x)
        x = cF.relu(x)
        x = cF.dropout(x, .85)
        x = self.fc1(x)
        x = cF.relu(x)
        x = cF.dropout(x, .85)
        x = self.fc2(x)
        x = cF.sigmoid(x)
        #x = cF.flatten(x)
        if t is None:
            #print 'x t shape', x.shape
            return x
        else:
            #return cF.hinge(x, t)
            #z = BCEloss(x, t)
            z = cF.sigmoid_cross_entropy(x, t)
            #print 'x, t, z, shape', x.shape, t.shape, z.shape
            return z



# VGG 16 model
class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = cL.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = cL.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return cF.relu(h)

class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block3_3 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(512, 3)
            self.block4_3 = Block(512, 3)
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)
            self.fc1 = cL.Linear(None, 512, nobias=True)
            self.bn_fc1 = cL.BatchNormalization(512)
            self.fc2 = cL.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = cF.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = cF.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = cF.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = cF.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = cF.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = cF.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = cF.max_pooling_2d(h, ksize=2, stride=2)

        h = cF.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = cF.relu(h)
        h = cF.dropout(h, ratio=0.5)
        return self.fc2(h)



# y_p: predicted y
def f1_score_torch(Y, Y_p, eps=1e-9):

    #print 'Y Y_p', Y, Y_p
    #y = Y.data.cpu().numpy()
    #y_p = Y_p.data.cpu().numpy()
    y, y_p = Y, Y_p
    TP = ((y * y_p) > 0).sum()+eps
    RP = (y>0).sum()+eps
    PP = (y_p>0).sum()+eps
    prc = TP * 1. / PP
    rec = TP * 1. / RP
    f1 = 2.*prc*rec/(prc+rec)
    return f1


##########################################################################
# the CNN class
##########################################################################
class CNN:

    def __init__(self, nb_filter=32, nb_pool=3, nb_conv=2, nb_epoch=10, batch_size=16, maxlen=128, save_path='./weights.hdf5'):
        # def __init__(self, nb_filter=64, nb_pool=3, nb_conv=2, nb_epoch=10,
        # batch_size=32, maxlen=128, save_path='./weights.hdf5'):

        self.nb_filter = nb_filter
        self.nb_pool = nb_pool
        self.nb_conv = nb_conv
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.maxlen = maxlen
        #self.opt = Adam(lr=5e-4, beta_1=0.995, beta_2=0.999, epsilon=1e-09)
        #self.opt = AdaBound(lr=1e-3, final_lr=0.1)
        self.opt = AdaBound(lr=1e-3, final_lr=0.1, gamma=1e-3, weight_decay=0., amsbound=False)

        #self.checkpointer = [ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True, mode='max', monitor='val_fbeta_score')]
        self.checkpointer = [ModelCheckpoint(
            filepath=save_path, verbose=1, save_best_only=True, mode='max', monitor='val_f1_score_keras')]
        #self.metric = keras.metrics.fbeta_score
        self.metric = f1_score_keras
        #self.metric = f1_score
        self.cross_val = 1. / 3

    def fit_2d(self, X_train, y_train, X_test=None, y_test=None):
        np.random.seed(42)
        set_random_seed(42)
        Y_train = np_utils.to_categorical(y_train).astype('float32')
        if type(y_test) == type(None):
            Y_test = None
        else:
            Y_test = np_utils.to_categorical(y_test).astype('float32')

        nb_classes = Y_train.shape[1]

        # set parameter for cnn
        loss = nb_classes > 2 and 'categorical_crossentropy' or 'binary_crossentropy'
        #6print 'loss function is', loss
        # number of convolutional filters to use
        nb_filters = self.nb_filter
        # size of pooling area for max pooling
        nb_pool = self.nb_pool
        # convolution kernel size
        nb_conv = self.nb_conv
        # traning iteration
        nb_epoch = self.nb_epoch
        batch_size = self.batch_size
        a, b, img_rows, img_cols = X_train.shape

        # set the conv model
        model = Sequential()
        #model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), border_mode='same', input_shape=(b, img_rows, img_cols), activation='relu', name='conv1_1', data_format='channels_first'))
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), border_mode='same', input_shape=(b, img_rows, img_cols), activation='relu', name='conv1_1', data_format='channels_first'))
        #model.add(Conv2D(nb_filters, 1, border_mode='same', input_shape=(b, img_rows, img_cols), name='conv1_1', data_format='channels_first'))

        #model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2', data_format='channels_first'))
        #model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2', data_format='channels_first'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))

        #model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_first'))
        #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_first'))
        #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_first'))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        #model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.85))
        model.add(Dense(nb_classes, activation='sigmoid'))

        # try:
        #    model = make_parallel(model, 2)
        # except:
        #    pass

        opt = self.opt
        model.compile(loss=loss, optimizer='adam', metrics=[self.metric])

        model.summary()

        # set the check pointer to save the best model
        if type(X_test) != type(None) and type(Y_test) != type(None):
        #if 0:
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                      validation_data=(X_test, Y_test), shuffle=True, validation_split=1e-4, callbacks=self.checkpointer)
        else:
            model.fit(X_train, Y_train, batch_size=batch_size,
                      epochs=nb_epoch, verbose=1, shuffle=True, validation_split=self.cross_val, callbacks=self.checkpointer)

        self.model_2d = model

    def retrain_2d(self, X_train, y_train, X_test=None, y_test=None):

        Y_train = np_utils.to_categorical(y_train).astype('float32')
        if type(y_test) == type(None):
            Y_test = None
        else:
            Y_test = np_utils.to_categorical(y_test).astype('float32')

        if type(X_test) != type(None) and type(Y_test) != type(None):
        #if 0:
            self.model_2d.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.nb_epoch, verbose=1,
                      validation_data=(X_test, Y_test), shuffle=True, validation_split=1e-4, callbacks=self.checkpointer)
        else:
            self.model_2d.fit(X_train, Y_train, batch_size=self.batch_size,
                      epochs=self.nb_epoch, verbose=1, shuffle=True, validation_split=self.cross_val, callbacks=self.checkpointer)

    def predict_2d(self, X):
        return self.model_2d.predict(X).argmax(1)


    def fit_lstm(self, X_train, y_train, X_test=None, y_test=None):
        self.max_features = 2**12
        #print X_train.shape, y_train.shape
        N, D = X_train.shape
        model = Sequential()
        model.add(Embedding(self.max_features, D))
        #model.add(LSTM(D, dropout=0.2, recurrent_dropout=0.2))
        model.add(Bidirectional(CuDNNGRU(D, return_sequences=True)))
        #model.add(Bidirectional(GRU(D, return_sequences=True)))
        #model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
        model.add(Dropout(0.2))

        model.add(Bidirectional(CuDNNGRU(D)))
        #model.add(Bidirectional(GRU(D)))

        #model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        nb_classes = len(set(y_train))
        loss = nb_classes > 2 and 'categorical_crossentropy' or 'binary_crossentropy'
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.metric])
        model.compile(loss=loss, optimizer='adam', metrics=[self.metric])
        print(('Train..., loss is %s %s'%(loss, D)))
        if type(X_test) != type(None) and type(y_test) != type(None):
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(X_test, y_test), shuffle=True, callbacks=self.checkpointer)
        else:
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(X_test, y_test), verbose=1, shuffle=True, validation_split=self.cross_val, callbacks=self.checkpointer)
        score, acc = model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(('Test score:', score))
        print(('Test accuracy:', acc))
        self.model_2d = model


    def retrain_lstm(self, X_train, y_train, X_test=None, y_test=None):
        if type(X_test) != type(None) and type(y_test) != type(None):
            self.model_2d.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(X_test, y_test), shuffle=True, callbacks=self.checkpointer)
        else:
            self.model_2d.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch, validation_data=(X_test, y_test), verbose=1, shuffle=True, validation_split=self.cross_val, callbacks=self.checkpointer)
        score, acc = self.model_2d.evaluate(X_test, y_test, batch_size=self.batch_size)


    def predict_lstm(self, X):
        return self.model_2d.predict_classes(X).flatten()


    # pytorch based cnn
    def fit_cnn(self, X_train, y_train, X_test=None, y_test=None):

        nb_filter = self.nb_filter
        # size of pooling area for max pooling
        nb_pool = self.nb_pool
        # convolution kernel size
        nb_conv = self.nb_conv
        # traning iteration
        nb_epoch = self.nb_epoch
        batch_size = self.batch_size
        a_tr, b_tr, rows, cols = X_train.shape

        #print 'data size', a, b, img_rows, img_cols
        #print 'model parameter', nb_filter, nb_pool

        loss_func = nn.BCELoss()
        #loss_func = F.nll_loss
        # add the layer
        cnn = Net_torch((b_tr, rows, cols), nb_filter, nb_pool, nb_conv)
        self.model_2d = None
        #cnn = Net()
        cnn.train()
        print('model parameters')
        print(cnn)

        cnn.cuda()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=5e-4)
        #optimizer = torch.optim.SGD(cnn.parameters(), lr=5e-4)
        #dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=y_train)
        #loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2,)

        #x_n = np.asarray(X_test, 'float32')
        #y_n = np.asarray(y_test, 'uint8')
        #x_t = torch.from_numpy(x_n).cuda()
        #y_t = torch.from_numpy(y_n).cuda()
        #X_t = Variable(x_t)
        #Y_t = Variable(y_t)


        f1_best = 0
        idx_tr = np.arange(a_tr)

        a_te, b_te, rows, cols = X_test.shape
        idx_te = np.arange(a_te)
        y_test_p = np.empty(a_te, dtype='int8')

        best_model = None
        for i in range(nb_epoch):
            #np.random.seed(42)
            np.random.shuffle(idx_tr)
            cnn.train()
            loss_val = 0
            #optimizer.zero_grad()
            for j in range(0, a_tr, batch_size):
                idx = idx_tr[j:j+batch_size]
                #x_n = X_train[j:j+batch_size]
                x_n = X_train[idx]
                #y_n = y_train[j:j+batch_size]
                y_n = y_train[idx]

                x_n = np.asarray(x_n, 'float32')
                y_n = np.asarray(y_n, 'float32')
                x = torch.from_numpy(x_n).cuda()
                y = torch.from_numpy(y_n).cuda()

            #for step, (x, y) in enumerate(loader):
                X = Variable(x)
                Y = Variable(y)
                optimizer.zero_grad()
                Y_pred = cnn(X)
                #print 'Y pred', i, x_n.shape, Y_pred.shape, Y.shape
                loss = loss_func(Y_pred, Y)
                loss_val += loss.data[0]
                loss.backward()
                optimizer.step()
                #print 'loss:', i, loss.data[0]
            print(('loss:', loss_val))
            # evaluation
            cnn.eval()

            #for j in xrange(0, a_te, batch_size):
            #    idx = idx_te[j:j+batch_size]
            #    x_n = X_test[idx]
            #    #y_n = y_test[idx]
            #    #x_n = np.asarray(x_n, 'float32')
            #    #y_n = np.asarray(y_n, 'float32')
            #    #x = torch.from_numpy(x_n).cuda()
            #    #y = torch.from_numpy(y_n).cuda()
            #    #X = Variable(x)
            #    #Y_p = cnn(X).squeeze()
            #    #Y_p = (Y_p > .5).data.cpu().numpy()
            #    Y_p = self.predict_cnn(x_n, cnn)
            #    y_test_p[j:j+batch_size] = Y_p

            y_test_p = self.predict_cnn_batch(X_test, cnn)

            prc = metrics.precision_score(y_test, y_test_p)
            rec = metrics.recall_score(y_test, y_test_p)
            f1 = metrics.f1_score(y_test, y_test_p)
            print(('iteration', i, 'precision:', prc, 'recall:', rec, 'f1:', f1, 'data_size:', X_test.shape))

            #print 'y_pred', Y_p, Y_t[:64]
            #if  f1 > f1_best and prc > .87 and rec > .87:
            if  f1 > f1_best and prc > .87 and rec > .87:
                #best_model = deepcopy(cnn)
                #if prc > .87 and rec > .87:
                #    print 'best f1 score:', f1
                #    #torch.save(cnn, 'pytch.pt')
                #    self.model_2d = best_model
                print('save best model')
                self.model_2d = deepcopy(cnn)
                f1_best = f1

        if not self.model_2d:
            self.model_2d = cnn


    def predict_cnn(self, X, model=None):
        #x_n = np.asarray(X, 'float32')
        N = len(X)
        #batch = self.batch_size
        batch = self.batch_size
        #Y = np.empty(N, dtype='int8')
        Y = Variable(torch.Tensor(N)).cuda()
        cnn = model and model or self.model_2d
        cnn.eval()

        for i in range(0, N, batch):
            j = min(N, i + batch)
            x_n = np.asarray(X[i:j], 'float32')
            x = torch.from_numpy(x_n).cuda()
            x_v = Variable(x)
            y_p = cnn(x_v).squeeze()
            #print 'y_p', y_p
            Y[i:j] = y_p
            del x_v


        Y = (Y > .5).data.cpu().numpy()
        #Y = (Y > .5).data.cpu().numpy()
        #Y = Y > .5
        return Y


    def predict_cnn_batch(self, X, model=None):
        N = len(X)
        batch_size = self.batch_size
        y_p = np.empty(N, 'int8')

        cnn = model and model or self.model_2d

        for i in range(0, N, batch_size):
            j = min(i+batch_size, N)
            x_n = X[i:j]
            p = self.predict_cnn(x_n, cnn)
            y_p[i:j] = p

        return y_p


    # chainer based cnn
    def fit_chn(self, X_train, y_train, X_test=None, y_test=None):
        chainer.cuda.cudnn_enabled = True

        nb_filter = self.nb_filter
        # size of pooling area for max pooling
        nb_pool = self.nb_pool
        # convolution kernel size
        nb_conv = self.nb_conv
        # traning iteration
        nb_epoch = self.nb_epoch
        batch_size = self.batch_size
        a_tr, b_tr, rows, cols = X_train.shape

        #loss = cF.sigmoid_cross_entropy
        #loss_func = F.nll_loss
        # add the layer
        n_out = np.unique(y_train).shape[0]
        #n_out = 1
        Net = Net_ch(nb_filter, nb_conv, nb_pool, n_out)
        #Net = VGG(n_out)
        #Net = cL.GoogLeNet()

        #cnn = Net
        #cnn = cL.Classifier(Net, lossfun=BCEloss)
        #cnn = cL.Classifier(Net, lossfun=cF.hinge)
        #cnn = cL.Classifier(Net, lossfun=cF.sigmoid_cross_entropy)
        cnn = Net
        #CNN = Net 
        #CNN.to_gpu()
        loss_func = n_out <= 2 and cF.sigmoid_cross_entropy or cF.softmax_cross_entropy

        self.model_2d = None
        #cnn = Net()
        print(('model parameters', n_out))

        # enable gpu if available
        try:
            cnn.to_gpu()
        except:
            pass

        optimizer = chainer.optimizers.Adam(5e-4, adabound=True)
        #optimizer = chainer.optimizers.MomentumSGD(1e-3)
        optimizer.setup(cnn)
        #optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))


        f1_best = 0
        idx_tr = np.arange(a_tr)

        a_te, b_te, rows, cols = X_test.shape
        idx_te = np.arange(a_te)
        y_test_p = np.empty(a_te, dtype='int')
        #y_test_p = np.empty(a_te, dtype='float32')

        best_model = None
        for i in range(nb_epoch):
            #np.random.seed(42)
            np.random.shuffle(idx_tr)
            loss_val = 0
            #optimizer.zero_grad()
            for j in range(0, a_tr, batch_size):
                idx = idx_tr[j:j+batch_size]
                xn = X_train[idx]
                yn = y_train[idx]
                xn = np.asarray(xn, 'float32')
                yn = np.asarray(yn, 'int')
                if n_out <= 2:
                    #print 'reshape'
                    yn = np_utils.to_categorical(yn).astype('i')

                #yn = np.asarray(yn, 'float32')
                X, Y = list(map(Var, (xn, yn)))

                # move data to gpu
                try:
                    X.to_gpu()
                    Y.to_gpu()
                except:
                    pass

                #Y_p = CNN(X)
                #print 'Y Y_p', Y.shape, Y_p.shape, Y[:2, ], Y_p[:2, ]
                #print 'Y loss get', cF.sigmoid_cross_entropy(Y_p, Y)
                #print('bceloss', BCEloss(Y, Y_p))

                Y_p = cnn(X)
                #loss = cF.sigmoid_cross_entropy(Y_p, Y)
                loss = loss_func(Y_p, Y)
                #loss = cnn(X, Y)

                # cal grad
                cnn.cleargrads()
                loss.backward()
                # update parameter
                optimizer.update()

                #optimizer.update(cnn, X, Y)

                #loss_val += loss.data

                #print 'loss:', i, loss.data[0]
            #print 'loss:', loss_val, Y_p.shape


            st = time()
            y_train_p = self.predict_chn(X_train, Net)

            prc = metrics.precision_score(y_train, y_train_p)
            rec = metrics.recall_score(y_train, y_train_p)
            f1 = metrics.f1_score(y_train, y_train_p)
            print((('train iteration', i, 'precision:', prc, 'recall:', rec, 'f1:', f1, 'data_size:', X_train.shape, 'time:', time()-st)), end=' ')
            #continue

            st = time()
            y_test_p = self.predict_chn(X_test, Net)

            prc = metrics.precision_score(y_test, y_test_p)
            rec = metrics.recall_score(y_test, y_test_p)
            f1 = metrics.f1_score(y_test, y_test_p)
            print(('test iteration', i, 'precision:', prc, 'recall:', rec, 'f1:', f1, 'data_size:', X_test.shape, 'time:', time()-st))

            #print 'y_pred', Y_p, Y_t[:64]
            #if  f1 > f1_best and prc > .87 and rec > .87:
            if  f1 > f1_best and prc > .84 and rec > .84:
                #best_model = deepcopy(cnn)
                #if prc > .87 and rec > .87:
                #    print 'best f1 score:', f1
                #    #torch.save(cnn, 'pytch.pt')
                #    self.model_2d = best_model
                print('save best model')
                #self.model_2d = deepcopy(Net)
                serializers.save_npz('my.model', Net)
                self.model_2d = 1
                f1_best = f1

        if not self.model_2d:
            self.model_2d = Net
        else:
            serializers.load_npz('my.model', Net)
            self.model_2d = Net



    def predict_chn(self, X, model=None, train=True):
        cnn = model and model or self.model_2d
        #print(model, cnn)

        N = X.shape[0]
        batch = self.batch_size
        #print('y size', N)
        #Y = Var(np.empty(N, dtype='int8'))
        Y = Var(np.empty(N, dtype='float32'))
        if train:

            try:
                Y.to_gpu()
            except:
                pass

        for i in range(0, N, batch):
            x = Var(np.asarray(X[i:i+batch], 'float32'))
            if train:

                try:
                    x.to_gpu()
                except:
                    pass

            y = cnn(x)
            #print('y is', y)
            #y.to_cpu()
            #y = cF.argmax(y, 1).data
            y = cF.argmax(y, 1)
            Y.data[i:i+batch] = y.data

        try:
            Y.to_cpu()
        except:
            pass

        #print Y.shape, Y.data[:5]
        #return Y.data.flatten() > .5
        return Y.data


    # load an training model
    def load(self, name, mode='2d'):
        #model = keras.models.load_model(name, custom_objects={'f1_score': f1_score, 'tf': tf, 'fbeta_score': f1_score})
        if mode == '2d' or mode == 'lstm':
            dependencies ={'f1_score_keras': f1_score_keras, 'fbeta_score': f1_score_keras}
            model = keras.models.load_model(name, custom_objects=dependencies)
            #model = keras.models.load_model(name, custom_objects={'fbeta_score': f1_score_keras})
            #model = keras.models.load_model(name)
            self.model_2d = model
        elif mode == 'torch':
            pass
        elif mode == 'chainer':
            #print('name', name)
            nets = np.load(name)
            #nb_filter=32
            #nb_pool=3
            #nb_conv=2
            #n_out = int(nets['n_out'])
            nb_filter, nb_conv, nb_pool, n_out = nets['para']
            del nets
            gc.collect()

            net = Net_ch(nb_filter, nb_conv, nb_pool, n_out)
            serializers.load_npz(name, net)
            self.model_2d = net

    # save the model
    def save(self, name, model='2d'):
        if model == '2d' or model == 'lstm':
            self.model_2d.save(name + '_' + model + '_%d_%d.hdf5' % (global_kmr, global_len))
        elif model == 'torch':
            out_name = name + '_' + model + '_%d_%d.pkl' % (global_kmr, global_len)
            torch.save(self.model_2d, out_name)

        else:
            pass


# run training
def run_train(train, seq_dict, clf, mode='2d', rounds=0):
    # get the training data
    split_rate = global_split_rate

    if mode == '2d':
        print('keras CNN training')
        f = open(train, 'r')
        #X, X1, X2, y = get_xxy(f, seq_dict, 3, 128)
        #X, X1, X2, y = get_xxy(f, seq_dict, 4, 64)
        X, X1, X2, y = get_xxy(f, seq_dict, global_kmr, global_len)
        X_train, X1_train, X2_train, y_train, X_test, X1_test, X2_test, y_test = split_xxy(
            X, X1, X2, y, split_rate)
        f.close()

        #X_train = np.asarray(X_train, 'float32')
        #X_test = np.asarray(X_test, 'float32')

        #clf.fit_2d(X_train, y_train, X_test, y_test)
        clf.fit_2d(X_train, y_train)

        for idx in range(rounds):
            print(('retraining cnn', idx))
            #X_train, X1_train, X2_train, y_train, X_test, X1_test, X2_test, y_test = split_xxy(X, X1, X2, y, split_rate, seed=43+idx)
            #clf.retrain_2d(X_train, y_train, X_test, y_test)
            X_train0, X1_train0, X2_train0, y_train0, X_test0, X1_test0, X2_test0, y_test0 = split_xxy(X_train, X1_train, X2_train, y_train, 3./4, seed=43+idx)
            clf.retrain_2d(X_train0, y_train0, X_test0, y_test0)
            #clf.retrain_2d(X_train0, y_train0, X_test, y_test)
        # the test score
        Y_test = np_utils.to_categorical(y_test).astype('float32')
        score = clf.model_2d.evaluate(X_test, Y_test, verbose=0)

        print(('   Test score:', score[0]))
        print(('Test accuracy:', score[1]))

        # validate
        y_test_pred = clf.predict_2d(X_test)
        y_train_pred = clf.predict_2d(X_train)

    elif mode == 'torch':
        print('pytorch CNN training')
        f = open(train, 'r')
        #X, X1, X2, y = get_xxy(f, seq_dict, 3, 128)
        #X, X1, X2, y = get_xxy(f, seq_dict, 4, 64)
        X, X1, X2, y = get_xxy(f, seq_dict, global_kmr, global_len)
        X_train, X1_train, X2_train, y_train, X_test, X1_test, X2_test, y_test = split_xxy(
            X, X1, X2, y, split_rate)
        f.close()
        clf.fit_cnn(X_train, y_train, X_test, y_test)
        y_test_pred = clf.predict_cnn_batch(X_test)
        y_train_pred = clf.predict_cnn_batch(X_train)



    elif mode.startswith('chain'):
        print('chainer CNN training')
        f = open(train, 'r')
        #X, X1, X2, y = get_xxy(f, seq_dict, 3, 128)
        #X, X1, X2, y = get_xxy(f, seq_dict, 4, 64)
        X, X1, X2, y = get_xxy(f, seq_dict, global_kmr, global_len)
        X_train, X1_train, X2_train, y_train, X_test, X1_test, X2_test, y_test = split_xxy(
            X, X1, X2, y, split_rate)
        f.close()
        print('X_test shape', X_test.shape, type(X_test))

        clf.fit_chn(X_train, y_train, X_test, y_test)
        y_test_pred = clf.predict_chn(X_test)
        y_train_pred = clf.predict_chn(X_train)

    elif mode == 'lstm':

        f = open(train, 'r')
        #X, X1, X2, y = get_xxy(f, seq_dict, 3, 128)
        #X, X1, X2, y = get_xxy(f, seq_dict, 4, 64)
        X, X1, y = get_lstm_xxy(f, seq_dict, global_kmr, global_len)
        X_train, X1_train, y_train, X_test, X1_test, y_test = split_lstm_xxy(X, X1, y, split_rate)

        f.close()

        print('LSTM training')
        print(('shape', X.shape, X_train.shape))
        clf.fit_lstm(X_train, y_train, X_test, y_test)
        for idx in range(rounds):
            print(('retraining lstm', idx))
            #X_train, X1_train, y_train, X_test, X1_test, y_test = split_lstm_xxy(X, X1, y, split_rate, seed=43+idx)
            #clf.retrain_lstm(X_train, y_train, X_test, y_test)

            X_train0, X1_train0, y_train0, X_test0, X1_test0, y_test0 = split_lstm_xxy(X_train, X1_train, y_train, 3./4, seed=43+idx)
            #clf.retrain_lstm(X_train0, y_train0, X_test, y_test)
            clf.retrain_lstm(X_train0, y_train0, X_test0, y_test0)

        # the test score
        #Y_test = np_utils.to_categorical(y_test).astype('float32')
        #score = clf.model_2d.evaluate(X_test, Y_test, verbose=0)
        #score = clf.model_2d.evaluate(X_test, y_test, verbose=0)
        y_test_pred = clf.predict_lstm(X_test)
        y_train_pred = clf.predict_lstm(X_train)
        print(('test', y_test_pred, y_test))


    clf.save(train, mode)
    precise = metrics.precision_score(y_test, y_test_pred)
    recall = metrics.recall_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    print(('test Precise:', precise))
    print(('test  Recall:', recall))
    print(('test      F1:', f1))

    precise = metrics.precision_score(y_train, y_train_pred)
    recall = metrics.recall_score(y_train, y_train_pred)
    f1 = metrics.f1_score(y_train, y_train_pred)
    print(('train Precise:', precise))
    print(('train Recall:', recall))
    print(('train     F1:', f1))



# train by lightgdm
def run_train_lgb(train, seq_dict, clf):
    # get the training data
    split_rate = 1. / 3

    f = open(train, 'r')
    #X, X1, X2, y = get_xxy(f, seq_dict, 3, 128)
    #X, X1, X2, y = get_xxy(f, seq_dict, 4, 64)
    X, X1, X2, y = get_xxy(f, seq_dict, global_kmr, global_len)
    X_train, X1_train, X2_train, y_train, X_test, X1_test, X2_test, y_test = split_xxy(
        X, X1, X2, y, split_rate)
    f.close()

    print(('data shape 1', X_train.shape, y_train.shape))
    a0, a1, a2, a3 = X_train.shape
    X_train = X_train.reshape(a0, a1 * a2 * a3)
    print(('data shape 2', X_train.shape, y_train.shape))

    clf.fit(X_train, y_train)
    # clf.save(train+'.lgb')

    _o = open(train + '.lgb', 'wb')
    dump(clf, _o)
    _o.close()

    a0, a1, a2, a3 = X_test.shape
    X_test = X_test.reshape(a0, a1 * a2 * a3)
    y_test_pred = clf.predict(X_test)

    precise = metrics.precision_score(y_test, y_test_pred)
    recall = metrics.recall_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    print(('Precise:', precise))
    print((' Recall:', recall))
    print(('     F1:', f1))


# run the adjacent prediction
def run_adjacent_predict(adjacent, seq_dict, model, clf, mode='2d'):
    adjacent, model = sys.argv[3: 5]
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    #print 'try loading lstm model', model
    clf.load(model, mode)

    # get the locus of genes
    f = open(adjacent, 'r')
    for i in f:
        j = i[:-1].split('\t')
        #x0, x1, x2 = get_xx_one(j, seq_dict, 3, 128, 'test')
        #x0, x1, x2 = get_xx_one(j, seq_dict, 4, 64, 'test')
        #x0, x1, x2 = get_xx_one(j, seq_dict, global_kmr, global_len, 'test')
        # print 'data shape', x0.shape, x1.shape
        if mode == '2d':
            x0, x1, x2 = get_xx_one(j, seq_dict, global_kmr, global_len, 'test')
            res = clf.predict_2d(x0)[0]
        elif mode == 'chainer':
            x0, x1, x2 = get_xx_one(j, seq_dict, global_kmr, global_len, 'test')
            #print('x0 shape', x0.shape)
            res = clf.predict_chn(x0, train=False)[0]
        else:
            x0, x1 = get_lstm_xx_one(j, seq_dict, global_kmr, global_len, 'test')
            res = clf.predict_lstm(x0)[0]

        res = res == 1 and 'True' or 'False'
        print((i[: -1] + '\t' + str(res)))

    f.close()


# run the adjacent prediction by lightgdm
def run_adjacent_predict_lgb(adjacent, seq_dict, model, clf, mode='2d'):
    adjacent, model = sys.argv[3: 5]
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    #clf.load(model, mode)
    with open(model, 'rb') as md:
        clf = load(md)

    # get the locus of genes
    f = open(adjacent, 'r')
    for i in f:
        j = i[:-1].split('\t')
        #x0, x1, x2 = get_xx_one(j, seq_dict, 3, 128, 'test')
        #x0, x1, x2 = get_xx_one(j, seq_dict, 4, 64, 'test')
        x0, x1, x2 = get_xx_one(j, seq_dict, global_kmr, global_len, 'test')
        a0, a1, a2, a3 = x0.shape
        x0 = x0.reshape(a0, a1 * a2 * a3)
        # print 'data shape', x0.shape, x1.shape
        res = clf.predict(x0)
        res = res == 1 and 'True' or 'False'
        print((i[: -1] + '\t' + str(res)))

    f.close()


# run the whole genome prediction

# generate adjacent gene pairs from the gene list
def adjacent_genes(f):
    locus_list = []
    for i in f:
        j = i[: -1].split('\t')
        if len(j) < 7:
            j.extend([0] * 7)
        locus, scaf, strand, start, end = j[: 5]
        start, end = list(map(int, [start, end]))
        locus_list.append([locus, scaf, strand, start, end])

    locus_list.sort(key=lambda x: x[1: 5])
    return locus_list


def run_genome_predict(genome, seq_dict, model, clf, mode='2d'):
    genome, model = sys.argv[3: 5]
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    clf.load(model, mode)

    # get the locus of genes
    f = open(genome, 'r')
    locus_list = adjacent_genes(f)
    f.close()
    for a, b in zip(locus_list[: -1], locus_list[1:]):
        j = a + b
        #x0, x1, x2 = get_xx_one(j, seq_dict, 3, 128, 'test')
        x0, x1, x2 = get_xx_one(j, seq_dict, global_kmr, global_len, 'test')
        if mode == '2d':
            if a[1] == b[1] and a[2] == b[2]:
                res = clf.predict_2d(x0)[0]
            else:
                res = 0
        else:
            pass

        res = res == 1 and 'True' or 'False'
        i = '\t'.join(map(str, j))
        print((i + '\t' + str(res)))


if __name__ == '__main__':
    import sys
    if len(sys.argv[1:]) < 3:
        print(('#' * 79))
        print('# To train a model:')
        print(('#' * 79))
        print('python this.py train foo.fasta foo.train.txt [mode]\n')
        print('foo.train.txt is the gene location in the format:')
        print('       locus1\tscf1\tstrand1\tstart1\tend1\tlocus2\tscf2\tstrand2\tstart2\tend2\tcat\n')

        print(('#' * 79))
        print('# To make a adjacent genes prediction')
        print(('#' * 79))
        print('python this.py adjacent foo.fasta foo.adjacent.txt foo.model [mode]\n')
        print('foo.adjacent.txt is the gene location in the format:')
        print('       locus1\tscf1\tstrand1\tstart1\tend1\tlocus2\tscf2\tstrand2\tstart2\tend2\n')

        print(('#' * 79))
        print('# To make a whole genome prediction')
        print(('#' * 79))
        print('python this.py genome foo.fasta foo.genome.txt foo.model [mode]')
        print('foo.genome.txt is the gene location in the format:')
        print('       locus1\tscf1\tstrand1\tstart1\tend1')

        print('')
        print(('#' * 79))
        print('start1/2: start of the gene in the genome, start > 0 need be adjust in the program')
        print('     cat: indicate whether  operon or not')
        print('    mode: 2d')
        raise SystemExit()

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding, BatchNormalization
    #from tensorflow.keras.layers import Input, Merge, LSTM, GRU, Bidirectional, UpSampling2D, InputLayer, CuDNNGRU
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
    #from tensorflow.keras.utils import np_utils
    from tensorflow.keras import utils as np_utils
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
    #from tensorflow.keras import objectives
    from tensorflow.keras.layers import Input, Dense, Lambda
    import numpy as np

    model, fasta = sys.argv[1: 3]

    # save the genome to an dict
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))

    if model.startswith('train'):
        train = sys.argv[3]
        try:
            mode = sys.argv[4]
        except:
            mode = '2d'

        clf = CNN(nb_epoch=global_epoch, maxlen=128, save_path=train + '_' +
                  mode + '_%d_%d' % (global_kmr, global_len) + '.hdf5')
        #clf = lgb.LGBMClassifier(n_estimators=50, max_bin=350, boosting_type='dart', learning_rate=.05)
        #clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=.15)
        #clf = SVC(C=2)
        run_train(train, seq_dict, clf, mode)
        #run_train(train, seq_dict, clf, 'lstm')
        #run_train_lgb(train, seq_dict, clf)

    elif model.startswith('predict'):
        if len(sys.argv[1:]) < 4:
            print(('#' * 79))
            print('# To make a adjacent genes prediction')
            print(('#' * 79))
            print('python this.py predict foo.fasta foo.adjacent.txt foo.model [mode]\n')
            print('foo.adjacent.txt is the gene location in the format:')
            print('       locus1\tscf1\tstrand1\tstart1\tend1\tlocus2\tscf2\tstrand2\tstart2\tend2\n')
            raise SystemExit()

        test, model = sys.argv[3: 5]
        try:
            mode = sys.argv[5]
        except:
            mode = '2d'

        clf = CNN(nb_epoch=32, maxlen=128)

        # determine the number of col
        #print('test', test)
        f = open(test, 'r')
        #print(f.readline())
        header = f.readline().split('\t')
        #header = f.next().split('\t')
        f.close()
        if header.count('+') + header.count('-') > 1:
            #print 'try loading lstm or 2d model'
            run_adjacent_predict(test, seq_dict, model, clf, mode)
            #run_adjacent_predict(test, seq_dict, model, clf, 'lstm')
            #run_adjacent_predict_lgb(test, seq_dict, model, clf, mode)

        else:
            run_genome_predict(test, seq_dict, model, clf, mode)
    else:
        pass
