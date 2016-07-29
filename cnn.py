#!usr/bin/env python


#from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from time import time

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier



from collections import Counter
from Bio import SeqUtils
from math import log
from time import time


##################################################
# the motif found
box_up10 = ['TATAAT', [77, 76, 60, 61, 56, 82]]
box_up35 = ['TTGACA', [69, 79, 61, 56, 54, 54]]

#box_up10 = ['GGAATC', [77, 76, 60, 61, 56, 82]]
#box_up35 = ['GCCAGG', [69, 79, 61, 56, 54, 54]]

# find the best region that may be a candidate of a motif
def findbest(seq, motif, bg = None):
    if bg is None:
        bg = {}
    l = len(motif[0])
    #best = float('-inf')
    best = -100
    idx = -1
    for i in xrange(0, len(seq) - l + 1):
        lmer = seq[i: i + l]
        score = 0
        for a, b, c in zip(lmer, motif[0], motif[1]):
            if a == b:
                score += log(float(c) / bg.get(a, 1.))
            else:
                score += log((100. - c) / bg.get(a, 1.))
                #try:
                #       score += log((100. - c) / bg.get(a, 1.))
                #except:
                #       print c, bg.get(a, 1.)

        if score >= best:
            idx = i
            best = score

    return [seq[idx: idx + l], len(seq) - idx, best]

# convert ATCG based kmer number
#code = {'A': 1, 'a': 1, 'T': 2, 't': 2, 'G': 3, 'g': 3, 'C': 4, 'c': 4}
code = [0] * 256
flag = 0
for i in 'ATGC':
    code[ord(i.lower())] = code[ord(i)] = flag
    flag += 1
#code[ord('A')] = 1; code[ord('a')] = 1; code[ord('T')] = 2; code[ord('t')] = 2; code[ord('G')] = 3; code[ord('g')] = 3; code[ord('C')] = 4; code[ord('c')] = 4

# convert string to number
def s2n(s, code = code, scale = None):
    if scale == None:
        scale = max(code) + 1
    N = 0
    output = 0
    for i in s[::-1]:
        #output += code.get(i, 0) * scale ** N
        output += code[ord(i)] * scale ** N
        N += 1

    return output

# convert the dna sequence to kmer-position matrix.
# if length of dna < given, then add NNN in the center of the sequence.
# else if length of dna > given, then trim the center of the sequence.
def kpm(S, d = 64, k = 3, code = code, scale = None):
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

    mat = np.zeros((N, d), dtype='int8')
    #mat = [array('B', [0]) * d for elem in xrange(N)]
    for i in xrange(0, d - k + 1):
        kmer = seq[i: i + k]
        R = s2n(kmer, code = code, scale = scale)
        mat[R, i] = 1
        #mat[R][i] = 1

    return mat


###################################################
try:
    qry = sys.argv[1]
except:
    qry = 'training.txt'

f = open(qry, 'r')

header = f.next()[: -1]

seqmat = []

y = []
x = []
# save the species name to z
z = []
for i in f:
    j = i[: -1].split('\t')

    pred1, gene1, Orth1, cai1, pred2, gene2, Orth2, cai2, share_cog, operon, operon_index, dist, codon_bias, len_log, species, conserve, distn, sq1, sq2, sqi = j
    if int(dist) > 3000:
        continue

    start = -100
    #bgs = Counter(sq1 + sq2 + sqi)
    bgs = Counter(sqi[start: ])
    a, b = findbest(sqi[start: ], box_up10, bgs), findbest(sqi[start: ], box_up35, bgs)

    cog = [0] * 26
    if share_cog != 'no_share':
        for char in share_cog:
            cog[ord(char) - 65] = 1

    feat = [cai1, cai2, abs(float(cai1) - float(cai2)), Orth1 !='unknown', Orth2 != 'unknown', share_cog != 'no_share' and len(share_cog) or 0, len_log, dist, codon_bias, conserve, distn]
    if sqi[start: ]:
         #a, b = SeqUtils.GC_skew(sqi[start: ]), SeqUtils.GC(sqi[start: ])
        gc = SeqUtils.GC(sqi[start: ])
        try:
            skew = SeqUtils.GC_skew(sqi[start: ])[0]
        except:
            skew = 0.
    else:
        #a = b = 0
        gc = 0.
        skew = 0.

    # the seq matrix
    seqarr = kpm(sqi, d = 128, k = 3, scale = 4)

    #print 'seqarr shape', seqarr.shape

    feat += [gc, skew] + a[1:] + b[1: ]

    seqmat.append([seqarr])
    #feat += [gc, skew]
    #feat = [dist]
    #feat = [conserve]
    #feat = [len_log]
    #feat = [cai1, cai2]
    #feat = [abs(float(cai1) - float(cai2))]
    #feat.extend(cog)
    Ndim = len(feat)
    #print 'feature', feat, species, conserve
    if operon == 'Positive':
        #y.append([0, 1])
        y.append(1)
        #y.append('Positive')
    else:
        #y.append([1, 0])
        y.append(0)
        #y.append('Negative')
    x.append(map(float, feat))
    z.append(species)


f.close()

S_data = np.asarray(seqmat)
print 'Smat shape', S_data.shape, len(seqmat)
X = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)
x = X

Y = np_utils.to_categorical(y)


# random split
idx = np.arange(len(S_data))
np.random.seed(42) 
np.random.shuffle(idx)
N = len(idx)

start = N / 3
train_idx = idx[: start]
test_idx = idx[start: ]


################################################################################
#
# the conv2d
#
################################################################################

# input image dimensions
print 'S mat', S_data.shape
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 3
# convolution kernel size
nb_conv = 2

# traning iteration
nb_epoch = 12

batch_size = 64
nb_classes = 2

a, b, img_rows, img_cols = S_data.shape
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(b, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.35))


# build the model
model.add(Dense(64)) 
model.add(Activation('relu'))
model.add(Dropout(0.35))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(S_data[train_idx, :], Y[train_idx, :], batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(S_data[test_idx, :], Y[test_idx, :]))

#model.fit([X[train_idx, :], X[train_idx, :]], Y[train_idx, :], batch_size=batch_size, nb_epoch=nb_epoch,
#        verbose=1, validation_data=([X[test_idx, :]], Y[test_idx, :]))





S_data = S_data.astype('float32')
X = X.astype('float32')

score = model.evaluate(S_data[test_idx, :], Y[test_idx, :], verbose=0)
#score = model.evaluate([X[test_idx, :], X[train_idx, :]], Y[test_idx, :], verbose=0)


print('Test score:', score[0])
print('Test accuracy:', score[1])


Y_pred = model.predict(S_data[test_idx, :]).argmax(1)

#Y_pred = model.predict([X[test_idx, :], X[train_idx, :]]).argmax(1)

Y_true = Y[test_idx, :].argmax(1)

Y_true_cnn = Y_true
Y_pred_cnn = Y_pred

pos = 1
sn = 1. * np.sum(Y_pred[Y_true == pos] == Y_true[Y_true == pos]) / len(Y_true[Y_true == pos])

print 'SN', sn

neg = 0
sp = 1. * np.sum(Y_pred[Y_true == neg] == Y_true[Y_true == neg]) / len(Y_true[Y_true == neg])
print 'SP', sp 

f1 = 2. * sp * sn / (sp + sn)
print 'F1', f1


#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


X_train = [elem.flatten() for elem in S_data[train_idx, :]]
Y_train = Y[train_idx, :].argmax(1)

X_test = [elem.flatten() for elem in S_data[test_idx, :]]
Y_test = Y[test_idx, :].argmax(1)
Y_true = Y_test

clf = GradientBoostingClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print 'GradientBoostingClassifier'
pos = 1
sn = 1. * np.sum(Y_pred[Y_true == pos] == Y_true[Y_true == pos]) / len(Y_true[Y_true == pos])
print 'SN', sn 

neg = 0
sp = 1. * np.sum(Y_pred[Y_true == neg] == Y_true[Y_true == neg]) / len(Y_true[Y_true == neg])
print 'SP', sp

f1 = 2. * sp * sn / (sp + sn)
print 'F1', f1



print 'CNN vs GB', 1. * sum(Y_pred_cnn == Y_pred) / len(Y_pred)


y_gb_pred = Y_pred

from xgboost import XGBClassifier


X_train = [elem.flatten() for elem in S_data[train_idx, :]]
Y_train = Y[train_idx, :].argmax(1)

X_test = [elem.flatten() for elem in S_data[test_idx, :]]
Y_test = Y[test_idx, :].argmax(1)
Y_true = Y_test

clf = XGBClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print 'xgboost'
pos = 1
sn = 1. * np.sum(Y_pred[Y_true == pos] == Y_true[Y_true == pos]) / len(Y_true[Y_true == pos])
print 'SN', sn 

neg = 0
sp = 1. * np.sum(Y_pred[Y_true == neg] == Y_true[Y_true == neg]) / len(Y_true[Y_true == neg])
print 'SP', sp

f1 = 2. * sp * sn / (sp + sn)
print 'F1', f1



N = 0
T = 0
for a, b, c, d in zip(Y_pred_cnn, Y_pred, y_gb_pred, Y_true):
    if a == d or b == d or c == d:
        T += 1.

    N += 1.

print 'combined accuracy', T / N


