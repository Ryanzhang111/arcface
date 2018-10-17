import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

srcs = glob.glob('log/mobilenetv2_wb2/*.index')
srcs = [s.split('.')[0] for s in srcs]
# print(srcs)

srcs.append('./log/mobilenetv2_wb/mobilenetv2_38-67470')

for src in srcs:
    print(src)
    # src = './log/mobilenetv2_wb2/mobilenetv2_2-5190'
    cp = tf.train.NewCheckpointReader(src)

    w = cp.get_tensor('AAM/weights')
    w_n = np.linalg.norm(w, axis=0, keepdims=True)
    w /= w_n
    print(w.shape, w_n.mean(), w_n.var(), np.dot(w[:, 0], w[:, 1001]))

    mean_point = np.mean(w, axis=1)
    print(np.linalg.norm(mean_point))
    dis = np.dot(mean_point, w)
    print(dis.mean(), dis.var())

print('-'*40)

src = './log/mobilenetv2_wb2/mobilenetv2_2-5190'
cp = tf.train.NewCheckpointReader(src)

w = cp.get_tensor('AAM/weights')
w /= np.linalg.norm(w, axis=0, keepdims=True)

feats = np.genfromtxt('./lfw-feats/reps.csv', delimiter=',')
print(feats.shape)
feats_n = np.linalg.norm(feats, axis=1, keepdims=True)
feats /= feats_n

f_mean = np.mean(feats, axis=0)
f_mean_n = np.linalg.norm(f_mean)
f_mean /= f_mean_n
print(f_mean_n, feats_n.mean(), feats_n.var())

ff_sim = np.dot(feats, f_mean)
print(ff_sim.mean(), ff_sim.var())

selected = np.random.choice(w.shape[1], 10, False)
w_s = w[:, selected]

selected = np.random.choice(w.shape[1], 10, False)
feats_s = feats[selected]

sim = np.dot(feats_s, w_s)
print(sim)

