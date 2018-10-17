import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# srcs = glob.glob('log/resnet_18_half_ep_1300w/resnet_18_half_13-52528.meta')
# srcs = [s.split('.')[0] for s in srcs]
# # print(srcs)

# srcs.append('./log/mobilenetv2_wb/mobilenetv2_38-67470')

# for src in srcs:
#     print(src)
#     # src = './log/mobilenetv2_wb2/mobilenetv2_2-5190'
#     cp = tf.train.NewCheckpointReader(src)

#     w = cp.get_tensor('AAM/weights')
#     w_n = np.linalg.norm(w, axis=0, keepdims=True)
#     w /= w_n
#     print(w.shape, w_n.mean(), w_n.var(), np.dot(w[:, 0], w[:, 1001]))

#     mean_point = np.mean(w, axis=1)
#     print(np.linalg.norm(mean_point))
#     dis = np.dot(mean_point, w)
#     print(dis.mean(), dis.var())

# print('-'*40)

with open('/home/hzl/projects/explore-1300w/tf-1300w/idx.txt') as f:
    idx2name = {}
    for l in f:
        n, idx = l.split()
        idx2name[int(idx)] = int(n)

src = './log/resnet_18_half_ep_1300w/resnet_18_half_13-52528'
cp = tf.train.NewCheckpointReader(src)

w = cp.get_tensor('AAM/weights')
w /= np.linalg.norm(w, axis=0, keepdims=True)
# print(w.shape)

batch_size = 32
for i in range(0, w.shape[1], batch_size):
    sim = np.dot(w[:, i:i+batch_size].T, w)
    sim[:, [j + i for j in range(sim.shape[0])]] = -1
    idx_max = np.argmax(sim, axis=1)
    for j, idx in enumerate(idx_max):
        if sim[j, idx] > 0.46:
            # print(idx2name[i+j], idx2name[idx], sim[j, idx])
            print(idx2name[i+j], idx2name[idx])
    # print('Processed: %d' % i)
