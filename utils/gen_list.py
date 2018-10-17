from pathlib import Path
import random

src = [Path('/mnt/ssd_raid0/hzl/msra_align'), Path('/mnt/ssd_raid0/hzl/celebrity_align')] 

peoples = []
for s in src:
    peoples.extend(s.iterdir())

peoples = sorted(peoples, key=lambda p: str(p))
p2id = {}
for i, p in enumerate(peoples):
    imgs = list(p.iterdir())
    if len(imgs) < 3:
        print(p)
        continue

    n = '{}/{}'.format(p.parent.name, p.name)
    p2id[n] = i

with open('id.txt', 'w') as f:
    ps = sorted(p2id, key=lambda k:p2id[k])
    for p in ps:
        f.write('{} {:d}\n'.format(p, p2id[p]))

train_file = open('train.txt', 'w')
val_file = open('val.txt', 'w')

for p in peoples:
    imgs = list(p.iterdir())
    if len(imgs) < 3:
        continue

    n = '{}/{}'.format(p.parent.name, p.name)
    idx = p2id[n]

    random.shuffle(imgs)
    # select val
    if len(imgs) > 5 and random.randint(0, 3) == 0:
        val_file.write('{} {:d}\n'.format(str(imgs[0]), idx))
        imgs = imgs[1:]

    for img in imgs:
        train_file.write('{} {:d}\n'.format(str(img), idx))

train_file.close()
val_file.close()
