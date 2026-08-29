import pickle, numpy as np, sys

src = sys.argv[1]   # e.g. data/mesh/SMPL_NEUTRAL.pkl
dst = sys.argv[2]

with open(src, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

for k, v in list(data.items()):
    if 'chumpy' in str(type(v)):
        data[k] = np.array(v)

with open(dst, 'wb') as f:
    pickle.dump(data, f, protocol=2)

print("done")