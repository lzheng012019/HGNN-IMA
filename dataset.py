import numpy as np
#target node

# num_nodes = 4278
# train_ratio = 0.2
# val_ratio = 0.1
# test_ratio= 0.7

# indices = np.random.permutation(num_nodes)

# num_train = int(train_ratio * num_nodes)
# num_val = int(val_ratio * num_nodes)
# num_test = int(test_ratio * num_nodes)

# train_idx = indices[:num_train]
# val_idx = indices[num_train:num_train + num_val]
# test_idx = indices[num_train + num_val:]

# np.savez('imdb/train_val_test_idx_imdb_20.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
num_nodes = 6157
train_ratio = 0.7
val_ratio = 0.1
test_ratio= 0.2

indices = np.random.permutation(num_nodes)

num_train = int(train_ratio * num_nodes)
num_val = int(val_ratio * num_nodes)
num_test = int(test_ratio * num_nodes)

train_idx = indices[:num_train]
val_idx = indices[num_train:num_train + num_val]
test_idx = indices[num_train + num_val:]

np.savez('amazon/train_val_test_idx_amazon_70.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
# num_nodes = 1887
# train_ratio = 0.2
# val_ratio = 0.1
# test_ratio= 0.7

# indices = np.random.permutation(num_nodes)

# num_train = int(train_ratio * num_nodes)
# num_val = int(val_ratio * num_nodes)
# num_test = int(test_ratio * num_nodes)

# train_idx = indices[:num_train]
# val_idx = indices[num_train:num_train + num_val]
# test_idx = indices[num_train + num_val:]

# np.savez('douban/train_val_test_idx_douban_20.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)