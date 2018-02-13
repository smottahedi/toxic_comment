import pandas as pd
import config
import os
from os.path import join


n = len(os.listdir(join(config.DATA_PATH, 'submissions')))
dfs = None
for i, f in enumerate(os.listdir(join(config.DATA_PATH, 'submissions'))):
    import pdb; pdb.set_trace()
    if i == 0:
        dfs = pd.read_csv(join(config.DATA_PATH, 'submissions', f))
        id = dfs.iloc[:, 0]
    else:
        dfs.iloc[:, 1:] += pd.read_csv(join(config.DATA_PATH, 'submissions', f)).iloc[:, 1:]


dfs_temp = dfs.iloc[:, 1:] / n
dfs_temp.insert(0, 'id', id)
dfs_temp.to_csv(join(config.DATA_PATH, 'submit.csv'), index=False)


