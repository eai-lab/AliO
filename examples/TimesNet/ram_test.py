# %%
import numpy as np
from tqdm.auto import tqdm

size = 10000

data = np.empty((1000, size), dtype=np.float64)
first = np.empty((1, size), dtype=np.float64)

if not data.flags['C_CONTIGUOUS']:
    print('data is not contiguous')
    data = np.ascontiguousarray(data)

for i in tqdm(range(len(data))):
    print(i)
    data[i] = first



del data
del first