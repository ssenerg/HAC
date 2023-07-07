import numpy as np

n = np.nan
nn = np.array([
    [1, 2, 3, n, 5, n, 7, 8],
    [5, 4, 7, 2, 6, n, 4, 4],
    [3, 2, 5, 6, 7, n, 4, 2],
    [7, 4, 6, 3, n, n, 6, 3]
])

slice = nn[[0, 1, 3]][:, [3, 4, 5]]
mm = nn[:, 5]
print(np.nanmin(mm))


print(np.isnan(nn).any())

print(np.random.rand(20, 5))