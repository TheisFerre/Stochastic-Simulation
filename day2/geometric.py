from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


num_samples = 10000
# Using p=0.3
p = 0.3
samples = stats.geom.rvs(p, size=num_samples)

plt.hist(samples, bins=max(samples), rwidth=0.85)
plt.title(f'Geometric p={p}')
plt.show()

bins_counts = np.bincount(samples)[1:]

pmf_arr = np.array([stats.geom.pmf(i+1, p)
                    for i in range(len(bins_counts))]) * num_samples

print(pmf_arr)
print(bins_counts)

print(stats.chisquare(bins_counts, pmf_arr))
