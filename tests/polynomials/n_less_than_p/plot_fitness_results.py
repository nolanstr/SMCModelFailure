import numpy as np
import pickle
import matplotlib.pyplot as plt


f = open("relevant_info.pkl", "rb")
data = pickle.load(f)
f.close()

fitness = data['fitness']
sigmas = data['sigmas']
tags = data['model tags']
COLORS = [plt.cm.tab20c(i) for i in range(len(tags))]

fitness_medians = np.nanmedian(fitness, axis=2)

for i, tag in enumerate(tags):
    plt.plot(sigmas, fitness_medians[:, i], label=tag, c=COLORS[i])

plt.xlabel('standard deviation of added noise')
plt.ylabel('-NMLL')
plt.legend()
plt.tight_layout()
plt.savefig('noisy_vs_fitness', dpi=1000)
plt.show()
