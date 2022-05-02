import numpy as np
import matplotlib.pyplot as plt

general_name = "data/general_time.txt"

times_file = np.loadtxt(general_name)
M = 1000

NGroups = 100
numbers = np.arange(1, 1000, 2)
times = np.zeros(len(numbers))
size = np.array(numbers) * M

print(times_file.max())

for i in range(len(numbers)):
    for j in range(NGroups):
        times[i] = np.random.choice(times_file, numbers[i]).mean()
    times[i] = times[i] / NGroups

plt.title("Time vs Sample Size")
plt.plot(size, times, 'o')
plt.xlabel("Sample size")
plt.ylabel("Time (s)")
plt.savefig("tvsN.png")
plt.show()