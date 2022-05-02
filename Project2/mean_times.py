import numpy as np


# rej_name = "data/rejection_timetest.txt"
# rej_name2 = "data/rejection_pararellizedtimetest.txt"
# print(np.loadtxt(rej_name).mean(), np.loadtxt(rej_name2).mean())

# exit()
estimate_name = "data/estimates_time.txt"
adapt_name = "data/adapt_time.txt"
rej_name = "data/rejection_time.txt"
convergence_name = "data/convergence_time.txt"
general_name = "data/general_time.txt"

names = [estimate_name, adapt_name, rej_name, convergence_name, general_name]
process = ["estimations", "adapting", "rejection", "convergence"]
times = []

for i in range(len(names)):
    time = np.loadtxt(names[i]).mean()
    times.append(time)

print("\n")
for i in range(len(names) - 1):
    time = np.loadtxt(names[i]).mean()
    print("%20s %5.1f " % (process[i], round(time / times[-1] * 100, 1)))

print("%20s %5.1f " % ("total paralalizable", round(np.array(times[:-1]).sum() * 100, 1)))
print("\n")
# print(np.array(times[:-1]).sum())
# print("percent:", times / times[-1])

time = np.loadtxt("data/copy.txt").mean()
print(time, times[-1])