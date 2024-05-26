import numpy as np

size = 10
num = 5000
k = 0.9

data = []

ip =      "0123456789"
mapping = "abcdefghij"


for i in range(num):
    data_i = ""
    data_o = ""
    idxs = np.random.randint(0, len(ip), size)
    for i in idxs:
        data_i += ip[i]
        if np.random.uniform(0, 1) <= k:
            data_o += mapping[i]
        else:
            data_o += mapping[np.random.randint(0, len(ip))]
    data += f"Q:{data_i} A:{data_o}\n"

f = open("train.txt", "x")
for d in data:
    f.write(d)
f.close()
