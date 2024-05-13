import pickle

data = pickle.load(open('results/data.p', 'rb')) # data file too big for github

for key, item in data.items():
    data[key] = data[key][::10]

pickle.dump(data, open('results/data_compressed.p', 'wb'))