from data.batches_handler import Dataset

dataset = Dataset("parsed")

print("train data size: %d, test data size: %d" % (len(dataset.train.data), len(dataset.test.data)))
print("train labels size: %d, test labels size: %d" % (len(dataset.train.labels), len(dataset.test.labels)))
