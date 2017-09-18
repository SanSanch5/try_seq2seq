from data.batches_handler import Dataset

dataset = Dataset("parsed")

print("train data size: %d, test data size: %d" % (len(dataset.train.data), len(dataset.test.data)))
print("train labels size: %d, test labels size: %d" % (len(dataset.train.labels), len(dataset.test.labels)))

batch_size = 100
iter_count = 10
for i in range(iter_count):
    data, labels = dataset.train.next_batch(batch_size)
    assert len(data) == len(labels) == batch_size, "Batch size is incorrect"

    batch = dataset.train.next_batch(batch_size)
    assert len(batch[0]) == len(batch[1]) == batch_size, "Batch size is incorrect"

    print("batch %d of length %d" % (i+1, len(data)))

batch_size_to_print = 10
print()
for d, l in list(zip(*dataset.train.next_batch(batch_size_to_print))):
    print(d, "->", l)

print()
for d, l in list(zip(*dataset.test.next_batch(batch_size_to_print))):
    print(d, "->", l)
