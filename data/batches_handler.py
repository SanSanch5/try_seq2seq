import os.path
import random

from settings import TRAINING_DATA_KOEF


class DatasetPart:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def next_batch(self, batch_size):
        common_list = list(zip(self.data, self.labels))
        sample = random.sample(common_list, batch_size)
        zipped_sample = list(zip(*sample))
        return zipped_sample


class Dataset:
    def __init__(self, path_to_parsed_files):
        self.path_to_parsed_files = path_to_parsed_files
        data_file = os.path.join(path_to_parsed_files, "orig.txt")
        labels_file = os.path.join(path_to_parsed_files, "norm.txt")

        if os.path.isfile(os.path.abspath(data_file)) is False:
            raise ValueError('Missed parsed corpora')
        if os.path.isfile(os.path.abspath(labels_file)) is False:
            raise ValueError('Missed parsed corpora')

        with open(data_file) as f:
            data = f.readlines()

        with open(labels_file) as f:
            labels = f.readlines()

        data = [x.strip() for x in data]
        labels = [x.strip() for x in labels]

        data_size = len(data)
        assert data_size == len(labels), "Количество данных не совпадает с количеством эталонов!"

        training_data_size = round(data_size * TRAINING_DATA_KOEF)
        self.train = DatasetPart(data[:training_data_size], labels[:training_data_size])
        self.test = DatasetPart(data[training_data_size:], labels[training_data_size:])

    def save_train(self):
        train_data_file = os.path.join(self.path_to_parsed_files, "train.data.txt")
        train_labels_file = os.path.join(self.path_to_parsed_files, "train.labels.txt")

        with open(train_data_file, "w") as f:
            for line in self.train.data:
                print(line, file=f)

        with open(train_labels_file, "w") as f:
            for line in self.train.labels:
                print(line, file=f)

    def save_test(self):
        test_data_file = os.path.join(self.path_to_parsed_files, "test.data.txt")
        test_labels_file = os.path.join(self.path_to_parsed_files, "test.labels.txt")

        with open(test_data_file, "w") as f:
            for line in self.test.data:
                print(line, file=f)

        with open(test_labels_file, "w") as f:
            for line in self.test.labels:
                print(line, file=f)

    def save_all(self):
        self.save_train()
        self.save_test()
