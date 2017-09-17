import os.path
import random

from settings import TRAINING_DATA_KOEF


class DatasetPart:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def next_batch(self, batch_size):
        return random.sample(zip(self.data, self.labels), batch_size)


class Dataset:
    data_file = "../parsed/orig.txt"
    labels_file = "../parsed/norm.txt"

    train = None
    test = None

    def __init__(self):
        if os.path.isfile(self.data_file) is False:
            raise ValueError('Missed parsed corpora')
        if os.path.isfile(self.labels_file) is False:
            raise ValueError('Missed parsed corpora')

        with open(self.data_file) as f:
            data = f.readlines()

        with open(self.labels_file) as f:
            labels = f.readlines()

        data = [x.strip() for x in data]
        labels = [x.strip() for x in labels]

        data_size = len(data)
        assert data_size == len(labels), "Количество данных не совпадает с количеством эталонов!"

        training_data_size = round(data_size * TRAINING_DATA_KOEF)
        self.train = DatasetPart(data[:training_data_size], labels[:training_data_size])
        self.test = DatasetPart(labels[training_data_size:], labels[training_data_size:])
