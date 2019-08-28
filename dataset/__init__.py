import os

import torch
from torch.utils.data import Dataset


class Office(Dataset):
    def __init__(self, type, source, target):
        if type == '31':
            self.dataset_base_path = './dataset/office31_resnet50'
        elif type == 'home':
            self.dataset_base_path = './dataset/Office-Home_resnet50'

        self.data_path = os.path.join(self.dataset_base_path, source + '_' + target + '.csv')
        print(self.data_path)

        self.data_list = []
        self.label_list = []
        with open(self.data_path, 'r') as data_file:
            for line in data_file.readlines():
                line_data = line.strip().split(',')
                data = list(map(float, line_data[:-1]))
                label = int(float(line_data[-1]))
                self.data_list.append(data)
                self.label_list.append(label)

        print("load dataset: " + source + '_' + target + '.csv')

    def __getitem__(self, i):
        return torch.tensor(self.data_list[i]), torch.tensor(self.label_list[i])

    def __len__(self):
        return len(self.label_list)


class MultiDomainOffice(Dataset):
    """
    融合了多个 source_source 用来训autoencoder
    """

    def __init__(self, type, source, target):
        if type == '31':
            self.dataset_base_path = './dataset/office31_resnet50'
        elif type == 'home':
            self.dataset_base_path = './dataset/Office-Home_resnet50'

        self.domain_names = [source, target]
        self.domain_ids = [0, 1]

        # 多个domain合起来
        self.data_paths = list(map(lambda x: os.path.join(self.dataset_base_path, x + "_" + x + '.csv'), self.domain_names))

        self.data_list = []
        self.domain_list = []
        self.label_list = []
        for data_path, domain_id in zip(self.data_paths, self.domain_ids):
            with open(data_path, 'r') as source_data_file:
                for line in source_data_file.readlines():
                    line_data = line.strip().split(',')
                    data = list(map(float, line_data[:-1]))
                    label = int(float(line_data[-1]))
                    self.data_list.append(data)
                    self.domain_list.append(domain_id)
                    self.label_list.append(label)

        print("MultiDomainOffice, load dataset: " + " ".join(self.domain_names))

    def __getitem__(self, i):
        return torch.tensor(self.data_list[i]), torch.tensor(self.domain_list[i]), torch.tensor(self.label_list[i])

    def __len__(self):
        return len(self.label_list)
