import torch

class LanguageModelTensorDataset(torch.utils.data.Dataset):
    """Some Information about LanguageModelDataset"""
    def __init__(self, data):
        super(LanguageModelTensorDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        x, y = x[:-1], x[1:]
        return x, y

    def __len__(self):
        return len(self.data)