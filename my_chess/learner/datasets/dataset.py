from torch.utils.data import Dataset as DatasetTorch

class Dataset(DatasetTorch):
    def getName(self):
        return self.__class__.__name__