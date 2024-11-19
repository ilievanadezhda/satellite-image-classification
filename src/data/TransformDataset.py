""" Custom Transform Dataset Class for PyTorch """

from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # get image and label
        image, label = self.base_dataset[idx]
        # apply transform
        if self.transform:
            image = self.transform(image)
        # return image and label
        return image, label
