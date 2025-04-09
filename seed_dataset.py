from torch.utils.data import  Dataset

class SEEDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].float()  # 确保数据是 float32 类型
        y = self.labels[idx]
        y = y.long().squeeze()  # 确保标签是整数索引
        return x, y