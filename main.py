import os
import torch
from seed import SEED
from seed_dataset import SEEDDataset
from torch.utils.data import DataLoader

from comformer import Conformer
from eegnet import EEGNet
from trainer import ClassifierTrainer

from lightning.pytorch.loggers import TensorBoardLogger

def prepare_datasets(seed_dataset, train_ratio=0.7, val_ratio=0.1, batch_size=32):
    """
    划分数据集并使用训练集的均值和标准差对验证集和测试集进行标准化。

    参数:
        seed_dataset (Dataset): 原始数据集。
        train_ratio (float): 训练集占比，默认为 0.7。
        val_ratio (float): 验证集占比，默认为 0.1。
        batch_size (int): 数据加载器的批量大小，默认为 32。

    返回:
        train_dataloader (DataLoader): 标准化后的训练集数据加载器。
        val_dataloader (DataLoader): 标准化后的验证集数据加载器。
        test_dataloader (DataLoader): 标准化后的测试集数据加载器。
    """
    # 划分数据集
    train_size = int(train_ratio * len(seed_dataset))
    val_size = int(val_ratio * len(seed_dataset))
    test_size = len(seed_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        seed_dataset, [train_size, val_size, test_size]
    )

    with torch.no_grad():

        # 提取训练集数据
        train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])  # [train_samples, channels, time]
        train_labels = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])  # [train_samples]

        # 计算训练集的均值和标准差
        train_mean = train_data.mean(dim=(0, 2), keepdim=True)  # [1, channels, 1]
        train_std = train_data.std(dim=(0, 2), keepdim=True)    # [1, channels, 1]
        train_std[train_std == 0] = 1  # 避免除以 0

        # 对训练集进行标准化
        normalized_train_data = (train_data - train_mean) / train_std

        # 对验证集进行标准化
        val_data = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])  # [val_samples, channels, time]
        val_labels = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])  # [val_samples]
        normalized_val_data = (val_data - train_mean) / train_std

        # 对测试集进行标准化
        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])  # [test_samples, channels, time]
        test_labels = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])  # [test_samples]
        normalized_test_data = (test_data - train_mean) / train_std

    # 创建新的标准化数据集
    train_dataset = SEEDDataset(normalized_train_data, train_labels)
    val_dataset = SEEDDataset(normalized_val_data, val_labels)
    test_dataset = SEEDDataset(normalized_test_data, test_labels)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    data_path = '/home/yuehao02/EEGDataSet/SEED/SEED/Preprocessed_EEG/'
    seed = SEED(data_path)
    # 加载数据并合并 session
    seed.load_data(session_merge=True)

    for subject_id in range(seed.eeg_data.shape[0]):
        subject_data = seed.eeg_data[subject_id]  # [segments, 62, 200hz]
        subject_labels = seed.labels[subject_id]

        subject_data = torch.from_numpy(subject_data)
        subject_labels = torch.from_numpy(subject_labels)

        print(f"Subject {subject_id} data shape: {subject_data.shape}")
        print(f"Subject {subject_id} labels shape: {subject_labels.shape}")

        # 创建数据集
        seed_dataset = SEEDDataset(subject_data, subject_labels)

        # 调用封装函数准备数据加载器
        train_dataloader, val_dataloader, test_dataloader = prepare_datasets(seed_dataset, 
                                                                             train_ratio=0.7, 
                                                                             val_ratio=0.1, 
                                                                             batch_size=32)

        # 定义模型
        model = Conformer(num_electrodes=62,
                          sampling_rate=200,
                          hid_channels=40,
                          depth=6,
                          heads=10,
                          dropout=0.5,
                          forward_expansion=4,
                          forward_dropout=0.5,
                          num_classes=3)
        # model = EEGNet(chunk_size=200,
        #                num_electrodes=62,
        #                dropout=0.5,
        #                kernel_1=64,
        #                kernel_2=16,
        #                F1=8,
        #                F2=16,
        #                D=2,
        #                num_classes=3)

        # 定义训练器
        trainer = ClassifierTrainer(model=model, 
                                    num_classes=3, 
                                    lr=0.001, 
                                    accelerator='gpu', 
                                    devices=1, 
                                    metrics=['accuracy', 'f1score'])
        
        
        # 训练和测试
        trainer.fit(train_loader=train_dataloader, 
                    val_loader=val_dataloader, 
                    max_epochs=800, 
                    logger=TensorBoardLogger(save_dir=os.getcwd(), version=f'subject_train_{subject_id}', name="Comformer"))
        
        trainer.test(test_loader=test_dataloader, 
                     logger=TensorBoardLogger(save_dir=os.getcwd(), version=f'subject_test_{subject_id}', name="Comformer"))
