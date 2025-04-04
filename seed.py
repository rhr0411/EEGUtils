"""
This script prepares the seed data for model.
1. 读取.mat文件数据
2. 分割成1s不重叠的片段
"""

import os
import scipy.io as sio
import numpy as np

class SEED():# s n c t
    def __init__(self, data_path):
        self.data_path = data_path
        self.eeg_data = None 
        self.labels = None
    

    def load_data(self,session_merge=True):
        # 读取数据
        skip_files = ['label.mat', 'readme.txt']
        file_list = [f for f in os.listdir(self.data_path) if f not in skip_files] # ['{i}_session{j}.mat'] i: 1-15, j: 1-3
        file_list = sorted(file_list, key=lambda x: (int(x.split('_')[0]), x.split('_')[1].split('.')[0])) # 按照被试编号和日期排序

        label = self.load_mat_label()  # 加载标签数据

       # 初始化存储结构
        num_subjects = 15
        num_sessions = 3
        trial_segments = 3394  # 用于记录最大的 trial*segments 数量
        self.eeg_data = np.zeros((num_subjects, num_sessions, trial_segments, 62, 200))  # 用零填充
        self.labels = np.zeros((num_subjects, num_sessions, trial_segments, 1))  # 用零填充

        # 遍历文件并加载数据
        current_subject = -1
        session_index = -1
        for file_name in file_list:
            subject = int(file_name.split('_')[0]) - 1  # 被试编号，从 0 开始
            session_date = file_name.split('_')[1].split('.')[0]  # 日期作为 session
            # 如果是新的 subject，重置 session_index
            if subject != current_subject:
                current_subject = subject
                session_index = 0
            else:
                session_index += 1

            sample = self.load_mat_sample(file_name)  # 加载 .mat 文件数据

            trial_data, trial_label = self.process(sample, label)  # 处理数据为 [trial*segments, channel, time] 和 [trial*segments, 1]

            print(f"加载数据: subject:{subject + 1}, session_date:{session_date}, session_index:{session_index + 1}, trial_data.shape:{trial_data.shape}, trial_label.shape:{trial_label.shape}")

            # 将数据直接填充到最终的 NumPy 数组中
            self.eeg_data[subject, session_index, :trial_data.shape[0], :, :] = trial_data
            self.labels[subject, session_index, :trial_label.shape[0], :] = trial_label

        print(f"数据加载完成: eeg_data.shape={self.eeg_data.shape}, labels.shape={self.labels.shape}")

        # 如果需要合并 session
        if session_merge:
            self.eeg_data = self.eeg_data.reshape(15, -1, 62, 200)
            self.labels = self.labels.reshape(15, -1, 1)
            print(f"合并 session 后: eeg_data.shape={self.eeg_data.shape}, labels.shape={self.labels.shape}")

    def load_mat_sample(self,file_name):
        # 读取.mat文件
        sample = sio.loadmat(os.path.join(self.data_path, file_name)) # {'name_eeg{i}': array[62][200hz*scecond] } i: 1-15
        return sample
    
    def load_mat_label(self):
        label = sio.loadmat(os.path.join(self.data_path, 'label.mat'))['label'][0] # {'label': array[1][15]}
        return label

    def split_data(self, data,label):
        """
        将数据分割成1s不重叠的片段
        """
        data = np.array(data)
        split_num = int(data.shape[1] / 200)

        segmented_data = np.empty((split_num,data.shape[0], 200)) # [segments, 62, 200hz]
        segmented_label = []

        for tmp_num in range(split_num):
            segmented_data[tmp_num,:,:]=data[:,tmp_num*200:(tmp_num+1)*200]
            segmented_label.append(label)

        return segmented_data,np.array(segmented_label).reshape(-1,1) # [segments, 1]

    def process(self,sample,trial_label):
        """
        将每个sample组成数组[trials]
        """
        trial_name=[key for key in sample.keys() if '_eeg' in key] # trial_name+{i}
        trial_base_name = trial_name[0].split('_')[0] # trial_name
        trial_datas=[]
        labels=[]
        for trial_id in range(len(trial_name)):
            trial = sample[f'{trial_base_name}_eeg{trial_id+1}'] #
            label = trial_label[trial_id] # 0 1 -1
            segmented_data,segmented_label = self.split_data(trial,label) # [segments, 62,200hz] [segments,1]

            labels.append(segmented_label) 
            trial_datas.append(segmented_data)
        
        return np.concatenate(trial_datas, axis=0),np.concatenate(labels, axis=0) # [trials*segments, 62, 200hz] [trials*segments,1]

if __name__ == '__main__':
    data_path = '/home/yuehao02/EEGDataSet/SEED/SEED/Preprocessed_EEG/'
    seed = SEED(data_path)
    seed.load_data(session_merge=True)
    


        