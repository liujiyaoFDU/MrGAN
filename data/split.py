import os
import random
from glob import glob

class SplitFiles():
    """按行分割文件"""

    def __init__(self, dirpath, ratio = [0.8,0.2,0]):
        """初始化要分割的源文件名和分割后的文件行数"""
        self.nii_path = dirpath
        self.ratio = ratio
        self.npy_path = '/home/zhouyc/jiyaoliu/Data/liver_PMX/Liver_Fibrosis_PMX_REG_transformed5/PMX_T1_DYN_HBP/npy/'
        self.save_path = '/home/zhouyc/jiyaoliu/Projects/Liver_PMX/Reg-GAN-PMX/data'


    def get_names(self):
        '''
        获取nii数据集的ID
        '''
        files = sorted(os.listdir(self.nii_path))
        return files

    def get_slice(self,file_names):
        '''
        获取训练、测试、验证集的npy file names
        '''
        a = []
        for name in file_names:
            b = glob(f'{self.npy_path}{name}*')
            b = sorted(b,key=lambda x: int(x.split('.')[0].split('_')[-1]))
            b = [i.split('/')[-1]+'\n' for i in b]

            a.extend(b)
        return a

    def split_names(self,length):
        train,test = int(self.ratio[0] * length), int(self.ratio[2] * length)
        val = length - train - test
        return train,val,test

    def split_file(self):
        file_names = self.get_names()
        length = len(file_names)
        train,val,test = self.split_names(length)
        trainset = self.get_slice(file_names[:train])
        valset = self.get_slice(file_names[train:train+val])
        testset = self.get_slice(file_names[train+val:])

        self.write_file('train', self.save_path ,trainset)
        self.write_file('val', self.save_path ,valset)
        self.write_file('test', self.save_path ,testset)
        return 


    def write_file(self, type_, path ,lines):
        """将按行分割后的内容写入相应的分割文件中"""
        try:
            with open(f'{path}/{type_}.txt', "a") as part_file:
                part_file.writelines(lines)
        except IOError as err:
            print(err)
    


if __name__ == "__main__":
    # 1. 8:2:0划分训练、测试、验证集
    file = SplitFiles('/home/zhouyc/Data/Liver_Fibrosis/Liver_Fibrosis_PMX_REG_transformed')
    file.split_file()

    # 2. 读取划分好的数据
    # txt_path = '/home/zhouyc/jiyaoliu/Projects/liver_PMX/autoencoder/data/train.txt'
    # with open(txt_path,'r') as file:
    #     lines = file.readlines()
    #     lines = [line[:-1] for line in lines]  # 去掉换行符
    #     print(lines[:2])


