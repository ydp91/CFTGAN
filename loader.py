import torch
import pandas as pd
import glob
from torch.utils.data import DataLoader
from torch.utils import data
import random
import numpy as np

def collate(data):
    move, move_cond, part_move, part_cond, fake_cond, x_max, x_min, y_max, y_min=zip(*data)
    _len = [len(seq) for seq in move]  # per crowd human number
    _partlen = [len(seq) for seq in part_move]  # per part crowd human number
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    cum_part_start_idx = [0] + np.cumsum(_partlen).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # start pos ,end pos
    part_start_end = [[start, end]
                     for start, end in zip(cum_part_start_idx, cum_part_start_idx[1:])]
    move=torch.concat(move,dim=0)
    move_cond=torch.concat(move_cond,dim=0)
    part_move = torch.concat(part_move, dim=0)
    part_cond = torch.concat(part_cond, dim=0)
    fake_cond = torch.concat(fake_cond, dim=0)
    return move,move_cond,part_move,part_cond,fake_cond,seq_start_end,part_start_end,(x_max, x_min, y_max, y_min)

def collate_test(data):
    move, cond, x_max, x_min, y_max, y_min=zip(*data)
    _len = [len(seq) for seq in move]  # 每组的人数
    _partlen = [len(seq) for seq in cond]  # 每组的人数
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    cum_part_start_idx = [0] + np.cumsum(_partlen).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    part_start_end = [[start, end]
                     for start, end in zip(cum_part_start_idx, cum_part_start_idx[1:])]
    move=torch.concat(move,dim=0)
    cond=torch.concat(cond,dim=0)
    return move,cond,seq_start_end,part_start_end,(x_max, x_min, y_max, y_min)

class FMDateset(data.Dataset):
    def __init__(self, path,seq_len):
        '''
        :param path:
        :param seq_len:
        '''
        super(FMDateset, self).__init__()
        self.paths = glob.glob(path + '*.csv')
        self.seq_len=seq_len
        initpos=[]
        for path in self.paths:
            moveinfo = torch.from_numpy(pd.read_csv(path).iloc[:, 1:].values).type(torch.float32).view(-1, seq_len, 2)
            x_max = moveinfo[:, :, 0].max()
            x_min = moveinfo[:, :, 0].min()
            y_max = moveinfo[:, :, 1].max()
            y_min = moveinfo[:, :, 1].min()

            moveinfo[:, :, 0] = (moveinfo[:, :, 0] - x_min) / (x_max - x_min)*2.-1.
            moveinfo[:, :, 1] = (moveinfo[:, :, 1] - y_min) / (y_max - y_min)*2.-1.
            initpos.append(moveinfo[:,0,:])
        self.initpos = torch.concat(initpos,dim=0)



    def __getitem__(self, index):
        p = self.paths[index]
        moveinfo = torch.from_numpy(pd.read_csv(p).iloc[:, 1:].values).type(torch.float32).view(-1, self.seq_len, 2)
        x_max=moveinfo[:, :, 0].max()
        x_min=moveinfo[:, :, 0].min()
        y_max = moveinfo[:, :, 1].max()
        y_min = moveinfo[:, :, 1].min()

        moveinfo[:, :, 0] = (moveinfo[:, :, 0] -x_min) / (x_max-x_min)*2.-1.
        moveinfo[:, :, 1] = (moveinfo[:, :, 1]-y_min) / (y_max-y_min)*2.-1.
        part=moveinfo.clone()
        part_num=random.randint((int)(part.size(0) * 0.8), part.size(0))
        init_num = random.sample(range(0, self.initpos.size(0)), part_num)
        init_fake= self.initpos[init_num]
        part_num  =random.sample(range(0,part.size(0)),part_num)
        part_num.sort()
        part=part[part_num]

        return moveinfo , moveinfo.clone()[:,0,:],part,part.clone()[:,0,:],init_fake.clone(),x_max,x_min,y_max,y_min

    def __len__(self):
        return len(self.paths)

class FM_Test_Dateset(data.Dataset):
    def __init__(self, path,seq_len):
        super(FM_Test_Dateset, self).__init__()
        self.style_paths = glob.glob(path+'style/' + '*.csv')
        self.cond_paths = glob.glob(path + 'condition/' + '*.csv')
        self.seq_len=seq_len

    def __getitem__(self, index):
        p = self.style_paths[index]
        c = self.cond_paths[index]
        moveinfo = torch.from_numpy(pd.read_csv(p).iloc[:, 1:].values).type(torch.float32).view(-1, self.seq_len, 2)
        x_max=moveinfo[:, :, 0].max()
        x_min=moveinfo[:, :, 0].min()
        y_max = moveinfo[:, :, 1].max()
        y_min = moveinfo[:, :, 1].min()

        moveinfo[:, :, 0] = (moveinfo[:, :, 0] - x_min) / (x_max - x_min)*2.-1.
        moveinfo[:, :, 1] = (moveinfo[:, :, 1] - y_min) / (y_max - y_min)*2.-1.

        condition = torch.from_numpy(pd.read_csv(c).iloc[:, 1:].values).type(torch.float32).view(-1, 1, 2)
        condition[:, :, 0] = (condition[:, :, 0] -x_min) / (x_max-x_min)*2.-1.
        condition[:, :, 1] = (condition[:, :, 1]-y_min) / (y_max-y_min)*2.-1.

        return moveinfo ,condition,x_max,x_min,y_max,y_min

    def __len__(self):
        return len(self.style_paths)


def get_data_loader(args,shuffle=True):
    dataset = FMDateset(args.path,args.seq_len)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,collate_fn=collate)
    return dl

def get_test_data_loader(args):
    dataset = FM_Test_Dateset(args.testpath,args.seq_len)
    dl = DataLoader(dataset, batch_size=args.batch_size,collate_fn=collate_test)
    return dl
