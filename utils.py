import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def saveMoveInfo(g_out,start_end,x_max,x_min,y_max,y_min, args,epoch,type,t_type):
    '''
    :param g_out:[person_count,seq_len,2]
    :param args:
    :return:
    '''
    timestamp = args.timestamp
    output_dir = '%s%s' % (args.out_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    move = g_out.detach().clone().cpu()
    for i in range(len(start_end)):
        move[start_end[i][0]:start_end[i][1], :, 0] = (move[start_end[i][0]:start_end[i][1], :, 0]+1.)/2. * (x_max[i]-x_min[i])+x_min[i]
        move[start_end[i][0]:start_end[i][1], :, 1] = (move[start_end[i][0]:start_end[i][1], :, 1]+1)/2. * (y_max[i]-y_min[i])+y_min[i]
        moveinfo = move[start_end[i][0]:start_end[i][1], :, :].view(start_end[i][1]-start_end[i][0], -1)
        csv = pd.DataFrame(moveinfo.numpy())
        csv.to_csv('%s/%s_%s_%s_%s.csv' % (output_dir,type,t_type,epoch, i))



def getRealPos(g_out,start_end,x_max,x_min,y_max,y_min):
    '''
    :param g_out:[person_count,seq_len,2]
    :return:
    '''
    move = g_out.detach().clone().cpu()
    for i in range(len(start_end)):
        move[start_end[i][0]:start_end[i][1], :, 0] = (move[start_end[i][0]:start_end[i][1], :, 0] + 1.) / 2. * (
                    x_max[i] - x_min[i]) + x_min[i]
        move[start_end[i][0]:start_end[i][1], :, 1] = (move[start_end[i][0]:start_end[i][1], :, 1] + 1) / 2. * (
                    y_max[i] - y_min[i]) + y_min[i]
    return move


def chamfer_distance(point_cloud1, point_cloud2):
    """
    """
    min_distances1 = torch.min(torch.sqrt(torch.sum((point_cloud1.unsqueeze(1) - point_cloud2.unsqueeze(0))**2, dim=2)), dim=1).values
    min_distances2 = torch.min(torch.sqrt(torch.sum((point_cloud2.unsqueeze(1) - point_cloud1.unsqueeze(0))**2, dim=2)), dim=1).values
    chamfer_dist = (torch.mean(min_distances1) + torch.mean(min_distances2)) / 2

    return chamfer_dist

def chamfer_crowd_distance(seq1, seq2):
    """
    """
    list = []
    for i in range(seq1.shape[1]):
        list.append(chamfer_distance(seq1[:, i, :], seq2[:, i, :]))

    return torch.mean(torch.tensor(list))

def show(move):
    move=move.permute(1,0,2)
    for i in range(move.size(0)):
        plt.scatter(move[i,:,0],move[i,:,1])
        plt.show()


def saveModel(E,G,D,epoch,args):
    timestamp = args.timestamp
    output_dir = '%s%s' % (args.model_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(E.state_dict(), '%s/E_%s_%s_%s_%s.pth' % (output_dir,args.seq_len,args.t_type,epoch,args.depth))
    torch.save(G.state_dict(), '%s/G_%s_%s_%s_%s.pth' % (output_dir, args.seq_len,args.t_type, epoch,args.depth))
    torch.save(D.state_dict(), '%s/D_%s_%s_%s_%s.pth' % (output_dir, args.seq_len,args.t_type, epoch,args.depth))

def loadModel(E,G,D,args):
    E.load_state_dict(args.E_dict)
    G.load_state_dict(args.G_dict)
    D.load_state_dict(args.D_dict)



def cal_ade(real, fake,  mode='mean'):
    loss = real-fake
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == 'mean':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss

def cal_fde(real, fake, mode='mean'):
    loss = real[:,-1,:] - fake[:,-1,:]
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.mean(loss)

def getMaxSpan(move):
    rel=move[:,1:,:]-move[:,:-1,:]
    m=(move[:,1:,:]-move[:,:-1,:])**2
    m=torch.sqrt(m.sum(dim=2))
    #torch.max(m).item(),
    return max( torch.abs(rel[:,:,0]).max().item(),torch.abs(rel[:,:,1]).max().item())
