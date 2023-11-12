import torch
from model import StyleEncoder,Generator
from config import parse_args
from loader import get_data_loader
from utils import getRealPos,cal_ade,cal_fde,chamfer_crowd_distance
from numpy import *
import numpy as np

if __name__ == '__main__':
    args=parse_args()
    E = StyleEncoder(args.dim, args.depth, args.heads,  args.dropout,args.t_type).to(args.device)
    G = Generator(args.dim, args.depth, args.heads, args.seq_len,  args.noise_dim, args.dropout,args.t_type).to(args.device)
    E.load_state_dict(torch.load(args.E_dict))
    G.load_state_dict(torch.load(args.G_dict))

    E.eval()
    G.eval()

    train_dl=get_data_loader(args,False)
    ade_all, fde_all,cd_all = [], [],[]
    for i in range(100):
        ade,fde,cd=[],[],[]
        for move, move_cond, part_move, part_cond, fake_cond, seq_start_end, part_start_end, (
                x_max, x_min, y_max, y_min) in train_dl:
            move = move.to(args.device)
            cond = move_cond.to(args.device)
            e_out = E(move, seq_start_end)
            g_out = G(e_out, cond, seq_start_end)
            real = getRealPos(move, seq_start_end, x_max, x_min, y_max, y_min)
            fake = getRealPos(g_out, seq_start_end, x_max, x_min, y_max, y_min)
            for se in seq_start_end:
                cur_ade = round(cal_ade(real[se[0]:se[1], :, :], fake[se[0]:se[1], :, :]).item(), 4)
                cur_fde = round(cal_fde(real[se[0]:se[1], :, :], fake[se[0]:se[1], :, :]).item(), 4)
                cur_cd=chamfer_crowd_distance(fake[se[0]:se[1], :, :],real[se[0]:se[1], :, :])
                ade.append(cur_ade)
                fde.append(cur_fde)
                cd.append(cur_cd)
        print('epoch:',i)
        print('ade:',round( mean(ade),4))
        ade_all.append(ade)
        print('fde:',round(  mean(fde),4))
        fde_all.append(fde)
        print('cd:', round(mean(cd), 4))
        cd_all.append(cd)
        print('ade:', ade)
        print('fde:', fde)
        print('cd:', cd)
    ade_all=np.array(ade_all)
    fde_all=np.array(fde_all)
    cd_all = np.array(cd_all)
    print('ade:', np.mean(ade_all))
    print('fde:', np.mean(fde_all))
    print('cd:', np.mean(cd_all))
    print('vade:', np.var(ade_all,axis=0).mean())
    print('vfde:', np.var(fde_all,axis=0).mean())
    print('vcd:', np.var(cd_all,axis=0).mean())