from config import parse_args
from model import StyleEncoder,Generator,Discrimiter
from loader import get_data_loader
import torch
from loss import LossCompute
from utils import saveMoveInfo,saveModel,chamfer_crowd_distance
from torch import nn
from utils import getRealPos,cal_ade,cal_fde
from numpy import *
import datetime
import dateutil

def train(args,E,G,D,w1,w2,w3):
    optimizer_E = torch.optim.RMSprop(E.parameters(), lr=args.lr)
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    train_dl=get_data_loader(args)
    lossfn=LossCompute(E,G,D,args)
    best_ade,best_fde=100,100
    for i in range(args.epoch):
        E.train()
        G.train()
        D.train()
        loss_d_t, loss_r_t, loss_f_t, loss_r_p_t, loss_f_p_t,loss_e_t,loss_cycle_t,loss_part_t,loss_g_t,loss_g_d_t,loss_g_p_t,error_g_l2_t,loss_g_move,error_s_l2,=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for move,move_cond,part_move,part_cond,fake_cond,seq_start_end,part_start_end,_ in train_dl:
            move, move_cond, part_move, part_cond, fake_cond=move.to(args.device), move_cond.to(args.device), part_move.to(args.device), part_cond.to(args.device), fake_cond.to(args.device)
            for d_it in range(args.n_critic):
                optimizer_D.zero_grad()
                loss_d, loss_r, loss_f,loss_r_p,loss_f_p = lossfn.compute_discriminator_loss(move, part_move, part_cond,seq_start_end,part_start_end)
                loss_d.backward()
                optimizer_D.step()
                # Clip weights of discriminator
                for p in D.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                if d_it+1==args.n_critic:
                    loss_d_t.append(loss_d.item())
                    loss_r_t.append(loss_r)
                    loss_f_t.append(loss_f)
                    loss_r_p_t.append(loss_r_p)
                    loss_f_p_t.append(loss_f_p)

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            loss_e,loss_cycle,loss_part=lossfn.compute_encoder_loss(move,move_cond,part_move,seq_start_end,part_start_end)
            loss_e.backward()
            optimizer_E.step()
            optimizer_G.step()
            loss_e_t.append(loss_e.item())
            loss_cycle_t.append(loss_cycle)
            loss_part_t.append(loss_part)


            optimizer_G.zero_grad()
            loss_g,loss_g_d,loss_g_p,error_g_l2,error_f_l2,error_g_move = lossfn.compute_generator_loss(move,part_cond,part_move,fake_cond,seq_start_end,part_start_end,w1,w2,w3)
            loss_g.backward()
            optimizer_G.step()
            loss_g_t.append(loss_g.item())
            loss_g_d_t.append(loss_g_d)
            loss_g_p_t.append(loss_g_p)
            error_g_l2_t.append(error_g_l2)

            error_s_l2.append(error_f_l2)
            loss_g_move.append(error_g_move)

        print('Epoch:',i+1)
        print("D_loss:", round(sum(loss_d_t)/len(loss_d_t),3),
              "D_real_loss:", round(sum(loss_r_t)/len(loss_r_t),3),
              "D_fake_loss:", round(sum(loss_f_t)/len(loss_d_t),3),
              "D_real_part_loss:", round(sum(loss_r_p_t)/len(loss_d_t),3),
              "D_fake_part_loss:", round(sum(loss_f_p_t)/len(loss_d_t),3))
        print("E_loss:", round(sum(loss_e_t)/len(loss_d_t),3),
              "E_cycle_loss:", round(sum(loss_cycle_t)/len(loss_d_t),3),
              "E_part_loss:", round(sum(loss_part_t)/len(loss_d_t),3))
        print("G_loss:",round(sum(loss_g_t)/len(loss_d_t),3),
              "G_loss_d:",round(sum(loss_g_d_t)/len(loss_d_t),3),
              "G_loss_part:",round(sum(loss_g_p_t)/len(loss_d_t),3),
              "G_loss_l2:",round(sum(error_g_l2_t)/len(loss_d_t),3) ,
              "G_loss_s_l2:",round(sum(error_s_l2)/len(loss_d_t),3),
              "G_loss_move:",round(sum(loss_g_move)/len(loss_d_t),3))
        if (i+1) % 20 == 0:
            ade,fde,cd=evaluate(args, E, G)
            if ade<best_ade and fde<best_fde:
                best_ade, best_fde=ade,fde
                if (i+1)>4000:
                    saveModel(E, G, D, 'best', args)

        if (i+1)%1000==0 or i==0:
            test(args, E, G,(i+1))
        if (i+1)%1000==0:
            saveModel(E,G,D,(i+1),args)



def evaluate(args,E,G):
    E.eval()
    G.eval()
    train_dl = get_data_loader(args)
    ade, fde,cd = [], [],[]
    for move,move_cond,part_move,part_cond,fake_cond,seq_start_end,part_start_end, (x_max, x_min, y_max, y_min) in train_dl:
        move = move.to(args.device)
        cond = move_cond.to(args.device)
        e_out = E(move,seq_start_end)
        g_out = G(e_out, cond,seq_start_end)
        real = getRealPos(move,seq_start_end, x_max, x_min, y_max, y_min)
        fake = getRealPos(g_out,seq_start_end, x_max, x_min, y_max, y_min)
        for se in seq_start_end:
            cur_ade = round(cal_ade(real[se[0]:se[1],:,:], fake[se[0]:se[1],:,:]).item(), 4)
            cur_fde = round(cal_fde(real[se[0]:se[1],:,:], fake[se[0]:se[1],:,:]).item(), 4)
            cur_cd=chamfer_crowd_distance(real[se[0]:se[1],:,:],fake[se[0]:se[1],:,:])
            ade.append(cur_ade)
            fde.append(cur_fde)
            cd.append(cur_cd)
    ade= round(mean(ade), 4)
    fde= round(mean(fde), 4)
    cd=mean(cd)
    print('ade:', ade)
    print('fde:', fde)
    print('cd:', cd)
    return ade,fde,cd

def test(args,E,G,epoch):
    E.eval()
    G.eval()
    train_dl = get_data_loader(args)
    move,move_cond,part_move,part_cond,fake_cond,seq_start_end,part_start_end,(x_max,x_min,y_max,y_min)=next(iter(train_dl))
    move=move.to(args.device)
    part_cond=part_cond.to(args.device)
    e_out = E(move,seq_start_end)
    g_out = G(e_out, part_cond,part_start_end)
    saveMoveInfo(part_move,part_start_end,x_max,x_min,y_max,y_min, args,epoch, 'real',args.t_type)
    saveMoveInfo(g_out,part_start_end,x_max,x_min,y_max,y_min,args,epoch,'fake',args.t_type)


if __name__ == '__main__':
    args=parse_args()
    E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
        args.device)
    D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    train(args, E, G, D,args.w1,args.w2,args.w3)

    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D,1,1,0)
    # #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 1, 0,0.5)
    # #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0, 1,0.5)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.4, 0.6)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.5, 0.5)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.6, 0.4)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.7, 0.3)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.8, 0.2)
    #
    # args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    # E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    # train(args, E, G, D, 0.9, 0.1)


