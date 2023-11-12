import torch
from model import StyleEncoder,Generator
from config import parse_args
from loader import get_test_data_loader
from utils import saveMoveInfo

if __name__ == '__main__':
    args=parse_args()
    E = StyleEncoder(args.dim, args.depth, args.heads, args.dropout, args.t_type).to(args.device)
    G = Generator(args.dim, args.depth, args.heads, args.seq_len, args.noise_dim, args.dropout, args.t_type).to(args.device)
    E.load_state_dict(torch.load(args.E_dict))
    G.load_state_dict(torch.load(args.G_dict))

    E.eval()
    G.eval()

    test_dataloader=get_test_data_loader(args)
    for i,( move,cond,seq_start_end,part_start_end,(x_max, x_min, y_max, y_min)) in enumerate(test_dataloader):
        move = move.to(args.device)
        cond = cond.to(args.device)
        e_out = E(move,seq_start_end)
        for j in range(20):
            g_out = G(e_out, cond,part_start_end)
            saveMoveInfo(g_out,part_start_end, x_max, x_min, y_max, y_min, args, j, 'fake',args.t_type)