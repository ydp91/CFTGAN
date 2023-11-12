import argparse
import torch
import datetime
import dateutil

def parse_args():
    parser = argparse.ArgumentParser(description='FTGAN')
    #If the amount of data is smaller, reduce the batch_size or increase the epoch to increase the training timesw
    parser.add_argument('--epoch', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--noise_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--path', dest='path', type=str,help='train data path', default='./data/15/')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='./out/')
    parser.add_argument('--t_type', dest='t_type', type=str, default='transformer')#transformer lstm gru rnn
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./model/')
    parser.add_argument('--E_dict', dest='E_dict', type=str, default='./model/best/E_15_transformer_best_2.pth')
    parser.add_argument('--G_dict', dest='G_dict', type=str, default='./model/best/G_15_transformer_best_2.pth')
    parser.add_argument('--testpath', dest='testpath', type=str, default='./test/')
    parser.add_argument('--D_dict', dest='D_dict', type=str, default='')
    parser.add_argument('--seq_len', dest='seq_len', type=int, default=15)
    parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument('--w1', type=float, default=1)
    parser.add_argument('--w2', type=float, default=1)
    parser.add_argument('--w3', type=float, default=0)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'timestamp' not in args:
        args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    return args


if __name__ == '__main__':
    print(parse_args())
    