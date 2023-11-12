import torch
from torch import nn
from einops import repeat
from torch.autograd import Variable

class PositionalEncoding( nn.Module ):
    def __init__( self, e_dim, dropout = 0.1, max_len = 800 ):
        super().__init__()
        self.dropout = nn.Dropout( p = dropout )
        pe = torch.zeros( max_len, e_dim ).float()
        position = torch.arange( 0, max_len ).unsqueeze( 1 )
        div_term = 10000.0 ** ( torch.arange( 0., e_dim, 2. ) / e_dim )
        pe[ :, 0::2 ] = torch.sin( position / div_term )
        pe[ :, 1::2 ] = torch.cos( position / div_term )
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward( self, x ):
        x = x + Variable( self.pe[:, : x.size( 1 ) ], requires_grad = False ).cuda()
        return self.dropout( x )







class TNet(nn.Module):
    def __init__(self,dim, depth, heads, dropout=0.,ttype='transformer'):
        super(TNet, self).__init__()
        self.ttype=ttype
        self.tnet=nn.Identity()
        if ttype=='transformer':
            self.tnet=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,dim_feedforward=128, nhead=heads,batch_first=True,dropout=dropout),num_layers=depth)
        elif ttype=='rnn':
            self.tnet=nn.RNN(input_size=dim,hidden_size=dim,num_layers=depth,batch_first=True,dropout=dropout)
        elif ttype=='lstm':
            self.tnet=nn.LSTM(input_size=dim,hidden_size=dim,num_layers=depth,batch_first=True,dropout=dropout)
        elif ttype=='gru':
            self.tnet = nn.GRU(input_size=dim, hidden_size=dim, num_layers=depth, batch_first=True, dropout=dropout)

    def forward(self,x):
        return self.tnet(x)

class StyleEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dropout,type):
        super(StyleEncoder, self).__init__()
        self.ttype = type
        self.to_dim = nn.Sequential(nn.Linear(2, dim),nn.GELU())
        self.s_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,dim_feedforward=128,  nhead=heads,batch_first=True,dropout=dropout),num_layers=depth)
        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, dim)+1e-5)
        self.t_net = TNet(dim, depth, heads, dropout,type)
        self.dim=dim

    def forward(self, moveinfo,seq_start_end):
        s_in = self.to_dim(moveinfo) #n,len,dim
        s_in =s_in.permute(1,0,2).contiguous()
        s_out = []
        for se in seq_start_end:
            token=nn.Parameter(torch.zeros(s_in.size(0), 1, self.dim)+1e-5,requires_grad=False).cuda()
            s_out.append(self.s_transformer(torch.cat((token, s_in[:, se[0]:se[1], :]), dim=1))[:,0:1,:])
        t_in = torch.concat(s_out, dim=1).permute(1,0,2).contiguous()
        final= self.t_net(t_in)
        if self.ttype!='transformer':
            final=final[0]
        return final


class Generator(nn.Module):
    def __init__(self, dim, depth, heads, len,noise_dim, dropout,type):
        super(Generator, self).__init__()
        self.ttype = type
        self.len=len
        self.noise_dim=noise_dim
        self.cond_to_dim = nn.Sequential(nn.Linear(2, dim),nn.GELU())
        self.noise_to_dim =nn.Sequential( nn.Linear(noise_dim+dim, dim),nn.GELU())
        self.s_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,dim_feedforward=128,  nhead=heads,batch_first=True,dropout=dropout),num_layers=depth)
        self.t_net = TNet(dim, depth, heads, dropout,type)
        self.to_pos =nn.Sequential(
            nn.Linear(dim, 2),
            nn.Tanh()
        )

    def forward(self, stylelatent, condition,part_start_end):
        cond=self.cond_to_dim(condition).view(1,condition.size(0),-1)
        s_out = []
        for se in part_start_end:
            s_out.append(self.s_transformer(cond[:, se[0]:se[1], :]).permute(1,0,2).contiguous())

        noise = torch.randn([stylelatent.size(0), stylelatent.size(1), self.noise_dim], requires_grad=False).cuda()
        style = torch.concat([stylelatent,noise],dim=-1)
        style = self.noise_to_dim(style)
        for i in range(style.size(0)):
            _style = repeat(style[i:i+1,:,:], '() n d -> b n d', b=s_out[i].size(0))
            s_out[i]=torch.concat([s_out[i],_style],dim=1)

        p=torch.concat(s_out, dim=0)
        p=self.t_net(p)
        if self.ttype!='transformer':
            p=p[0]
        out=self.to_pos(p[:,1:,:])
        return out


class Discrimiter(nn.Module):
    def __init__(self, dim, depth, heads, dropout,type):
        super(Discrimiter, self).__init__()
        self.ttype = type
        self.to_dim = nn.Sequential(nn.Linear(2, dim),nn.GELU())
        self.t_net = TNet(dim*2, depth, heads, dropout,type)
        self.s_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim*2,dim_feedforward=128, nhead=heads,batch_first=True,dropout=dropout),num_layers=depth)
        self.cond_to_dim = nn.Sequential(nn.Linear(2, dim*2),nn.GELU())
        self.all_final =  nn.Sequential(
            nn.Linear(dim*2, 1),
        )
        self.per_final = nn.Sequential(
            nn.Linear(dim*2, 1),
        )
        self.dim=dim


    def forward(self, moveinfo, stylelatent, condition,part_start_end):

        p0 = self.cond_to_dim(condition).view(condition.size(0), 1, -1)
        t_in = self.to_dim(moveinfo)
        style=[]
        for i in range(stylelatent.size(0)):
            _style = repeat(stylelatent[i:i + 1, :, :], '() n d -> b n d', b=(part_start_end[i][1]-part_start_end[i][0]))
            style.append(_style)
        style = torch.concat(style,dim=0)
        t_in = torch.concat([t_in,style],dim=-1)
        t_in = torch.concat([p0, t_in], dim=1)
        t_out = self.t_net(t_in)
        if self.ttype!='transformer':
            t_out=t_out[0]
        final_t = t_out[:, 0, :]
        if self.ttype!='transformer':
            final_t = t_out[:, -1, :]
        per_out=self.per_final(final_t)

        s_in=final_t.view(1,-1,self.dim*2)
        all_out=[]
        for se in part_start_end:
            s_out = self.s_transformer(s_in[:,se[0]:se[1],:])
            final = s_out[:, 0, :]
            all_out.append(self.all_final(final))
        all_out=torch.concat(all_out,dim=0)
        return all_out,per_out
