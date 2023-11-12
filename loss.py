import torch

class LossCompute:
    def __init__(self,netE,netG,netD, args):
        self.netE=netE
        self.netG=netG
        self.netD=netD
        self.noise_dim=args.noise_dim
        self.device=args.device

    def compute_encoder_loss(self,real_moves,all_conditions,part_move,seq_start_end,part_start_end):
        real_style = self.netE(real_moves,seq_start_end)
        part_style = self.netE(part_move,part_start_end)
        fake_move = self.netG(real_style,all_conditions,seq_start_end)
        e_cycle_loss = self.l2_loss(fake_move,real_moves)
        e_part_loss = self.l2_loss(part_style,real_style)
        e_loss=e_cycle_loss + e_part_loss
        return e_loss ,e_cycle_loss.item(),e_part_loss.item()

    def compute_generator_loss(self,real_moves, part_conditions,part_move,fake_cond,seq_start_end,part_start_end,w1,w2,w3):
        style = self.netE(real_moves,seq_start_end).detach()
        fake_part_move = self.netG(style, part_conditions,part_start_end)
        error_g_l2=self.l2_loss(fake_part_move,part_move)
        fake_fake_move=self.netG(style,fake_cond,part_start_end)
        fake_style=self.netE(fake_fake_move,part_start_end)
        error_f_l2 = self.l2_loss(style, fake_style)

        fake_all_logits,fake_per_logits = self.netD(fake_part_move, style,part_conditions,part_start_end)
        errD_fake_all =  -torch.mean(fake_all_logits)
        errD_fake_per =  -torch.mean(fake_per_logits)

        fake_part_move2 = self.netG(style, part_conditions,part_start_end)

        error_g_move= -self.l2_loss(fake_part_move, fake_part_move2)


        return w1*(errD_fake_all+errD_fake_per)+w2*error_g_l2+error_f_l2+w3*error_g_move,errD_fake_all.item(),errD_fake_per.item(),error_g_l2.item(),error_f_l2.item(),error_g_move.item()

    def compute_discriminator_loss(self,real_moves,part_move, conditions,seq_start_end,part_start_end):
        style = self.netE(real_moves,seq_start_end).detach()
        fake_move = self.netG(style, conditions,part_start_end).detach()
        real_all_logits ,real_per_logits = self.netD(part_move,style,conditions,part_start_end)
        fake_all_logits ,fake_per_logits= self.netD(fake_move,style,conditions,part_start_end)

        errD_all_real =-torch.mean(real_all_logits)
        errD_all_fake =torch.mean(fake_all_logits)
        errD_per_real =-torch.mean(real_per_logits)
        errD_per_fake =torch.mean(fake_per_logits)
        errD = errD_all_real + errD_all_fake+errD_per_real+errD_per_fake
        return errD, errD_all_real.item(), errD_all_fake.item(),errD_per_real.item(),errD_per_fake.item()


    def l2_loss(self,pred_traj, pred_traj_gt,mode='average'):
        loss = (pred_traj_gt - pred_traj) ** 2
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'average':
            return torch.mean(loss)
        elif mode == 'raw':
            return loss.sum(dim=2).sum(dim=1)



