import os
import sys
import cv2
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TrainValDataset
from model import MDMTN 

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = MDMTN().cuda()
        self.crit = MSELoss().cuda()

        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=settings.lr)
        self.sche = MultiStepLR(self.opt, milestones=[15000, 17500], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        O, L = batch['O'].cuda(), batch['L'].cuda()
        O, L = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        P = self.net(O)
      
        w=torch.tensor(0.1).cuda()
        loss_list2 = self.crit(P[:,2,:,:], L[:,2,:,:])
        loss_list1 = w*self.crit(P[:,1,:,:], L[:,1,:,:])
        loss_list0 = self.crit(P[:,0,:,:], L[:,0,:,:])
    
        loss_list=[]
        loss_list.append(loss_list0)
        loss_list.append(loss_list1)
        loss_list.append(loss_list2)
        if name == 'train':
            self.net.zero_grad()
            sum(loss_list).backward()
            self.opt.step()

        losses = {
            'loss_s': loss_list0[0],'loss_a': loss_list1[0],'loss_t': loss_list2[0]
        }
       
        self.write(name, losses)
        s=P[:,0,:,:]
        s=s.unsqueeze(1)
        s = s.repeat(1, 3,1, 1)
        a=P[:,1,:,:]
        a=a.unsqueeze(1)
        a = a.repeat(1, 3,1, 1)
        t=P[:,2,:,:]
        t=t.unsqueeze(1)
        t = t.repeat(1, 3,1, 1)
        prev=(O-(s+a))/t+(s+a)
        return prev

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 5 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    
                    s=pred[idx][0,:,:]
                    s = s.repeat(3,1, 1)
                   
                    a=pred[idx][1,:,:]
                    a = a.repeat(3,1, 1)
                    t=pred[idx][2,:,:]
                    t = t.repeat(3,1, 1)

                    ss=label[idx][0,:,:]
                    ss = ss.repeat(3,1, 1)
                    aa=label[idx][1,:,:]
                    aa = aa.repeat(3,1, 1)
                    tt=label[idx][2,:,:]
                    tt = tt.repeat(3,1, 1)
                    
                    tmp_list = [data[idx], pred[idx], ss, aa, tt]
                    for k in range(5):
                        col = (j * 5 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
            
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train')
    sess.tensorboard('val')

    dt_train = sess.get_dataloader('train62')
    dt_val = sess.get_dataloader('val4')

    while sess.step < 20000:
        sess.sche.step()
        sess.net.train()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train62')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)

        if sess.step % 4 == 0:
            sess.net.eval()
            try:
                batch_v = next(dt_val)
            except StopIteration:
                dt_val = sess.get_dataloader('val4')
                batch_v = next(dt_val)
            pred_v = sess.inf_batch('val', batch_v)

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
            if sess.step % 4 == 0:
                sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
            logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model)

