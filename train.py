import datetime
import argparse
import sys
import os
sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from model.LPNet import LPNet

# 创建一个自定义的输出流类，将输出同时打印到终端和文件
class Tee:
    def __init__(self, file_name):
        self.stdout = sys.stdout  # 保存原本的标准输出
        self.file = open(file_name, "a")  # 打开日志文件用于追加
    def write(self, message):
        self.stdout.write(message)  # 输出到终端
        self.file.write(message)  # 输出到文件
    def flush(self):
        self.stdout.flush()  # 刷新终端输出
        self.file.flush()  # 刷新文件内容

# 设置日志文件路径
log_file = "training_log.txt"

# 重定向 stdout 到终端和日志文件
sys.stdout = Tee(log_file)

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--savepath', default="../LFDIS/saveWeight", type=str)
    parser.add_argument('--datapath', default="../DIS5K/DIS-TR", type=str)
    parser.parse_args()
    return parser.parse_args()

def train(Dataset, Network):
    # dataset
    args = parser()
    cfg = Dataset.Config(datapath=args.datapath, savepath=args.savepath, mode='train', batch=args.batchsize, lr=0.05, momen=0.9, decay=5e-4, epoch=48)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=2)

    # network
    net = Network(cfg)
    net.train(True)
    net.cuda()

    # parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')

    sw = SummaryWriter(cfg.savepath)
    global_step = 0
    torch.save(net.state_dict(), cfg.savepath + '/model-test')

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask, trunk, struct, depth) in enumerate(loader):  # 包括 depth 图像
            image, mask, trunk, struct, depth = image.cuda(), mask.cuda(), trunk.cuda(), struct.cuda(), depth.cuda()  # 将 depth 图像传递到 GPU
            out_trunk, out_struct, out_mask = net(image, depth)  # 传递 depth 图像

            trunk = F.interpolate(trunk, size=out_trunk.size()[2:], mode='bilinear')
            loss_t = F.binary_cross_entropy_with_logits(out_trunk, trunk)
            struct = F.interpolate(struct, size=out_struct.size()[2:], mode='bilinear')
            loss_s = F.binary_cross_entropy_with_logits(out_struct, struct)
            mask_ = F.interpolate(mask, size=out_mask.size()[2:], mode='bilinear')
            lossmask = F.binary_cross_entropy_with_logits(out_mask, mask_) + iou_loss(out_mask, mask_)
            
            loss = (loss_t + loss_s + lossmask ) / 2

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()

            optimizer.step()

            # log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_t': loss_t.item(), 'loss_s': loss_s.item(), 'lossmask': lossmask.item()}, global_step=global_step)

            # 打印并保存日志
            if step % 10 == 0:
                log_msg = '%s | step:%d/%d/%d | lr=%.6f | lossmask=%.6f|' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'], lossmask.item())
                
                # 打印日志到终端
                print(log_msg, flush=True)
         
              

        if epoch > cfg.epoch * 3 / 4:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))

if __name__ == '__main__':
    train(dataset, LPNet)


