import torch
from torch import optim
from torch.utils.data import DataLoader
from net.RISTDnet_model import RISTDnet
from utils.DataLoad import DataLoader
from utils.MSELoss import MSELoss
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

global global_step
global_step = 0

vm_writer = SummaryWriter(comment="--value-MSE")


@torch.no_grad()
def val_net(net):
    global global_step
    print('begin val')
    val_dataset = DataLoader("data/val/")
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=8,
                                             shuffle=True)
    net.eval()
    val_loss = 0
    for index, (image, label,) in enumerate(val_loader):
        img = image.to(device=device, dtype=torch.float32)
        gt_bg = label[0].to(device=device, dtype=torch.float32)
        gt_tg = label[1].to(device=device, dtype=torch.float32)
        pre = net(img)
        loss = MSELoss([gt_bg, gt_tg], pre)
        val_loss += loss
    print('--val_MSE:', val_loss.item() / len(val_loader))
    vm_writer.add_scalar('value-MSE', val_loss.item() / len(val_loader), global_step=global_step)
    return val_loss / len(val_loader)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = RISTDnet()
    net.to(device=device)
    net.train()
    # 数据准备
    train_dataset = DataLoader("data/train/")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=8,
                                               shuffle=True)
    # 优化方法
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    Loss = MSELoss
    best_loss = float('inf')

    # summary
    tl_writer = SummaryWriter(comment="--train-loss")
    res_wirter = SummaryWriter(comment='--result')

    # train
    for epoch in range(1000):
        avg_loss = 0
        print('start train')
        for index, (image, label,) in enumerate(train_loader):
            optimizer.zero_grad()
            img = image.to(device=device, dtype=torch.float32)
            gt_bg = label[0].to(device=device, dtype=torch.float32)
            gt_tg = label[1].to(device=device, dtype=torch.float32)
            pre = net(img)

            cat = torch.cat([img, gt_tg, pre])
            img_cat = make_grid(cat, nrow=8,
                                padding=2, pad_value=1, normalize=True)
            res_wirter.add_image('res-tg', img_cat, global_step=global_step)

            loss = Loss([gt_bg, gt_tg], pre)
            avg_loss += loss.item()
            print('--epoch:', epoch, '--index:', index + 1, '/', len(train_loader), '--loss: ', loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1
        print('--train-avg-loss:', avg_loss / len(train_loader))
        tl_writer.add_scalar('train-loss', avg_loss / len(train_loader), global_step=epoch)
        # 验证网络训练效果并保存
        if epoch % 1 == 0:
            val_loss = val_net(net)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(),
                           'model/best_model-' + str(epoch) + '-' + str(val_loss.item()) + '.pth')
