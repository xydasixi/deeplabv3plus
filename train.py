import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.deeplab_model import DeeplabV3_plus
from dataset import data_load, get_transform
from evaluation_index import ConfusionMatrix, boundary_iou
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch DeeplabV3+')
parser.add_argument("--data-path", default="image\\train_val", help="weizmann_horse_db root")
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--device", default="cuda", help="training device")
parser.add_argument("-b", "--batch-size", default=5, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--epochs", default=40, type=int, metavar="N",
                    help="number of total epochs to train")
parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch')
args = parser.parse_args()


def train(model, device, train_loader, optimizer, lr_scheduler, epoch, log):
    """
    训练函数
    """
    model.train()
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        out = model(image)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        # tensoboard记录train loss
        writer.add_scalar('train loss', loss, i + epoch * len(train_loader))

        # 打印记录、txt记录训练数据
        if i % 10 == 0:
            print('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tepoch:{}\n'.format(
                i * len(image), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item(), epoch))
        log.append('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tepoch:{}\n'.format(
            i * len(image), len(train_loader.dataset),
            100. * i / len(train_loader), loss.item(), epoch))

    #一个epoch训练完成后更新学习率
    lr_scheduler.step()


def val(model, device, val_loader, epoch, log, num_classes):
    """
    验证函数
    """
    model.eval()
    #定义混淆矩阵和boundary iou
    confmat = ConfusionMatrix(num_classes)
    b_iou = boundary_iou()
    with torch.no_grad():
        for index, (image, target) in enumerate(val_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            #更新混淆矩阵
            confmat.update(target.flatten(), output.argmax(1).flatten())
            #更新boundary iou
            b_iou.update(target, output.argmax(1))

    #计算准确率，iou
    acc_global, acc, iu = confmat.compute()
    #计算boundary iou
    biu = b_iou.compute()

    # tensoboard记录准确率，m_iou，boundary iou
    writer.add_scalar('global correct', acc_global.item() * 100, epoch)
    writer.add_scalar('mean IoU', iu.mean().item() * 100, epoch)
    writer.add_scalar('boundary IoU', biu * 100, epoch)

    # 打印记录、txt记录验证数据
    print('Test : global correct: {:.1f}\taverage row correct: {}\tIoU: {}\tmean IoU: {:.1f}\tBoundary IoU: {:.1f}\tepoch:{}\n'.format(
        acc_global.item() * 100,
        ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100,
        biu * 100,
        epoch))
    log.append('Test : global correct: {:.1f}\taverage row correct: {}\tIoU: {}\tmean IoU: {:.1f}\tBoundary IoU: {:.1f}\tepoch:{}\n'.format(
        acc_global.item() * 100,
        ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100,
        biu * 100,
        epoch))


if __name__ == '__main__':
    #定义DeeplabV3+模型,num_classes默认为2
    model = DeeplabV3_plus(num_classes=args.num_classes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义记录训练验证数据的相关项
    log = []
    writer = SummaryWriter()

    # 导入训练数据并进行预处理
    train_path = os.path.join(args.data_path,  'train')
    train_transforms = get_transform(base_size = 460, pad_size = 30, train = True)
    train_data = data_load(root_path=train_path, transforms=train_transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # 导入验证数据并进行预处理
    val_path = os.path.join(args.data_path, 'val')
    val_transforms = get_transform(base_size=460, pad_size=30, train=False)
    val_data = data_load(root_path=val_path, transforms=val_transforms)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # 优化器为Nesterov动量优化器
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    #学习率更新策略
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.94)
    #损失函数为交叉熵
    criterion = nn.CrossEntropyLoss()
    #预训练模型权重文件位置
    weights_path = 'weights/best_weights'

    #如果有预训练模型，则载入模型权重
    if glob.glob(weights_path + '/*.pth'):
        weights = torch.load(glob.glob(weights_path + '/*.pth')[-1])
        model.load_state_dict(weights['model'])
        optimizer.load_state_dict(weights['optimizer'])
        lr_scheduler.load_state_dict(weights['lr_scheduler'])
        args.start_epoch = weights['epoch'] + 1

    for epoch in range(args.start_epoch, args.epochs):
        log.append(f"Train epoch: {epoch} / {args.epochs}\n")

        # 训练及验证
        train(model, device, train_loader, optimizer, lr_scheduler, epoch, log)
        val(model, device, val_loader, epoch, log, num_classes=2)

        # 存储每个epoch训练完的模型
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, "weights/model_{}.pth".format(epoch))

    # 以txt形式储存训练及验证记录
    with open('result/train_val_result.txt', 'w+') as f:
        for i in range(len(log)):
            f.write(log[i])



