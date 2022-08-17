import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.deeplab_model import DeeplabV3_plus
from dataset import data_load, get_transform
from evaluation_index import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch DeeplabV3+')
parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
parser.add_argument("--num-classes", default=20, type=int)
parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
parser.add_argument("--device", default="cuda", help="training device")
parser.add_argument("-b", "--batch-size", default=5, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--epochs", default=30, type=int, metavar="N",
                    help="number of total epochs to train")

parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch')
# Mixed precision training parameters
parser.add_argument("--amp", default=False, type=bool,
                    help="Use torch.cuda.amp for mixed precision training")
args = parser.parse_args()


def train(model, device, train_loader, optimizer, lr_scheduler, epoch, log):
    model.train()
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        out = model(image)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train loss', loss, i + epoch * len(train_loader))
        if i % 10 == 0:
            print('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tepoch:{}\n'.format(
                i * len(image), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item(), epoch))
        log.append('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tepoch:{}\n'.format(
            i * len(image), len(train_loader.dataset),
            100. * i / len(train_loader), loss.item(), epoch))
    lr_scheduler.step()


def test(model, device, test_loader, epoch, log, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for index, (image, target) in enumerate(test_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

    acc_global, acc, iu = confmat.compute()
    writer.add_scalar('global correct', acc_global.item() * 100, epoch)
    writer.add_scalar('mean IoU', iu.mean().item() * 100, epoch)
    print('Test : global correct: {:.1f}\taverage row correct: {}\tIoU: {}\tmean IoU: {:.1f}\tepoch:{}\n'.format(
        acc_global.item() * 100,
        ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100,
        epoch))
    log.append('Test : global correct: {:.1f}\taverage row correct: {}\tIoU: {}\tmean IoU: {:.1f}\tepoch:{}\n'.format(
        acc_global.item() * 100,
        ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100,
        epoch))


if __name__ == '__main__':
    model = DeeplabV3_plus(num_classes=2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    log = []
    writer = SummaryWriter()

    train_path = os.path.join('data', 'weizmann_horse_db', 'train')
    train_transforms = get_transform(base_size = 520, crop_size = 480, train = True)
    train_data = data_load(root_path=train_path, transforms=train_transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_path = os.path.join('data', 'weizmann_horse_db', 'val')
    val_transforms = get_transform(base_size=520, crop_size=520, train=False)
    val_data = data_load(root_path=val_path, transforms=val_transforms)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.94)

    criterion = nn.CrossEntropyLoss()

    weights_path = 'weights/best_weights'
    if os.listdir(weights_path):
        weights_file = os.listdir('weights/best_weights')[-1]
        weights = torch.load(os.path.join(weights_path, weights_file))
        model.load_state_dict(weights['model'])
        optimizer.load_state_dict(weights['optimizer'])
        lr_scheduler.load_state_dict(weights['lr_scheduler'])
        args.start_epoch = weights['epoch'] + 1

    for epoch in range(args.start_epoch, args.epochs):
        log.append(f"Train epoch: {epoch} / {args.epochs}\n")
        train(model, device, train_loader, optimizer, lr_scheduler, epoch, log)
        test(model, device, val_loader, epoch, log, num_classes=2)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, "weights/model_{}.pth".format(epoch))

    with open('result/SNN_train.txt', 'w+') as f:
        for i in range(len(log)):
            f.write(log[i])



