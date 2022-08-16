import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.deeplab_model import DeeplabV3_plus
from dataset import data_load, get_transform
from evaluation_index import ConfusionMatrix


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


def train(model, device, train_loader, optimizer, epoch, epoch_loss):
    model.train()
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        out = model(image)
        optimizer.zero_grad()
        loss = criterion(out, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('%4d %4d / %4d loss = %2.4f' % (epoch + 1, i, len(train_data) // args.batch_size, loss.item()))


def test(model, device, test_loader, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for index, (image, target) in enumerate(test_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


if __name__ == '__main__':
    model = DeeplabV3_plus(num_classes=2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_path = os.path.join('data', 'weizmann_horse_db', 'train')
    train_transforms = get_transform(base_size = 520, crop_size = 480, train = True)
    train_data = data_load(root_path=train_path, transforms=train_transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_path = os.path.join('data', 'weizmann_horse_db', 'val')
    val_transforms = get_transform(base_size=520, crop_size=520, train=False)
    val_data = data_load(root_path=val_path, transforms=val_transforms)
    l= val_data[0]
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.94)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        epoch_loss = 0
        confmat = test(model, device, val_loader, num_classes=2)
        val_info = str(confmat)
        print(val_info)
        train(model, device, train_loader, optimizer, epoch, epoch_loss)



