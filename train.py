import torch
import argparse
from model.deeplab_model import DeeplabV3_plus
from dataset import data_load


parser = argparse.ArgumentParser(description='PyTorch DeeplabV3+')
parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
parser.add_argument("--num-classes", default=20, type=int)
parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
parser.add_argument("--device", default="cuda", help="training device")
parser.add_argument("-b", "--batch-size", default=4, type=int)
parser.add_argument("--epochs", default=30, type=int, metavar="N",
                    help="number of total epochs to train")

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch')
# Mixed precision training parameters
parser.add_argument("--amp", default=False, type=bool,
                    help="Use torch.cuda.amp for mixed precision training")
args = parser.parse_args()


if __name__ == '__main__':
    model = DeeplabV3_plus(num_classes=2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    train_data = data_load