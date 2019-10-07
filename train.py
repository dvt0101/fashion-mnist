import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import model
import utils
import time
import argparse
import os
import csv
from imgaug import augmenters as iaa
from DatasetTransformer import DatasetTransformer
from ImgAugTransform import ImgAugTransform
# from tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='FashionSimpleNet', help="model")
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='MNIST', help="MNIST, or FashionMNIST")
args = parser.parse_args()

#viz
tsboard = SummaryWriter()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

# log_dir = 'logs/' + datetime.now().strftime('%B%d  %H:%M:%S')
# train_writer = SummaryWriter(os.path.join(log_dir ,'train'))
# test_writer = SummaryWriter(os.path.join(log_dir ,'test'))

# Define transforms.
train_transforms1 = transforms.Compose([
    transforms.RandomRotation(20)
])
train_transforms2 = transforms.Compose([
    transforms.RandomCrop(28, padding=4)
])
train_transforms3 = transforms.Compose([
    transforms.RandomHorizontalFlip()
])
train_transforms4 = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform = [train_transforms1, train_transforms2, train_transforms3, train_transforms4]
# transform = ImgAugTransform()
#Data augumentation

# seq = iaa.Sequential([
#     iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#     iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
# ])
# Create dataloaders. Use pin memory if cuda.

if args.data == 'FashionMNIST':
    trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=None)
    print(trainset)
    for trans in transform:
        trainset = DatasetTransformer(trainset, trans)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on FashionMNIST')
else:
    trainset = datasets.MNIST('./data-mnist', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.MNIST('./data-mnist', train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on MNIST')  


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)
        # dat = seq(images=zip(X, y))
        # X, y = unzip(dat)
        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)



if __name__ == '__main__':

    # Init network, criterion and early stopping
    net = model.__dict__[args.model]().to(device)
    print(net)
    criterion = torch.nn.CrossEntropyLoss()



    # Define optimizer
    optimizer = optim.Adam(net.parameters())

    # Train the network
    patience = args.patience
    best_train = 1e4
    best_loss = 1e4
    result = {'train loss': 0, 'train acc': 0, 'val loss': 0, 'val acc': 0}
    writeFile = open('{}/stats.csv'.format(current_dir), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(net, train_loader,
                                      criterion, optimizer)
        val_loss, val_acc = run_model(net, val_loader,
                                      criterion, optimizer, False)
        end = time.time()

        # print stats
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format(e+1, train_loss, train_acc, val_loss,
                                        val_acc, end - start)
        print(stats)

        # viz
        tsboard.add_scalar('data/train-loss',train_loss,e)
        tsboard.add_scalar('data/val-loss',val_loss,e)
        tsboard.add_scalar('data/val-accuracy',val_acc.item(),e)
        tsboard.add_scalar('data/train-accuracy',train_acc.item(),e)


        # Write to csv file
        writer.writerow([e+1, train_loss, train_acc.item(), val_loss, val_acc.item()])
        # early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            result['train acc'] = train_acc.item()
            result['train loss'] = train_loss
            result['val acc'] = val_acc.item()
            result['val loss'] = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        train_writer.add_scalar('acc', train_acc, e)
        train_writer.add_scalar('loss', train_loss, e)
        test_writer.add_scalar('acc', val_acc, e)
        test_writer.add_scalar('loss', val_loss, e)
    print(result)
    writer.writerow([result[x] for x in result])
    writeFile.close()
    tsboard.close()
    # train_writer.close()
    # test_writer.close()
        # else:
        #     patience -= 1
        #     if patience == 0:
        #         print('Run out of patience!')
        #         writeFile.close()
        #         # tsboard.close()
        #         break
