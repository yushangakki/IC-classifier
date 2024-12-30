from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

from data_release_final import MCB
from model_release_final import DGCNN, GeometricBackbone
from utils_release_final import calculate_smoothed_loss, IOStream


def init_directories(args):
    """Create necessary directories for checkpoints and backup files."""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f'checkpoints/{args.exp_name}', exist_ok=True)
    os.makedirs(f'checkpoints/{args.exp_name}/models', exist_ok=True)

    # Backup source files
    base_path = f'checkpoints/{args.exp_name}'
    files_to_backup = {
        'main.py': 'main.py.backup',
        'model_new.py': 'model.py.backup',
        'util.py': 'util.py.backup',
        'data.py': 'data.py.backup'
    }
    
    for source, target in files_to_backup.items():
        os.system(f'cp {source} {base_path}/{target}')


def train(args, io):
    """Train the model."""
    # Data loading
    train_loader = DataLoader(
        MCB(partition='train', num_points=args.num_points),
        num_workers=48,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        MCB(partition='test', num_points=args.num_points),
        num_workers=48,
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False
    )

    # Device configuration
    device = torch.device("cuda" if args.cuda else "cpu")

    # Model initialization
    if args.model == 'gb':
        model = GeometricBackbone(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise ValueError(f"Model {args.model} not implemented")

    print(str(model))
    model = DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs!")

    # Optimizer setup
    if args.use_sgd:
        print("Using SGD optimizer")
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr * 100,
            momentum=args.momentum,
            weight_decay=1e-4
        )
    else:
        print("Using Adam optimizer")
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    criterion = calculate_smoothed_loss
    best_test_acc = 0

    # Training loop
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0
        model.train()
        train_pred = []
        train_true = []

        # Train phase
        for data, label in train_loader:
            data = data.to(device).permute(0, 2, 1)
            label = label.to(device).squeeze()
            batch_size = data.size()[0]

            optimizer.zero_grad()
            
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        # Calculate training metrics
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log training results
        train_str = f'Train {epoch}, loss: {train_loss/count:.6f}, train acc: {train_acc:.6f}, train avg acc: {avg_per_class_acc:.6f}'
        train_stat_str = f'train acc: {train_acc:.6f}, avg acc: {avg_per_class_acc:.6f}, test {epoch}'
        trstat.cprint(train_stat_str)
        io.cprint(train_str)

        # Test phase
        test_loss = 0.0
        count = 0
        model.eval()
        test_pred = []
        test_true = []

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device).permute(0, 2, 1)
                label = label.to(device).squeeze()
                batch_size = data.size()[0]

                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]

                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.cpu().numpy())

        # Calculate test metrics
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        # Log test results
        test_str = f'Test {epoch}, loss: {test_loss/count:.6f}, test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}'
        test_stat_str = f'test acc: {test_acc:.6f}, avg acc: {avg_per_class_acc:.6f}, test {epoch}'
        io.cprint(test_str)
        tsstat.cprint(test_stat_str)

        # Save models
        model_path = f'checkpoints/{args.exp_name}/models/model.t7'
        torch.save(model.state_dict(), model_path)
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_path)


def test(args, io):
    """Evaluate the model."""
    test_loader = DataLoader(
        MCB(partition='test', num_points=args.num_points),
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False
    )

    device = torch.device("cuda" if args.cuda else "cpu")
    model = GeometricBackbone(args).to(device)
    model = DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_true = []
    test_pred = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device).permute(0, 2, 1)
            label = label.to(device).squeeze()
            
            logits = model(data)
            preds = logits.max(dim=1)[1]
            
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    
    io.cprint(f'Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                      help='Name of the experiment')
    parser.add_argument('--model', type=str, default='gb', metavar='N',
                      choices=['gb', 'dgcnn'],
                      help='Model to use, [gb, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                      choices=['modelnet40'])
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                      help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                      help='Size of batch')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                      help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=True,
                      help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                      help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                      help='SGD momentum (default: 0.9)')
    
    # Model parameters
    parser.add_argument('--num_points', type=int, default=1024,
                      help='num of points to use')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                      help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                      help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                      help='Num of nearest neighbors to use')
    
    # Other settings
    parser.add_argument('--no_cuda', type=bool, default=False,
                      help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                      help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                      help='Pretrained model path')
    
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    args = parse_args()
    init_directories(args)
    
    # Initialize logging
    io = IOStream(f'checkpoints/{args.exp_name}/run.log')
    io.cprint(str(args))
    trstat = IOStream(f'checkpoints/{args.exp_name}/train.log')  # Train stat
    tsstat = IOStream(f'checkpoints/{args.exp_name}/test.log')   # Test stat
    
    # Setup CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    if args.cuda:
        io.cprint(f'Using GPU: {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # Run training or evaluation
    if not args.eval:
        train(args, io)
    else:
        test(args, io)