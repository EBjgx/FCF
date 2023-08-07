# -*- coding: utf-8 -*-

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from scipy.stats import beta
from torch.autograd import Variable

from dataloader import CIFAR10, CIFAR100
from resnet import ResNet34, VNet

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='SymNoise',
                    help='Type of corruption ("SymNoise" or "AsymNoise").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=180, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Resnet34', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = True

torch.manual_seed(args.seed)
device = torch.device("cuda:0" if use_cuda else "cpu")

print()
print(args)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=TransformTwice(train_transform), download=True,
            seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=TransformTwice(train_transform), download=True,
            seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_data, train_loader, train_meta_loader, test_loader


def build_model():
    model = ResNet34(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)) * (0.1 ** int(epochs >= 150)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#  Calculate/Update label noise Probabilities in Correction
def update_Probs_C(NP):
    probs = torch.softmax(NP, dim=1)
    probs = probs.data.cpu().numpy()
    probs = np.max(probs, axis=1)
    return probs

#  Calculate label Noise Probabilities in Filtering
def Noise_prob_F(confidence):
    eps = 0.01
    aph2Map0 = np.interp(confidence, (confidence.min(), confidence.max()), (eps, 1 - eps))
    aph2Map = confidence
    a1, b1 = 2, 8
    a2, b2 = 8, 2
    lmd1, lmd2 = 1 / 2, 1 / 2

    ps = np.arange(0.001, 1, 0.001)
    y1 = beta.pdf(ps, a1, b1)
    y2 = beta.pdf(ps, a2, b2)

    pars = [a1, b1, a2, b2]

    for t in range(30):
        pst1 = beta.pdf(aph2Map, a1, b1)
        pst2 = beta.pdf(aph2Map, a2, b2)

        gm1 = lmd1 * pst1 / (lmd1 * pst1 + lmd2 * pst2)
        gm2 = lmd2 * pst2 / (lmd1 * pst1 + lmd2 * pst2)

        l1 = np.sum(gm1 * aph2Map) / np.sum(gm1)
        l2 = np.sum(gm2 * aph2Map) / np.sum(gm2)

        sk1 = np.sum(gm1 * (aph2Map - l1) ** 2) / np.sum(gm1)
        sk2 = np.sum(gm2 * (aph2Map - l2) ** 2) / np.sum(gm2)

        lmd1 = np.mean(gm1)
        lmd2 = np.mean(gm2)

        a1 = l1 * (l1 * (1 - l1) / sk1 - 1)
        a2 = l2 * (l2 * (1 - l2) / sk2 - 1)

        b1 = a1 * (1 - l1) / l1
        b2 = a2 * (1 - l2) / l2

        pars = np.vstack((pars, [a1, b1, a2, b2]))

        parsA = pars[-1, :]
        parsB = pars[-2, :]

        if np.mean(np.abs((parsA - parsB) / parsB)) < 0.01:
            break

        y1 = beta.pdf(ps, a1, b1)
        y2 = beta.pdf(ps, a2, b2)

    c1 = 1
    c2 = 1

    y1 = beta.pdf(ps, parsA[0] / c1, parsA[1] / c1)
    y2 = beta.pdf(ps, parsA[2] / c2, parsA[3] / c2)

    eps = 0.01


    pt1 = beta.pdf(aph2Map, parsA[0] / c1, parsA[1] / c1)
    pt2 = beta.pdf(aph2Map, parsA[2] / c2, parsA[3] / c2)
    pstP1 = lmd1 * pt1 / (lmd1 * pt1 + lmd2 * pt2)
    pstP2 = lmd2 * pt2 / (lmd1 * pt1 + lmd2 * pt2)
    id = np.argsort(aph2Map)
    pstP1s = pstP1[id]
    pstP2s = pstP2[id]

    pNSs = pstP1s ** (1 / 3)

    return pNSs

#  Calculate the Filtering size & Threshold of noise Probability
def count_Kf_and_thN(Pi):
    # hyper-parameters
    # default settings
    N = len(Pi)  # sample size
    d = 34  # VC-dimension
    delt = 0.05  # delta prob.
    Reh = 0.1  # real empirical risk
    nr = np.mean(Pi)  # noise ratio

    # Filtering bound
    etaF = np.flip(np.cumsum(np.flip(Pi)) / np.arange(1, N + 1))

    fn = 1
    sn = round(N / 2)
    BndF = np.zeros(N)
    for k in range(N):  # number of removed samples
        # -------------------Real---------------------%
        if k < sn:  # i>=sn % 0.13*N
            etaFi = etaF[k]
            epF = np.sqrt((8 * d * np.log(2 * np.exp(1) * (N - k) / d) + 8 * np.log(4 / delt)) / (N - k))  # N=i
            BndF[k] = Reh + (1 - 2 * Reh) * etaFi + epF / fn
        else:
            BndF[k] = max(BndF)
    k2 = np.argmin(BndF)
    thN = Pi[k2]
    return k2, thN

#  Calculate/Update label noise Probabilities in Filtering
def update_Probs_F(model, trainloader):
    model.eval()
    probs_f_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            input1_var = to_var(inputs, requires_grad=False)
            outputs1 = model(input1_var)
            probs_1 = torch.softmax(outputs1, dim=1)
            probs_f_list.append(probs_1)
            probs = torch.cat(probs_f_list, dim=0)
            probs = probs.data.cpu().numpy()
            probs = np.max(probs, axis=1)
    return probs

#  Testing on 10k set
def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


#  Training model on the Corrected dataset
def train(train_loader, train_meta_loader, model, vnet, vnet1, optimizer_model, optimizer_vnet, optimizer_vnet1, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_loss_wp = 0

    global results, correct_label, target_var_oh

    # result-->shape(N, 10)
    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    correct = 0
    kc = 0
    train_meta_loader_iter = iter(train_meta_loader)
    probs_0_list = []
    probs_c_list = []
    probs_0_list1 = []
    target_var_oh_list = []
    inputs_list = [] = []
    correct_label_list = []
    probs_list = []


    for batch_idx, ((inputs, inputs_u), targets, targets_true, soft_labels, indexs) in enumerate(train_loader):
        model.train()
        input_var = to_var(inputs, requires_grad=False)
        target_var = to_var(targets, requires_grad=False).long()
        targets_true_var = to_var(targets_true, requires_grad=False).long()

        if epoch < 80:
            y_f = model(input_var)
            probs = F.softmax(y_f, dim=1)
            results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
            correct += target_var.eq(targets_true_var).sum().item()
            Loss = F.cross_entropy(y_f, target_var.long())
            optimizer_model.zero_grad()
            Loss.backward()
            optimizer_model.step()
            prec_train = accuracy(y_f.data, target_var.long().data, topk=(1,))[0]
            train_loss_wp += Loss.item()
            alpha_clean = alpha_corrupt = 0
            if (batch_idx + 1) % 100 == 0:
                print('Epoch: [%d/%d]\t'
                      'Iters: [%d/%d]\t'
                      'Loss: %.4f\t'
                      'Acc@1: %.2f\t' % (
                          epoch, args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                          (train_loss_wp / (batch_idx + 1)),
                          prec_train))
            # if (batch_idx + 1) % 200 == 0:
            #     test_acc = test(model=model, test_loader=test_loader)

        else:
            index_all = np.arange(args.batch_size)
            index_clean = np.where(np.array(targets.cpu()) == np.array(targets_true.cpu()))

            index_clean = np.array(index_clean)
            index_clean = [i for y in index_clean for i in y]

            index_corrupt = np.where(np.array(targets.cpu()) != np.array(targets_true.cpu()))
            index_corrupt = np.array(index_corrupt)
            index_corrupt = [i for y in index_corrupt for i in y]

            meta_model = build_model()
            meta_model.cuda()

            meta_model.load_state_dict(model.state_dict())
            y_f_hat = meta_model(input_var)


            z = torch.max(soft_labels, dim=1)[1].long().cuda()

            cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))

            l_lambda = vnet(cost_v.data)

            cost1 = F.cross_entropy(y_f_hat, target_var, reduce=False)
            cost_v1 = torch.reshape(cost1, (len(cost1), 1))
            l1 = torch.sum(cost_v1 * l_lambda) / len(cost_v1)

            cost2 = F.cross_entropy(y_f_hat, z, reduce=False)
            cost_v2 = torch.reshape(cost2, (len(cost2), 1))
            lambda1 = vnet1(cost_v2.data)

            current_label = torch.max(y_f_hat, dim=1)[1].cuda()
            cost3 = F.cross_entropy(y_f_hat, current_label, reduce=False)
            cost_v3 = torch.reshape(cost3, (len(cost3), 1))
            l2 = torch.sum(cost_v2 * (lambda1) * (1 - l_lambda)) / len(cost_v2) + torch.sum(
                cost_v3 * (1 - lambda1) * (1 - l_lambda)) / len(cost_v3)
            l_f_meta = l1 + l2

            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(epoch >= 100)))
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads
            try:
                input_validation, target_validation = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                input_validation, target_validation = next(train_meta_loader_iter)
            input_validation_var = to_var(input_validation, requires_grad=False)
            target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

            y_g_hat = meta_model(input_validation_var)
            l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
            prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

            optimizer_vnet.zero_grad()
            optimizer_vnet1.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
            optimizer_vnet1.step()

            y_f1 = model(input_var)
            probs = F.softmax(y_f1, dim=1)
            if epoch == 120:
                probs_list.append(probs)

            cost_w = F.cross_entropy(y_f1, target_var, reduce=False)
            cost_v21 = torch.reshape(cost_w, (len(cost_w), 1))
            prec_train = accuracy(y_f1.data, target_var.data, topk=(1,))[0]

            cost_w1 = F.cross_entropy(y_f1, z, reduce=False)
            cost_v22 = torch.reshape(cost_w1, (len(cost_w1), 1))

            cost_w2 = F.cross_entropy(y_f1, torch.max(y_f1, dim=1)[1].cuda(), reduce=False)
            cost_v23 = torch.reshape(cost_w2, (len(cost_w2), 1))
            # print(cost_v23)
            with torch.no_grad():
                w_v = vnet(cost_v21)
                w_v2 = vnet1(cost_v22)

            loss1 = torch.sum(w_v * cost_v21) / len(cost_v21)

            loss2 = torch.sum(cost_v22 * w_v2 * (1 - w_v)) / len(cost_v22) + torch.sum(
                cost_v23 * (1 - w_v2) * (1 - w_v)) / len(cost_v23)

            new_pseudolabel = (w_v2 * soft_labels.float().cuda()) + ((1 - w_v2) * probs)
            # new_pseudolabel_oh = torch.zeros(inputs.size()[0], num_classes).scatter_(1, new_pseudolabel_hard.cpu().view(-1,1), 1)
            # results[indexs.cpu().detach().numpy().tolist()] = new_pseudolabel.cpu().detach().numpy().tolist()
            target_var_oh = torch.zeros(inputs.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)
            target_var_oh_list.append(target_var_oh)
            new_label = new_pseudolabel.cuda() * (1 - w_v.cuda()) + w_v.cuda() * target_var_oh.cuda()
            results[indexs.cpu().detach().numpy().tolist()] = new_label.cpu().detach().numpy().tolist()
            correct_label = torch.max(new_label.cuda(), 1)[1]
            correct += targets_true_var.eq(correct_label).sum().item()

            kc += (target_var != correct_label).sum().item()

            Loss = loss1 + loss2

            optimizer_model.zero_grad()
            Loss.backward()
            optimizer_model.step()

            probs_0 = update_Probs_C(y_f1)
            probs_0_list.append(probs_0)
            probs_0_list1 = np.array(probs_0_list).flatten()

            probs_c = update_Probs_C(soft_labels)
            probs_c_list.append(probs_c)
            probs_c_list1 = np.array(probs_c_list).flatten()

            train_loss += Loss.item()
            meta_loss += l_g_meta.item()
            if (batch_idx + 1) % 100 == 0:
                print('Epoch: [%d/%d]\t'
                      'Iters: [%d/%d]\t'
                      'Loss: %.4f\t'
                      'Acc@1: %.2f' % (
                          (epoch), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                          (train_loss / (batch_idx + 1)), prec_train))

        if epoch == 120:
            inputs_list.append(inputs)
            correct_label_list.append(correct_label)


    if epoch >= 80:
        delta_P = [p0 - pc for p0, pc in zip(probs_0_list1, probs_c_list1)]
        delta_P = np.array(delta_P)
        delta_P_index = np.argsort(delta_P)[::-1]  # nagetive step

        prob_0_s = probs_0_list1[delta_P_index]
        prob_c_s = probs_c_list1[delta_P_index]

        N = len(train_loader.dataset)
        eta_c = np.zeros(int(N/2))
        for k in range(int(N / 2)):
            eta_c[k] = (sum(prob_c_s[:k + 1]) + sum(prob_0_s[k + 1:])) / N
        k1 = np.argmin(eta_c)
        kc = np.array(kc)
        print('\nCorrection Size: ', min(k1, kc))

        if kc > k1:
            recover_index = delta_P_index[k1:kc]
            target_N = torch.cat(target_var_oh_list, dim=0)
            contiguous_tensor = target_N.contiguous()
            contiguous_target = contiguous_tensor.cpu().detach().numpy()
            results[recover_index] = contiguous_target[recover_index]

    train_loader.dataset.label_update(results)
    return probs_list, inputs_list, correct_label_list

#  Training model on the Filtered dataset
def train_f(filtered_loader, model, optimizer_model, criterion, epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(filtered_loader):
        model.train()
        inputs, labels = data
        input_var = to_var(inputs, requires_grad=False)
        labels_var = to_var(labels, requires_grad=False).long()

        outputs = model(input_var)
        loss = criterion(outputs, labels_var)
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        prec_train = accuracy(outputs, labels_var.long(), topk=(1,))[0]
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Acc@1: %.2f\t' % (
                      epoch, args.epochs, batch_idx + 1, len(filtered_loader.dataset) / 100,
                      (running_loss / (batch_idx + 1)),
                      prec_train))


train_data, train_loader, train_meta_loader, test_loader = build_dataset()

# create model
model = build_model()
if os.path.exists('model.pkl'):
    model = torch.load('model.pkl')

vnet = VNet(1, 100, 1).cuda()
vnet1 = VNet(1, 100, 1).cuda()

if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100

optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
                                  weight_decay=1e-4)
optimizer_vnet1 = torch.optim.Adam(vnet1.params(), 1e-3,
                                   weight_decay=1e-4)


def evaluate(results, train_loader, evaluator):
    model.eval()
    correct = 0
    test_loss = 0
    evaluator.reset()
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
        for batch_idx, ((inputs, inputs_u), targets, targets_true, soft_labels, indexs) in enumerate(train_loader):
            outputs = model(inputs)
            # pred = torch.max(outputs,dim=1)[1]#.cuda()
            evaluator.add_batch(targets_true, results)
    return evaluator.confusion_matrix


def main():
    best_acc = 0
    global probs_f_save
    global probs_f, train_loader_final
    # Warm-up  &  Correction
    for epoch in range(1, 121):  # 1-120
        adjust_learning_rate(optimizer_model, epoch)
        probs_f, features, targets = train(train_loader, train_meta_loader, model, vnet, vnet1, optimizer_model, optimizer_vnet, optimizer_vnet1, epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc

        if epoch == 120:
            features = torch.cat(features, dim=0)
            targets = torch.cat(targets, dim=0)
            probs_f_save = torch.cat(probs_f, dim=0)
            probs_f_save = probs_f_save.data.cpu().numpy()
            probs_f_save = np.max(probs_f_save, axis=1)

            train_data_final = torch.utils.data.TensorDataset(features, targets)
            train_loader_final = torch.utils.data.DataLoader(
                dataset=train_data_final,
                batch_size=100,
                shuffle=True,
                num_workers=0
            )

    # Filtering
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params(), 0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(121, args.epochs):  # 121-180
        Pi = Noise_prob_F(probs_f_save)
        kf, thN = count_Kf_and_thN(Pi)
        print('Filtering Size: ', kf)

        filtered_samples = []
        for i in range(len(Pi)):
            if Pi[i] < thN:
                filtered_samples.append((features[i], targets[i]))
        filtered_data = FilteredDataset(filtered_samples)
        print("the number of filtered data", len(filtered_data))

        filtered_loader = torch.utils.data.DataLoader(dataset=filtered_data, batch_size=100, shuffle=True)

        train_f(filtered_loader, model, optimizer, criterion, epoch)
        probs_f_save = update_Probs_F(model=model, trainloader=train_loader_final)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc

    torch.save(model, 'model.pkl')
    print('best accuracy:', best_acc)

if __name__ == '__main__':
    main()
