from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet_model import MetaResnet34
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset


parser = argparse.ArgumentParser(description='Mixup Co-training for Semi-supervised Domain Adaptation')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='mico',
                    choices=['mico', 'mist'])
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./checkpoint',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--st',type=int, default=0)
parser.add_argument('--dataset', type=str, default='multi')
parser.add_argument('--num', type=int, default=3,
                    help='3-shot/1-shot')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--runs', type=int, default=999)
parser.add_argument('--eval', action='store_true', default=False)

args = parser.parse_args()


"""DomainNet-subset 7 adaptation scenarios"""
multi = [['real', 'clipart'], ['real', 'painting'],
         ['painting', 'clipart'], ['clipart','sketch'],
         ['sketch', 'painting'], ['real', 'sketch'],
         ['painting', 'real']]

if args.st != 0 and args.dataset == 'multi':
    args.source, args.target = multi[args.st-1]

print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args=args, return_idx=False)


""" net: w_f; twin: w_g """
use_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(args.seed)

if args.net == 'resnet34':
    net = MetaResnet34(num_class=len(class_list))
    twin = MetaResnet34(num_class=len(class_list))

params = []
for value in net.G.params():
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}]
params_F1 = []
for value in net.F1.params():
    if value.requires_grad:
        params_F1 += [{'params': [value], 'lr': args.multi,
                       'weight_decay': 0.0005}]

params_2 = []
for value in twin.G.params():
    if value.requires_grad:
        params_2 += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}]

params_F2 = []
for value in twin.F1.params():
    if value.requires_grad:
        params_F2 += [{'params': [value], 'lr': args.multi,
                       'weight_decay': 0.0005}]

""" record & resume path """
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

record_dir = './record/%s/mico' % args.dataset
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           'exp_net_%s_%s_to_%s_num_%s_%d' %
                           (args.net, args.source,
                            args.target, args.num, args.runs))


""" pre-train & resume """
pretrain_src_checkpoint = './pretrained_models/pretrained_src_real_to_clipart.pth.tar'.format(args.source, args.target)
pretrain_t_checkpoint = './pretrained_models/pretrained_tgt_real_to_clipart.pth.tar'.format(args.source, args.target)

if args.net_resume:

    p1 = os.path.join(args.checkpath, args.net_resume)
    p2 = os.path.join(args.checkpath, 'Twin' + args.net_resume[3:])

    net.load_state_dict(
        torch.load(p1)
    )
    twin.load_state_dict(
        torch.load(p2)
    )
else:
    net.load_state_dict(
        torch.load(pretrain_src_checkpoint)
    )
    twin.load_state_dict(
        torch.load(pretrain_t_checkpoint)
    )


lr = args.lr
net.cuda()
twin.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_tu_2 = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
im_data_tu_2 = im_data_tu_2.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_tu_2 = Variable(im_data_tu_2)
sgt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)


def train():
    net.train()
    twin.train()

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(params_F1, lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_g_2 = optim.SGD(params_2, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f_2 = optim.SGD(params_F2, lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_g_2.zero_grad()
        optimizer_f_2.zero_grad()

    param_lr_g = []
    param_lr_g_2 = []
    param_lr_f = []
    param_lr_f_2 = []

    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    for param_group in optimizer_g_2.param_groups:
        param_lr_g_2.append(param_group["lr"])
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    for param_group in optimizer_f_2.param_groups:
        param_lr_f_2.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_no_reduce = nn.CrossEntropyLoss(reduction='none').cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0

    for step in range(args.start, all_step):

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_g_2 = inv_lr_scheduler(param_lr_g_2, optimizer_g_2, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        optimizer_f_2 = inv_lr_scheduler(param_lr_f_2, optimizer_f_2, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        with torch.no_grad():

            im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
            gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])

        """stream 1"""
        zero_grad_all()
        data = im_data_t
        target = gt_labels_t
        out_1 = net(data)
        loss_1 = criterion_no_reduce(out_1, target).mean()
        zero_grad_all()

        """stream 2"""
        data = im_data_s
        target = gt_labels_s
        out_2 = twin(data)
        loss_2 = criterion_no_reduce(out_2, target).mean()
        zero_grad_all()

        """pseudo-label"""
        u_1_prob = torch.softmax(net(im_data_tu), dim=1)
        u_1_pred = u_1_prob.max(1)
        u_2_prob = torch.softmax(twin(im_data_tu), dim=1)
        u_2_pred = u_2_prob.max(1)

        u_1_mask = u_1_pred[0] >= args.th
        u_2_mask = u_2_pred[0] >= args.th

        im_u_1 = im_data_tu[u_2_mask]
        psl_u_1 = u_2_pred[1][u_2_mask]
        im_u_2 = im_data_tu[u_1_mask]
        psl_u_2 = u_1_pred[1][u_1_mask]

        """mix_up"""
        alpha = 1
        lam = np.random.beta(alpha, alpha)
        # stream 1
        if im_u_1.size(0) > 0:

            size_1 = im_u_1.size(0)
            print('stream 1: {}'.format(size_1))

            t_idx = torch.randperm(im_data_t.size(0))[0:size_1]
            mixed_x = lam * im_data_t[t_idx] + (1-lam) * im_u_1
            y_a, y_b = gt_labels_t[t_idx], psl_u_1

            out_mix = net(mixed_x)
            loss_mix_1 = lam * criterion(out_mix, y_a) + (1-lam) * criterion(out_mix, y_b)

            loss_1 += loss_mix_1
            zero_grad_all()
            loss_1.backward(retain_graph=True)
            optimizer_f.step()
            optimizer_g.step()
        else:
            zero_grad_all()
            loss_1.backward(retain_graph=True)
            optimizer_f.step()
            optimizer_g.step()

        zero_grad_all()
        # stream 2
        if im_u_2.size(0) > 0:
            size_2 = im_u_2.size(0)
            print('stream 2: {}'.format(size_2))
            s_idx = torch.randperm(im_data_s.size(0))[0:size_2]
            mixed_x = (1-lam) * im_data_s[s_idx] + lam * im_u_2
            y_a, y_b = gt_labels_s[s_idx], psl_u_2
            out_mix = twin(mixed_x)
            loss_mix_2 = (1-lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
            loss_2 += loss_mix_2
            zero_grad_all()
            loss_2.backward()
            optimizer_f_2.step()
            optimizer_g_2.step()
        else:
            zero_grad_all()
            loss_2.backward()
            optimizer_f_2.step()
            optimizer_g_2.step()
        zero_grad_all()

        log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                    'Method {}\n'.\
            format(args.source, args.target,
                   step, lr,
                   args.method)
        net.zero_grad()
        twin.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0:

            acc_test_net, acc_test_twin, acc_test = test_ensemble(target_loader_test)
            acc_val_net, acc_val_twin, acc_val = test_ensemble(target_loader_val)

            net.train()
            twin.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('test acc %f best acc test %f best acc val %f' % (acc_test,
                                                                    best_acc_test,
                                                                    best_acc))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d wf %f wg %f mico %f best mico %f best val %f \n' % (step,
                                                                                     acc_test_net,
                                                                                     acc_test_twin,
                                                                                     acc_test,
                                                                                     best_acc_test,
                                                                                     best_acc))
            net.train()
            twin.train()
            if args.save_check:
                print('saving model')
                torch.save(net.state_dict(),
                           os.path.join(args.checkpath,
                                        "Net_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(twin.state_dict(),
                           os.path.join(args.checkpath,
                                        "Twin_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


def test_ensemble(loader):
    net.eval()
    twin.eval()
    correct = 0
    correct_test_1 = 0
    correct_test_2 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            output1 = net(im_data_t)
            output2 = twin(im_data_t)

            """test 1 and 2"""
            pred_test_1 = output1.max(1)[1]
            pred_test_2 = output2.max(1)[1]

            correct_test_1 += pred_test_1.eq(gt_labels_t).sum().item()
            correct_test_2 += pred_test_2.eq(gt_labels_t).sum().item()

            """ensemble results"""
            output = torch.softmax(output1, dim=1) + torch.softmax(output2, dim=1)
            pred = output.max(1)[1]

            total += gt_labels_t.size(0)
            correct += pred.eq(gt_labels_t).sum().item()

    acc_test_1 = 100. * (float(correct_test_1)/total)
    acc_test_2 = 100. * (float(correct_test_2)/total)
    acc = 100. * (float(correct)/total)

    return acc_test_1, acc_test_2, acc


if args.eval:
    print('eval mode...')
    acc_test_net, acc_test_twin, acc_test = test_ensemble(target_loader_test)
    print('net acc: {}, twin acc: {}, mico acc: {}'.format(acc_test_net, acc_test_twin, acc_test))
else:
    train()
