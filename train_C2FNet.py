import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.SemiModel import SemiModel
import time
start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def update_ema_variables(model, ema_model, alpha):  #alpha是啥意思
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def train1(train_loader, val_loader, Eva_train,Eva_train2, Eva_val,Eva_val2,
           data_name, save_path, net,ema_net, criterion,semicriterion, optimizer,use_ema, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)
    ema_net.train(True)

    length = 0
    st = time.time()
    loss_semi=torch.zeros(1)
    with tqdm(total=len(train_loader), desc=f'Eps {epoch}/{num_epoches}', unit='img') as pbar:
        for i, (A, B, mask,with_label) in enumerate(train_loader): #with_label是？
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            with_label=with_label.cuda()

            optimizer.zero_grad()
            if use_ema is False:
                """
                如果不用ema半监督，则只对有标签的patch进行学习（with_label=True）
                """
                if with_label.any():
                    preds = net(A[with_label], B[with_label])
                    loss = criterion(preds[0], Y[with_label]) + criterion(preds[1], Y[with_label])
                    Y=Y[with_label]
                else:
                    """
                    整个batch都是没有label的，只能跳过（with_label.any()=False）
                    """
                    continue
            else:
                """
                    ema半监督，第一部分的loss是有标签的patch和预测值进行反向传播（同上）
                """
                preds = net(A,B)
                if with_label.any():
                    loss = criterion(preds[0][with_label], Y[with_label])  + criterion(preds[1][with_label], Y[with_label])
                else:
                    loss=0


            if use_ema is True:
                """
                    ema半监督，第二部分的loss是无标签的patch，用teacher的预测结果对student的预测值进行反向传播
                """
                with torch.no_grad():
                    z1 = A[~with_label]
                    z2 = B[~with_label]
                    pseudo_attn,pseudo_preds =  ema_net(z1, z2) #？分别是两个输出？attention_map是中间层知识,prediction是预测结果

                    # pseudo_attn,pseudo_preds =  ema_net(A[~with_label], B[~with_label]) #？分别是两个输出？attention_map是中间层知识,prediction是预测结果
                    pseudo_attn,pseudo_preds =  torch.sigmoid(pseudo_attn).detach(),torch.sigmoid(pseudo_preds).detach()
                loss_semi = semicriterion(preds[0][~with_label], pseudo_attn) + semicriterion(preds[1][~with_label], pseudo_preds)  #测试这里的效果，如果有用则方便讲故事
                # loss_semi =semicriterion(preds[1][~with_label],pseudo_preds) #test!!!!!!!!! loss_semi2
                loss=loss+0.2*loss_semi  #全监督损失+半监督损失，半监督系数默认为0.2，测试0.3，0.4，0.5！！
                Eva_train2.add_batch(mask[~with_label].cpu().numpy().astype(int), (preds[1][~with_label]>0).cpu().numpy().astype(int)) #~相反
            # ---- loss function ----

            loss.backward()
            optimizer.step()

            """
                ema更新teacher网络参数，teacher新参数=0.99*teacher旧参数+（1-0.99）*student参数
            """
            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha=0.99)  #0.9，0.995，0.999

            # scheduler.step()
            epoch_loss += loss.item()

            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_train.add_batch(target, pred)
            pbar.set_postfix(**{'LAll': loss.item(),'LSemi': loss_semi.item()}) #？
            pbar.update(1)
            length += 1


    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1], Eva_train2.Recall()[1], Eva_train2.F1()[1]))

    if use_ema is True:
        print(
            'Epoch [%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epoches, \
                IoU, Pre, Recall, F1))
    print("Strat validing!")


    net.train(False)
    net.eval()
    ema_net.train(False)
    ema_net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)

            preds_ema = ema_net(A, B)[1]
            Eva_val2.add_batch(target, (preds_ema>0).cpu().numpy().astype(int))
            length += 1
            """
                这里到底是存net的参数还是ema_net的参数，都可以，看哪个精度高
            """
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))

    print('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (Eva_val2.Intersection_over_Union()[1], Eva_val2.Precision()[1], Eva_val2.Recall()[1], Eva_val2.F1()[1]))
    new_iou = IoU[1]    #存教师模型？
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        # torch.save(net.state_dict(), save_path + '_best_student_iou.pth')
        # torch.save(ema_net.state_dict(), save_path + '_best_teacher_iou.pth') #当student的精度最高的时候，同时存teacher的精度，然后用teacher的精度进行测试
        print('best_epoch', epoch)
        student_dir = save_path + '_train1_' + '_best_student_iou.pth'
        # 1. 先建立一个字典，保存三个参数：
        student_state = {'best_student_net ': net.state_dict(),
                 'optimizer ': optimizer.state_dict(),
                 ' epoch': epoch}
        # 2.调用torch.save():其中dir表示保存文件的绝对路径+保存文件名，如'/home/qinying/Desktop/modelpara.pth'
        torch.save(student_state, student_dir)
        torch.save(ema_net.state_dict(), save_path + '_train1_' + '_best_teacher_iou.pth') #当student的精度最高的时候，同时存teacher的精度，然后用teacher的精度进行测试
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()



if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number') #修改这里！！！
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size') #修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=1, help='Proportion of the labeled images')#修改这里！！！
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')  #修改这里！！！
    parser.add_argument('--data_name', type=str, default='WHU', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='SemiModel_noema04',
                        help='the test rgb images root')
    # parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/WHU-5/')  # 半监督的模型保存路径！！
    parser.add_argument('--save_path', type=str, default='./output/C2FNet/WHU/')  # 全监督的模型保存路径！！

    opt = parser.parse_args()
    print('labeled ration=1,Ablation现在半监督损失函数系数为:0.2!')

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/data/chengxi.han/data/Building change detection dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/Building change detection dataset256/val/'
        # opt.train_root = '/data/chengxi.han/data/WHU-CD-256-Semi/train/'
        # opt.val_root = '/data/chengxi.han/data/WHU-CD-256-Semi/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
        opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    elif opt.data_name == 'DSIFN':
        opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
        opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    elif opt.data_name == 'SYSU':
        opt.train_root = '/data/chengxi.han/data/SYSU-CD/train/'
        opt.val_root = '/data/chengxi.han/data/SYSU-CD/val/'
    elif opt.data_name == 'S2Looking':
        opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
        opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'
    elif opt.data_name == 'GoogleGZ':
        opt.train_root = '/data/chengxi.han/data/Google_GZ_CD256/train/'
        opt.val_root = '/data/chengxi.han/data/Google_GZ_CD256/val/'
    elif opt.data_name == 'LEVIRsup-WHUunsup':
        opt.train_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/train/'
        opt.val_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/val/'

    train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize,opt.train_ratio, num_workers=8, shuffle=True, pin_memory=False)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=6, shuffle=False, pin_memory=False)
    # train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize,opt.train_ratio, num_workers=0, shuffle=True, pin_memory=True)
    # val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=False, pin_memory=True)

    Eva_train = Evaluator(num_class = 2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)


    model=SemiModel().cuda()
    ema_model = SemiModel().cuda()

    for param in ema_model.parameters():
        param.detach_()

    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")
    # args = parser.parse_args()
    # print('现在的数据是：',args.data_name)


    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # 可以先全用有标签的训练几个epoch，再进行半监督训练 !!!!
        # if epoch<5: #默认的为5，测试10，15，20
        #     use_ema=False
        #     # print('labeled ration=1,Ablation现在监督训练的次数为:20!')
        # else:
        #     use_ema=True

        # 全程ema=False，即一直只用有标签的进行训练，不进行半监督学习
        
        use_ema=False

        Eva_train.reset()
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()
        train1(train_loader, val_loader, Eva_train,Eva_train2, Eva_val,Eva_val2, data_name, save_path, model,
              ema_model, criterion,semicriterion, optimizer,use_ema, opt.epoch)

        lr_scheduler.step()
        # print('现在的数据是：', args.data_name)


end=time.time()
print('程序训练train的时间为:',end-start)

