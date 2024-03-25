import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import numpy as np
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator
from PIL import Image
from network.SemiModel import SemiModel

import time
start=time.time()

def test(test_loader, Eva_test, save_path, net):
    print("Strat validing!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)
            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy()

            for i in range(output.shape[0]):
                probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                final_savepath = save_path + filename[i] + '.png'
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            Eva_test.add_batch(target, pred)
    print('target.shape', target.shape)
    print('pred.shape', pred.shape)


    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA=Eva_test.OA()
    Kappa=Eva_test.Kappa()

    # print('[Test] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print('[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f,IoU: %.4f' % ( F1[1],Pre[1],Recall[1],OA[1],Kappa[1],IoU[1]))
    # print('F1-Score: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}\nOA: {:.2f}\nKappa: {:.2f}\nIoU: {:.2f}\n}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100, IoU[1] * 100))
    print('F1-Score: Precision: Recall: OA: Kappa: IoU: ')
    # print('{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))
    print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))
    # print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[0] * 100, Pre[0] * 100, Recall[0] * 100, OA[0] * 100, Kappa[0] * 100,IoU[0] * 100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')  #修改这里！！！
    parser.add_argument('--data_name', type=str, default='WHU', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='SemiModel_noema04', #修改这里！！！
                        help='the test rgb images root')

    # parser.add_argument('--save_path', type=str, default='./test_result/C2F-SemiCD/WHU/WHU-5-Student/') #半监督C2F-SemiCD影像保存路径！！！
    parser.add_argument('--save_path', type=str, default='./test_result/C2FNet/WHU/WHU-5-Student/') #全监督C2FNet影像保存路径！！


    opt = parser.parse_args()

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

    if opt.data_name == 'LEVIR':
        opt.test_root = '/data/chengxi.han/data/LEVIR CD Dataset256/test/'
        # opt.test_root = '/data/chengxi.han/data/LEVIR-CD-Dataset256/test256-sub/'
        # opt.test_root = '/data/chengxi.han/data/LEVIR CD Dataset256/test256-sub-331/'
        # opt.test_root = '/data/chengxi.han/data/LEVIR-CD-Dataset256/test256-sub-21/'
        # opt.test_root = '/data/chengxi.han/data/LEVIR-CD-Dataset256/test256-sub-1419/'
    elif opt.data_name == 'WHU':
        opt.test_root = '/data/chengxi.han/data/Building change detection dataset256/test/'
        # opt.test_root = '/data/chengxi.han/data/WHU-CD-256-Semi/test/'
        # opt.test_root = '/data/chengxi.han/data/Building change detection dataset256/test-sub/'
        # opt.test_root = '/data/chengxi.han/data/Building change detection dataset256/test-sub-61-3/'
    elif opt.data_name == 'CDD':
        opt.test_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/test/'
        # opt.test_root = '/data/chengxi.han/data/Building change detection dataset256/test-sub-CDD402/'
    elif opt.data_name == 'DSIFN':
        opt.test_root = '/data/chengxi.han/data/DSIFN256/test/'
    elif opt.data_name == 'SYSU':
        opt.test_root = '/data/chengxi.han/data/SYSU-CD/test/'
        # opt.test_root = '/data/chengxi.han/data/SYSU-CD/test-sub/'
    elif opt.data_name == 'S2Looking':
        opt.test_root = '/data/chengxi.han/data/S2Looking256/test/'
        # opt.test_root = '/data/chengxi.han/data/S2Looking256/test-sub/'
    elif opt.data_name == 'GoogleGZ':
        opt.test_root = '/data/chengxi.han/data/Google_GZ_CD256/test/'
    elif opt.data_name == 'LEVIRsup-WHUunsup':
        opt.test_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/test/'

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name + '/'
    test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_test = Evaluator(num_class=2)
    if opt.model_name == 'HANet_v2':
        model = HANet_v2().cuda()
    elif opt.model_name == 'SemiModel_noema04':
        model = SemiModel().cuda()

#-------测试模型teacher和student中任选一个！-------

#-------测试老师的模型teacher-------
    # save_path = './output/C2F-SemiCD/WHU-5/'  #半监督C2F-SemiCD模型路径！！
    # save_path = './output/C2FNet-WHU/'  #全监督C2FNet模型路径！！

    # # save_path = save_path + '/' + opt.model_name
    # save_path = save_path + opt.data_name + '/' + opt.model_name
    # opt.load = save_path + '_train1_' + '_best_teacher_iou.pth'
    # # opt.load = save_path + '_train1_'+ '_best_student_iou.pth'
    # if opt.load is not None:
    #     model.load_state_dict(torch.load(opt.load))
    #     print('load model from ', opt.load)

    # # print('路径是：WHU-labeled-5-semiloss-0.8')


    #-------测试学生的模型student-------
    # save_path = './output/C2F-SemiCD/WHU-5/'  #半监督C2F-SemiCD模型路径！！
    save_path = './output/C2FNet/WHU/'  #全监督C2FNet模型路径！！

    # save_path = save_path +  opt.model_name
    save_path = save_path + opt.data_name + '/' + opt.model_name
    # opt.load = save_path + opt.data_name + '/' + opt.model_name+ '_train1_' + '_best_student_iou.pth'
    
    # save_path = save_path + opt.model_name
    opt.load = save_path + '_train1_'+ '_best_student_iou.pth'
    # opt.load ='./output/LEVIR-5%/SemiModel_noema04_best_teacher_iou.pth'
    if opt.load is not None:
        print('load model from ', opt.load)
        checkpoint_stud = torch.load(opt.load)
        model.load_state_dict(checkpoint_stud['best_student_net '])
    
#-------测试模型teacher和student中任选一个！-------


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test(test_loader, Eva_test, opt.save_path, model)

end=time.time()
print('程序测试test的时间为:',end-start)
