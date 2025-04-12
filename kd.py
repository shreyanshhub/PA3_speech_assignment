import time
import numpy as np
import pickle
import os
import random
import argparse

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

from modelss import fc_3layers, FineTuneCNN14, FineTuneMBNetV2, FineTuneResNet38


os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
random.seed(1)

#%%

parser = argparse.ArgumentParser(description='training hyper-params')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Name of model to train (default: "countception"')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num_epochs', default=100, type=int, 
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--learning_rate', default=0.001, type=float, 
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=50, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--Temp', type=int, default=4, 
                    help='Image patch size (default: None => model default)')
parser.add_argument('--shuffle', type=int, default=1, choices=[0,1],
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N',
                    help='Input image center crop percent (for validation only)')
parser.add_argument('--par', default=None, type=str,
                    help='Participant name')
args = parser.parse_args()

P = 15
win_size = 10 
hop = .5
window_size = 1024
hop_size = 320
fmin, fmax = 50, 11000
mel_bins = 64
sr = 22050
device = 'cuda'
learning_rate = args.learning_rate 
num_epochs = args.num_epochs
alpha = args.alpha 
Temp = args.Temp
batch_size = args.batch_size 
sub = args.par
classes_num = args.num_classes

print('learning_rate:', learning_rate, 'num_epochs:', num_epochs, 'alpha:', alpha, 'Temp:', Temp, 'batch_size:', batch_size)


# load teacher set and model
def load_teacher_model(PATH_teacher_models, teacher_model_name, classes_num):

    device = 'cuda'
    # load teacher model
    Model = eval(teacher_model_name)
    
    if teacher_model_name == 'fc_3layers':
        model = Model(classes_num=classes_num)
    elif teacher_model_name == 'FineTuneCNN14':
        # all data is re-sampled to 16k as model input, no matter what target sr is
        model = Model(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, 
                      fmin=50, fmax=8000, classes_num=classes_num)
    elif teacher_model_name in ['FineTuneMBNetV2', 'FineTuneResNet38']:
        model = Model(sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64,
                                      fmin=50, fmax=8000, classes_num=classes_num)
    else:
        raise ValueError("teacher model name not supported!")

    print('GPU number: {}'.format(torch.cuda.device_count()))
    if 'cuda' in str(device):
        model.to(device)
    ckpt_path = [os.path.join(PATH_teacher_models, item) for item in os.listdir(PATH_teacher_models) if item.endswith('.pth')][0]
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.eval()
    if 'cuda' in str(device):
        model.to(device)

    return model


def load_data(PATH_data, sr_teacher, sr_student, sub, seg_t, classes_num, student_model_name):
    ''' Load the resampled audio data '''
    
    if student_model_name == 'fc_3layers':
        with open(PATH_data + 'split/sr_%d/%s/seg_%ds_train_emb_resampled.pkl' % (sr_teacher, sub, seg_t), 'rb') as f:
            tmp_blob = pickle.load(f)[:, :]
            X_train_t = tmp_blob[:, :-1].reshape((len(tmp_blob), 1024, 1))
            y_train_t = tmp_blob[:, -1].astype('int64')
        with open(PATH_data + 'split/sr_%d/%s/seg_%ds_train_emb_resampled.pkl' % (sr_student, sub, seg_t), 'rb') as f:
            tmp_blob = pickle.load(f)[:, :]
            X_train_s = tmp_blob[:, :-1].reshape((len(tmp_blob), 1024, 1))
            y_train_s = tmp_blob[:, -1].astype('int64')
        with open(PATH_data + 'split/sr_%d/%s/seg_%ds_test_emb_resampled.pkl' % (sr_student, sub, seg_t), 'rb') as f:
            tmp_blob = pickle.load(f)[:, :]
            X_test = tmp_blob[:, :-1].reshape((len(tmp_blob), 1024, 1))
            y_test = tmp_blob[:, -1].astype('int64')
            
    elif student_model_name in ['FineTuneCNN14', 'FineTuneMBNetV2', 'FineTuneResNet38']:
        with open(PATH_data + 'split_full_dataset/sr_%d/%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (sr_teacher, sub, seg_t), 'rb') as f:   # for esc: split_full_dataset; adl: split
            tmp_blob = pickle.load(f)[:, :]
            X_train_t = tmp_blob[:, :-1]
            y_train_t = tmp_blob[:, -1].astype('int64')
        with open(PATH_data + 'split_full_dataset/sr_%d/%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (sr_student, sub, seg_t), 'rb') as f:
            tmp_blob = pickle.load(f)[:, :]
            X_train_s = tmp_blob[:, :-1]
            y_train_s = tmp_blob[:, -1].astype('int64')
        with open(PATH_data + 'split_full_dataset/sr_%d/%s/seg_%ds_test_cnn14_inpt_resampled.pkl' % (sr_student, sub, seg_t), 'rb') as f:
            tmp_blob = pickle.load(f)[:, :]
            X_test = tmp_blob[:, :-1]
            y_test = tmp_blob[:, -1].astype('int64')
            
    else:
        raise ValueError("data for this model name not prepared!")
        
    assert len(X_train_t) == len(y_train_t), "teacher training labels and data not match!"
    assert len(X_train_s) == len(y_train_s), "student training labels and data not match!"
    assert abs(len(X_train_t) - len(X_train_s)) <= 1, "teacher and student training size not match, please check!"
    if abs(len(X_train_t) - len(X_train_s)) == 1:
        n = min(len(X_train_t), len(X_train_s))
        for i in range(n):
            if y_train_t[i] != y_train_s[i]:
                if len(y_train_t) > len(y_train_s):
                    y_train_t = np.delete(y_train_t, i, 0)
                    X_train_t = np.delete(X_train_t, i, 0)
                else:
                    y_train_t = np.insert(y_train_t, i, y_train_s[i])
                    print('instance length (16khz * sec):', X_train_s.shape[1])
                    X_train_t = np.insert(X_train_t, i, X_train_s[i], axis=0)
            if y_train_t[i] != y_train_s[i]:
                raise RuntimeError("teacher and student training size cannot fix in one-step distance, please check!")
                
    print('training and val size (model sr: %d):' %sr_student, np.shape(X_train_t), np.shape(X_test), np.shape(y_train_t))
    assert len(set(y_train_t)) == classes_num, "class # is not correct, please check!"
    assert np.array_equal(y_train_t, y_train_s), "teacher and student labels not sycned!"

    x_train_tensor_t = torch.from_numpy(np.array(X_train_t)).float()
    x_train_tensor_s = torch.from_numpy(np.array(X_train_s)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train_s)).float()
    x_test_tensor = torch.from_numpy(np.array(X_test)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).float()

    return x_train_tensor_t, x_train_tensor_s, y_train_tensor, x_test_tensor, y_test_tensor


def stat_metric(args, f1_macro, precision, recall, PATH_log, train_loss_list, test_loss_list, test_f1_list):
    lr = args.learning_rate
    epochs = args.num_epochs
    alp = args.alpha 
    temp = args.Temp
    bs = args.batch_size  
    par = args.par
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('CNN14FC', par,lr,epochs,alp,temp, bs, f1_macro, precision, recall, train_loss_list, test_loss_list, test_f1_list), 
          file=open(PATH_log + 'participant_{}.logs'.format(par),'a+'))


def kd_loss(logits, truth, T=8.0, alpha=0.9):
    p = F.log_softmax(logits/T, dim=1)
    q = F.softmax(truth/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / logits.shape[0]

    return l_kl * alpha

def seed_worker(worker_id):
    worker_seed = 0 
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#%%
    
def main():
    #PATH_data = '/home/dl33629@austin.utexas.edu/research-lowhigh/scripted_study_adl/'   # for adl
    PATH_data = './data/'   # for esc
    seg_t = 5   # in sec
    sr_teacher = 2000
    sr_student = 2000

    print('handling participant:', sub)
    
   
    # path to the 1st teacher model
    #PATH_teacher_models = './models/cnn14+fc/sr_%d_shuffle/seg_%ds/models_for_esc_50class/%s/' %(sr_teacher, seg_t, sub)   # for esc: models_for_esc_50class
    PATH_teacher_models = './models/mbnet+fc/student/esc50_50class/FineTuneMBNetV2/sr_%d_1teacher/seg_5s/%s/' %(1000, sub)   # for multi stage
    # path to the 2nd teacher model, if any
    #PATH_teacher_models2 = '/home/research-lowhigh/models/mbnet+fc/sr_%d/seg_%ds/models_for_adl/%s/' %(16000, seg_t, sub)
    PATH_teacher_models2 = None
    teacher_model_name = 'FineTuneMBNetV2'   #'FineTuneCNN14', 'fc_3layers', 'FineTuneMBNetV2', 'FineTuneResNet38'
    student_model_name = 'FineTuneMBNetV2'   #'FineTuneCNN14', 'fc_3layers', 'FineTuneMBNetV2', 'FineTuneResNet38'
    # adl and esc share the same path to save checkpoints / logs
    PATH_save_models = './models/cnn14+fc/student/esc50_50class/%s/sr_%d/seg_%ds/%s/' %(student_model_name, sr_student, seg_t, sub)
    PATH_log = './models/cnn14+fc/student/esc50_50class/%s/sr_%d/seg_%ds/%s/' %(student_model_name, sr_student, seg_t, sub)
    
    # prepare teacher model
    teacher_model = load_teacher_model(PATH_teacher_models, teacher_model_name, classes_num)
    
    if PATH_teacher_models2 is not None:
        teacher_model2 = load_teacher_model(PATH_teacher_models2, teacher_model_name, classes_num)
    
    # prepare student model
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)        
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
       
    torch.cuda.empty_cache()
    
    Model = eval(student_model_name)
    
    if student_model_name == 'fc_3layers':
        model = Model(classes_num=classes_num)
    elif student_model_name == 'FineTuneCNN14':
        model = Model(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, 
                      fmin=50, fmax=8000, classes_num=classes_num)
    elif student_model_name in ['FineTuneMBNetV2', 'FineTuneResNet38']:
        model = Model(sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64,
                                      fmin=50, fmax=8000, classes_num=classes_num)
    else:
        raise ValueError("student model name not supported!")
        
    if 'cuda' in str(device):
        model.to(device)
    
    # load training and test data    
    x_train_tensor_teacher, x_train_tensor_student, y_train_tensor_student, x_test_tensor_student, y_test_tensor_student = load_data(PATH_data, sr_teacher, 
                                                                                                             sr_student, sub, seg_t, classes_num, student_model_name)
          
    train_data = TensorDataset(x_train_tensor_teacher, x_train_tensor_student, y_train_tensor_student)
    test_data = TensorDataset(x_test_tensor_student, y_test_tensor_student)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                        batch_size=batch_size,
                        num_workers=0, pin_memory=True, shuffle = True,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                            batch_size=batch_size,
                            num_workers=0, pin_memory=True, shuffle = False,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
    
    # Parallel
    #print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)   # yamnet used parallel!
 
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    iteration = 0
    criterion = nn.CrossEntropyLoss()

    ### Training Loop ########
    f1_stop, recall_stop, stop_cnt = 0, 0, 0
    train_loss_list, test_loss_list, test_f1_list = [], [], []
    for epoch in range(num_epochs):
        if stop_cnt == 10:
            break
        
        for i, d in enumerate(train_loader, 0):
            # get the training inputs; data is a list of [inputs, labels]
            inputs_teacher, inputs_student, labels_student = d
            #assert labels_teacher == labels_student, "teacher and student labels not sycned!"
            
            with torch.no_grad():
                inputs_teacher = inputs_teacher.to(device)
                #labels_teacher = labels_teacher.to(device, dtype=torch.int64)
                teacher_out = teacher_model(inputs_teacher)
                teacher_logits = teacher_out['logits']
                
                if PATH_teacher_models2 is not None:
                    teacher_out2 = teacher_model2(inputs_teacher)
                    teacher_logits2 = teacher_out2['logits']

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # for the student
            inputs_student = inputs_student.to(device)
            labels_student = labels_student.to(device, dtype=torch.int64)
            
            # training
            model.train()       
            yhat = model(inputs_student)['logits']
            
            if PATH_teacher_models2 is None:
                loss = criterion(yhat, labels_student)*alpha + (1-alpha)*kd_loss(yhat, teacher_logits, T=Temp, alpha=1)
            else:
                loss = criterion(yhat, labels_student)*alpha + (1-alpha)*(kd_loss(yhat, teacher_logits,T=Temp, alpha=1) + kd_loss(yhat, teacher_logits2,T=Temp, alpha=1)) / 2
                #loss = criterion(yhat, labels_student)*alpha + (1-alpha)*kd_loss(yhat, (teacher_logits + teacher_logits2) / 2.0, T=Temp, alpha=1)               
            #loss = criterion(yhat, labels_student)
            #loss = criterion(yhat, labels_student)*alpha + (1-alpha)*criterion2(yhat, teacher_logits)

            loss.backward()
            optimizer.step()

        print('[Epoch %d]' % (epoch))
        print('Train loss: {}'.format(loss))
        train_loss_list.append(round(loss.item(), 5))

        # evaluation
        test_output = []
        true_test_output = []
        with torch.no_grad():
            for i_val, d_val in enumerate(test_loader, 0):
                inputs_val_student, y_val_student = d_val
                inputs_val_student = inputs_val_student.to(device)
                y_val_student = y_val_student.to(device, dtype=torch.int64)

                model.eval()
                yhat = model(inputs_val_student)['logits']
                test_loss  = criterion(yhat, y_val_student)

                test_output.append(yhat.data.cpu().numpy())
                true_test_output.extend(y_val_student.data.cpu().numpy())

            test_oo = np.argmax(np.vstack(test_output), axis = 1)
            true_test_oo = np.asarray(true_test_output)

            #accuracy = metrics.accuracy_score(true_test_oo, test_oo)
            # accuracy = metrics.balanced_accuracy_score(true_test_oo, test_oo)
            precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
            # record best metrics/model, recall = acc when class is balanced, e.g., esc-50
            #if f1_stop < fscore:
            if recall_stop < recall:
                model_best = copy.deepcopy(model)
                stop_cnt = 0
                f1_stop, recall_stop = fscore, recall
                precision_stop = precision
                # confusion = confusion_matrix(true_test_oo, test_oo, normalize=None).reshape((1, -1))
                iteration = epoch

                print('Test loss: {}'.format(test_loss))
                print('TEST average_precision: {}'.format(precision))
                print('TEST average recall: {}'.format(recall))
                print('TEST average f1: {}'.format(fscore))
    
            else:
                stop_cnt += 1
            test_loss_list.append(round(test_loss.item(), 5))
            test_f1_list.append(round(fscore, 5))

                
    print('Finished Training')

    ### Save model ########
    precision_stop, f1_stop, recall_stop = round(precision_stop, 4), round(f1_stop, 4), round(recall_stop, 4)
    if not os.path.exists(PATH_save_models):
        os.makedirs(PATH_save_models)
    torch.save(model_best.state_dict(), 
               PATH_save_models + 'alp%.2f_tmp%d_f1=%.4f_epoch%d.pth' % (args.alpha, args.Temp, f1_stop, iteration))

    # write to log
    stat_metric(args, f1_stop, precision_stop, recall_stop, PATH_log, train_loss_list, test_loss_list, test_f1_list)
    
    train_loader, test_loader = 0, 0
    train_data, test_data = 0, 0
    del model_best
    

main()
