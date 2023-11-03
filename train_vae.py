import numpy as np
import cv2
import os
from tqdm import tqdm



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from math import exp

from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

from models import Vae_mute_sim, Vae_mute_sim_new
from coder_new import Dataset_rna_seq, load_img, save_img, convert_bp2tensor, convert_feat2bp
import p_tqdm

from random import seed, shuffle, random, randint, choice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from coder_new import max_len, crop_size, pad_size, num_sub, len_sub, num_index, len_index, len_sub_index

dict_id2bp = {
    0:'A',
    1:'G',
    2:'C',
    3:'T',
}

dict_bp2onehot = {
    'A':[1,0,0,0],
    'G':[0,1,0,0],
    'C':[0,0,1,0],
    'T':[0,0,0,1],
    }

def image_split(img, img_size, stride=80, flag_all=False):
    h, w, c = img.shape
    list_img_packs = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            try:
                temp_pack = img[i:i+img_size, j:j+img_size, :]
                if flag_all:
                    temp_pack = cv2.resize(temp_pack, (img_size, img_size))
                    list_img_packs.append(temp_pack)
                else:
                    if np.max(temp_pack)<=10:
                        continue
                    else:
                        temp_pack = cv2.resize(temp_pack, (img_size, img_size))
                        list_img_packs.append(temp_pack)
                    
            except Exception as e:
                continue
    return list_img_packs            

def image_merge(img_packs, img_size, stride=80, size_h=100, size_w=50):
    h=img_size*size_h
    w=img_size*size_w
    image = np.zeros(shape=(h, w, 3))
    index = 0
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            try:
                if pad_size>0:
                    image[i:i+img_size, j:j+img_size,:] = img_packs[index][pad_size:-pad_size,pad_size:-pad_size,:]
                else:
                    image[i:i+img_size, j:j+img_size,:] = img_packs[index][:,:,:]
                index += 1
            except Exception as e:
                continue
    return image

def data_gen_img(img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True) 
    img = load_img(img_dir)
    img_packs = image_split(img, crop_size, crop_size, False)
    img_name = img_dir.split('/')[-1].split('.')[0]
    for (i, temp_pack) in enumerate(tqdm(img_packs, ncols=100)):
        save_img(save_dir+img_name+'pack_' + str(i)+'.png', temp_pack) 

def data_gen_train():
    root_dir = './images_0713/'
    list_img_names = os.listdir(root_dir)
    for temp_img_name in tqdm(list_img_names, ncols=100):
        temp_img_dir = root_dir + temp_img_name
        data_gen_img(temp_img_dir, './train_img_packs'  + str(crop_size) + '/')

def dataset_gen_train():
    list_data = []
    root_dir = './images_0713/'
    list_img_names = os.listdir(root_dir)
    for temp_img_name in tqdm(list_img_names, ncols=100):
        temp_img_dir = root_dir + temp_img_name

        img = load_img(temp_img_dir)
        img_packs = image_split(img, crop_size, crop_size, True)
        list_data += img_packs

    temp_save_dir = './dataset_' + str(crop_size) + '.npy' 
    np.save(temp_save_dir, list_data)
    print('generate finished~')


def get_img_pad(img, u,d,l,r):
    img_pad = cv2.copyMakeBorder(img, u, d, l, r, cv2.BORDER_REFLECT, dst=None)
    return img_pad

def random_trans(img):
    h, w = img.shape[:2]
    h_i = randint(0, h//2)
    w_i = randint(0, w//2)
    img_ori = img[h_i:,w_i:]
    if random() < 0.5:
        img_dst = cv2.resize(img_ori, (w, h))
    else:
        img_dst = get_img_pad(img_ori,h-h_i, 0, w-w_i, 0)
        img_dst = cv2.resize(img_dst, (w, h))
    return img_dst

def random_rotate(img):
    h, w = img.shape[:2]
    temp_rotate = randint(-90, 90)
    # temp_dx =  uniform(-0.5, 0.5)*(w//4)
    # temp_dy =  uniform(-0.5, 0.5)*(h//4)
    temp_scale = 1.0
    temp_rotate_matirx = cv2.getRotationMatrix2D((w//2, h//2), temp_rotate, scale=temp_scale)
    temp_rotate_matirx[0,2] += 0
    temp_rotate_matirx[1,2] += 0
    img_random = cv2.warpAffine(img, temp_rotate_matirx, dsize=(w, h))
    return img_random

def random_flip(img):
    temp_flag =  choice([0,1,-1])
    img = cv2.flip(img, temp_flag)
    return img

def random_clahe(img, cliplimits=[2,8], gridsizes=[4,12], ratio=0.5):
    # print(img.shape)
    if  random() < ratio:
        temp_cliplimit =  uniform(cliplimits[0], cliplimits[1])
        temp_grid =  randint(gridsizes[0], gridsizes[1])
        clahe=cv2.createCLAHE(temp_cliplimit, (temp_grid, temp_grid))
        b, g, r = cv2.split(img)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        img_dst = cv2.merge([b, g, r])
    else:
        img_dst = img
    return img_dst

def random_bright_contrast(img):
    alphas = np.random.uniform(0.8, 1.2)
    betas = np.random.uniform(-10, 10)
    img_bc = np.clip(alphas*img+betas, 0, 255)
    return img_bc

def random_noise(img):
    sigma = 0.05
    image = np.asarray(img / 255, dtype=np.float32)
    noise = np.random.normal(0, sigma, image.shape).astype(dtype=np.float32)
    output = image + noise
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output

def random_flur(img):
    image = np.asarray(img/255, dtype=np.float32)
    image = cv2.GaussianBlur(image, (3,3), 1, 1)
    output = np.clip(image, 0, 1)
    output = np.uint8(output * 255)
    return output

class Dataset_img_rna(Dataset):
    def __init__(self, list_data, status='train'):
        super(Dataset_img_rna, self).__init__()
        self.list_data = list_data
        self.status = status
    
    def __getitem__(self, index):
        temp_img = self.list_data[index]
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        
        temp_img = get_img_pad(temp_img, pad_size, pad_size, pad_size, pad_size)
        
        if self.status == 'train':
            pass
            
            # if  random() < 0.5:
            #     temp_img = random_trans(temp_img)

            # if random() < 0.2:
            #     temp_img = random_noise(temp_img)
            
            # if random() < 0.1:
            #     temp_img = random_flur(temp_img)

            # if random() < 0.25:
            #     temp_img = 255-temp_img

            if  random() < 0.5:
                temp_img = random_rotate(temp_img)

            if  random() < 0.5:
                temp_img = random_flip(temp_img)
            
            if random() < 0.5:
                temp_img = random_bright_contrast(temp_img)

            
            
            # if  random() < 0.25:
            #     temp_img = random_clahe(temp_img)

        # temp_img = np.expand_dims(temp_img, axis=2)
        temp_data = torch.from_numpy(temp_img).permute(2,0,1)
        # temp_data = norm_img(temp_data)
        data= {}
        data['input'] = temp_data.float()
        data['label'] = temp_data.float()
        return data

    def __len__(self):
        return len(self.list_data)



def mse(r1, r2):
    return torch.sqrt(torch.pow(r2-r1, 2)+1e-12)

def mae(r1, r2):
    return abs(r2-r1)

def GC_loss(feat):
    b = feat.shape[0]
    t1 = torch.sum(feat[:,:,0])
    t2 = torch.sum(feat[:,:,1])
    t3 = torch.sum(feat[:,:,2])
    t4 = torch.sum(feat[:,:,3])
    t_all = t1 + t2 + t3 + t4


    l_1 = t1/t_all
    l_2 = t2/t_all
    l_3 = t3/t_all
    l_4 = t4/t_all

    l23 = l_2+l_3 

    # loss_gc = mae(l_1, 0.25) + mae(l_2,0.25) + mae(l_3, 0.25) + mae(l_4, 0.25)
    loss_gc = mae(l23*1.9e4, 0.5*1.9e4)
    # loss_gc = mae(l_2,0.25) + mae(l_3, 0.25)
    
    return loss_gc

def BP_loss(feat):
    b,l,c = feat.shape
    list_feats = []
    for i in range(0,l-1,1):
        try:
            temp_f1 = feat[:,i:i+1,:]
            temp_f2 = feat[:,i+1:i+2,:]
            list_feats.append(temp_f1*temp_f2)
        except Exception as e:
            continue
    feats = torch.concatenate(list_feats, dim=1)
    # loss_bps = (torch.sum(feats)+1e-6)/(len(list_feats)/2)
    loss_bps = (torch.sum(feats)+1e-6)/b
    loss_bps = abs(2.0-loss_bps)
    return loss_bps

def diver_loss(feat):
    b = feat.shape[0]
    feat_flat = feat.view(b, -1)
    dist_matrix = torch.cdist(feat_flat, feat_flat, p=2.0)
    diversity_loss = 1e9/(dist_matrix.sum()+1e-7)
    return diversity_loss



def dataset_gen_train():
    list_data = []
    root_dir = './images_0713/'
    list_img_names = os.listdir(root_dir)
    for temp_img_name in tqdm(list_img_names, ncols=100):
        temp_img_dir = root_dir + temp_img_name

        img = load_img(temp_img_dir)
        img_packs = image_split(img, crop_size, crop_size, True)
        list_data += img_packs

    temp_save_dir = './dataset_' + str(crop_size) + '.npy' 
    np.save(temp_save_dir, list_data)
    print('generate finished~')

def ssim_loss(y_pred, y_true, window_size=11, size_average=True, sigma=1.5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    channel = y_pred.size(1)
    window = create_window(window_size, channel).to(y_pred.device)

    mu1 = nn.functional.conv2d(y_pred, window, padding=window_size//2, groups=channel)
    mu2 = nn.functional.conv2d(y_true, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(y_pred * y_pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(y_true * y_true, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(y_pred * y_true, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)

def sobel_edge_loss(y_pred, y_true):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    if y_pred.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()

    G_x_pred = nn.functional.conv2d(y_pred, sobel_x, padding=1)
    G_y_pred = nn.functional.conv2d(y_pred, sobel_y, padding=1)
    G_pred = torch.sqrt(G_x_pred ** 2 + G_y_pred ** 2)

    G_x_true = nn.functional.conv2d(y_true, sobel_x, padding=1)
    G_y_true = nn.functional.conv2d(y_true, sobel_y, padding=1)
    G_true = torch.sqrt(G_x_true ** 2 + G_y_true ** 2)

    loss = nn.functional.l1_loss(G_pred, G_true)
    return loss

def mutate(source_dna_sequence):
    mutate_number = int(0.0150 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1) + 1
    insert_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1) + 1
    delete_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1) + 1


    target_dna_sequence = list(source_dna_sequence)
    while True:
        for _ in range(mutate_number):
            location = randint(0, len(target_dna_sequence) - 1)
            source = target_dna_sequence[location]
            target = choice(list(filter(lambda base: base != source, ["A", "C", "G", "T"])))
            target_dna_sequence[location] = target
        for _ in range(insert_number):
            location = randint(0, len(target_dna_sequence))
            target_dna_sequence.insert(location, choice(["A", "C", "G", "T"]))
        for _ in range(delete_number):
            location = randint(0, len(target_dna_sequence) - 1)
            del target_dna_sequence[location]
        if "".join(target_dna_sequence) != source_dna_sequence:
            target_dna_sequence = "".join(target_dna_sequence)
            break
        target_dna_sequence = list(source_dna_sequence)
    return target_dna_sequence

def acc_seq(list_preds, list_labels):
    acc_num = 0
    for temp_p, temp_l in zip(list_preds, list_labels):
        if np.array_equal(np.array(temp_p), np.array(temp_l)):
            acc_num += 1
    
    print('Rec Acc:{:.4f}'.format(acc_num/len(list_preds)))


class Dataset_seq_mute(Dataset):
    def __init__(self, list_data, status='train', scale=1):
        super().__init__()
        self.list_data = list_data*scale
        self.status = status

    def __getitem__(self, index):
        # temp_seq = seq_gen(len_sub, num_sub)
        temp_seq = self.list_data[index]
        temp_seq_m = temp_seq

        len_max = len(temp_seq)
        temp_index = choice(['A','G','T','C'])*len_index
        temp_seq = temp_seq + temp_index

        if self.status == 'train':
            if  random() < 1.0:
                temp_seq_mute = mutate(temp_seq)
            else:
                temp_seq_mute = temp_seq_m
        else:
            temp_seq_mute = temp_seq_m
        
        temp_seq = temp_seq[:len_max]
        temp_seq_mute = temp_seq_mute[:len_max]

        seq_tensor = convert_bp2tensor(temp_seq, len_max)
        seq_label = torch.argmax(seq_tensor, dim=-1)
        seq_mute_tensor = convert_bp2tensor(temp_seq_mute, len_max)
        
        data= {}
        data['seq'] = seq_tensor.float()
        data['seq_mute'] = seq_mute_tensor.float()
        data['seq_label'] = seq_label.long()
        return data

    def __len__(self):
        return len(self.list_data)


def train_mute_all():
    batch_size = 3000
    epoch_max = 100
    lr = 1e-6
    num_workser = 2

    print('load dataet~~')
    try:
        list_data = np.load('./dataset_'+str(crop_size)+'.npy')
    except Exception as e:
        dataset_gen_train()
        list_data = np.load('./dataset_'+str(crop_size)+'.npy')
    print(list_data.shape)
    print(list_data[0].shape)
    print('crop_size:', crop_size)
    print('pad_size:', pad_size)
    print('max_len:', max_len)
    print('num_sub:', num_sub)
    print('len_sub:', max_len//num_sub)
    print('batchsize:', batch_size)
    print('epoch_max:', epoch_max)
    print('num_index:', num_index)
    print('len_index:', len_index)

    data_train = Dataset_img_rna(list_data)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workser, drop_last=True)
    

    model = Vae_mute_sim(max_len, crop_size, pad_size, num_sub).to(device)
    temp_dir = './'+str(crop_size)+'_'+str(max_len)+'_'+str(num_sub)+'_'+ str(len_sub) +'/'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        model.load_state_dict(torch.load(temp_dir+'./model_vae.pb', map_location='cpu'), strict=True)
        pass
    except Exception as e:
        print(e)
    loss_fun_l1 = nn.L1Loss(reduction='sum')
    loss_fun_ce = nn.CrossEntropyLoss(reduction='sum')

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)


    lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*1//3, epoch_max*4//5])

    best_ssim = 0

    for epoch in range(epoch_max):
        model.train()
        with tqdm(loader_train, ncols=150) as tqdmDataLoader:
            for i, data in enumerate(tqdmDataLoader):
                input = data['input'].to(device)
                label = data['label'].to(device)
                    
                
                feat, mu, logvar = model.img2seq(input)

                loss_kl = abs(-0.5 * torch.sum(1 + logvar - (mu)**2 - logvar.exp()) + 1e-6)

                loss_gc = GC_loss(feat)
                loss_bps = BP_loss(feat)
                loss_div = diver_loss(feat)
                loss_feat = loss_gc + loss_bps + loss_div


                _, out = model.seq2img(mu, logvar, feat)

                loss_l1 = loss_fun_l1(out, label)/batch_size


                
                loss_rec = loss_l1

                # feat_id = torch.argmax(feat, dim=-1)
                # loss_mute = loss_fun_ce(pred_rec.reshape(-1,4),  feat_id.reshape(-1))


                loss = loss_rec + loss_kl + loss_feat

                with torch.autograd.detect_anomaly():
                    loss.backward()
                optim.step()
                optim.zero_grad()


                tqdmDataLoader.set_postfix(ordered_dict={
                        "Epoch":epoch+1,

                        "L_l1":loss_l1.item(),

                        "L_gc":loss_gc.item(),
                        "L_bp":loss_bps.item(),
                        "L_div":loss_div.item(),
                        "L_kl":loss_kl.item(),
                        # "L_mute":loss_mute.item(),

                        "LR_en":optim.param_groups[0]['lr'],
                    })

        model.eval()
        torch.save(model.state_dict(), temp_dir+'./model_vae.pb')
        lr_sh.step()
        # lr_sh_en.step()


        if (epoch+1) % 1 == 0:
            eval_img_sim('./91.491/images_0713/15DPI_3.bmp', model)
            temp_ssim = eval_img_sim('./91.491/images_0713/Stage57.bmp', model)
            
            if temp_ssim >= best_ssim:
                best_ssim = temp_ssim
                torch.save(model.state_dict(), temp_dir+'./model_vae_best.pb')
                print('========================================================================good job===================================================================================')
            # eval_img_sim_test('./images_0713/15DPI_3.bmp', best_ssim_only)



def finetune_mute_all():
    batch_size = 4000
    epoch_max = 200
    lr = 1e-3
    num_workser = 8

    print('load dataet~~')
    try:
        list_data = np.load('./dataset_'+str(crop_size)+'.npy')
    except Exception as e:
        dataset_gen_train()
        list_data = np.load('./dataset_'+str(crop_size)+'.npy')
    print(list_data.shape)
    print(list_data[0].shape)
    print('crop_size:', crop_size)
    print('pad_size:', pad_size)
    print('max_len:', max_len)
    print('num_sub:', num_sub)
    print('len_sub:', max_len//num_sub)
    print('batchsize:', batch_size)
    print('epoch_max:', epoch_max)

    data_train = Dataset_img_rna(list_data, status='train')
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workser, drop_last=True)
    

    model = Vae_mute_sim(max_len, crop_size, pad_size, num_sub).to(device)
    temp_dir = './'+str(crop_size)+'_'+str(max_len)+'_'+str(num_sub)+'_'+ str(len_sub) +'/'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        model.load_state_dict(torch.load(temp_dir+'./model_vae_best.pb', map_location='cpu'), strict=False)
        pass
    except Exception as e:
        print(e)
    loss_fun_l1 = nn.L1Loss(reduction='sum')
    loss_fun_ce = nn.CrossEntropyLoss(reduction='sum')

    optim = torch.optim.AdamW(model.dna_mute.parameters(), lr=lr, weight_decay=5e-4)


    lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*1//2, epoch_max*4//5])

    list_feats = []
    with tqdm(loader_train, ncols=100) as tqdmDataLoader:
        for i, data in enumerate(tqdmDataLoader):
            input = data['input'].to(device)
            
            with torch.no_grad():
                feat, mu, logvar = model.img2seq(input)

            list_feats.append(feat.detach().cpu().numpy())
            
        list_feats = np.concatenate(list_feats, axis=0)
        list_feats = convert_feat2bp(list_feats)
    
    list_data = []
    for feat in list_feats:
        list_data.append(feat)

    data_train = Dataset_seq_mute(list_data)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workser, drop_last=True)


    best_ssim = 0

    for epoch in range(epoch_max):
        model.eval()
        model.dna_mute.train()
        list_preds = []
        list_labels = []
        with tqdm(loader_train, ncols=150) as tqdmDataLoader:
            for i, data in enumerate(tqdmDataLoader):
                # if i % 100==0:
                    # temp_id = 2000+i%30
                    # seed(temp_id)


                seq = data['seq'].to(device)
                seq_mute = data['seq_mute'].to(device)
                seq_label = data['seq_label'].to(device)
                    
                
                pred_rec = model.dna_mute(seq_mute)

                # loss_l1 = loss_fun_l1(pred_rec, seq)
                loss_ce = loss_fun_ce(pred_rec.reshape(-1,4), seq_label.reshape(-1))/batch_size

                loss = loss_ce

                # with torch.autograd.detect_anomaly():
                loss.backward()
                optim.step()
                optim.zero_grad()

                temp_pred = torch.argmax(pred_rec, dim=-1).detach().cpu().numpy().tolist()
                temp_label = torch.argmax(seq, dim=-1).detach().cpu().numpy().tolist()
                list_preds += temp_pred
                list_labels += temp_label


                tqdmDataLoader.set_postfix(ordered_dict={
                        "Epoch":epoch+1,
                        "L_l1":loss.item(),
                        "LR_en":optim.param_groups[0]['lr'],
                    })

        acc_seq(list_preds[:], list_labels[:])

        model.eval()
        torch.save(model.state_dict(), temp_dir+'./model_vae_mute.pb')
        lr_sh.step()



def metric_gc(dna_sequences):
    h_statistics, gc_statistics = [], []
    for dna_sequence in dna_sequences:
        homopolymer = 1
        while True:
            found = False
            for nucleotide in ["A", "C", "G", "T"]:
                if nucleotide * (1 + homopolymer) in dna_sequence:
                    found = True
                    break
            if found:
                homopolymer += 1
            else:
                break
        gc_bias = abs((dna_sequence.count("G") + dna_sequence.count("C")) / len(dna_sequence) - 0.5)

        h_statistics.append(homopolymer)
        gc_statistics.append(gc_bias)

    maximum_homopolymer, maximum_gc_bias = np.mean(h_statistics), np.mean(gc_statistics)
    h_score = (1.0 - (maximum_homopolymer - 1) / 5.0) / 2.0 if maximum_homopolymer < 6 else 0
    c_score = (1.0 - maximum_gc_bias / 0.3) / 2.0 if maximum_gc_bias < 0.3 else 0
    
    # temp_h = (1.0 - (maximum_homopolymer - 1) / 5.0) / 2.0
    # temp_c = (1.0 - maximum_gc_bias / 0.3) / 2.0
    print('homopoly:{:.4f}, {:.4f}'.format(maximum_homopolymer, h_score))
    print('gc_bias :{:.4f}, {:.4f}'.format(maximum_gc_bias, c_score))
    compatibility_score = 0.4 + c_score
    return compatibility_score


def eval_img_sim(img_dir, model, flag_mute=False):
    model.eval()
    
    expected_image = load_img(img_dir)

    list_img_packs = image_split(expected_image, crop_size, crop_size, True)
    data_test = Dataset_img_rna(list_img_packs, 'valid')
    loader_test = DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0, drop_last=False)
    list_img_packs_pred = []
    list_feats = []

    with tqdm(loader_test, ncols=100) as tqdmDataLoader:
        for i, data in enumerate(tqdmDataLoader):
            input = data['input'].to(device)
            with torch.no_grad():
                feat, _, _ = model.img2seq(input)
                # out = model.seq2img(feat)

            feat = feat.detach().cpu().numpy()
            list_feats.append(feat)
    
    
    feat_pred = np.concatenate(list_feats, axis=0)

    list_pred_bps = convert_feat2bp(feat_pred)
    # print(list_pred_bps[0][:100])
    print('diff seq num:', len(set(list_pred_bps)))
    compatibility_score = metric_gc(list_pred_bps)

    
    list_img_packs_pred = []

    data_test = Dataset_rna_seq(list_pred_bps, max_len)
    loader_test = DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0, drop_last=False)

    with tqdm(loader_test, ncols=100) as tqdmDataLoader:
        for i, data in enumerate(tqdmDataLoader):
            input = data['input'].to(device)
            with torch.no_grad():
                if flag_mute:
                    _, out = model.generate_mute(input)
                else:
                    _, out = model.generate(input)

            out = out.detach().permute(0,2,3,1).cpu().numpy()
            list_img_packs_pred.append(out)

    img_packs_pred = np.concatenate(list_img_packs_pred, axis=0)

    obtained_image = image_merge(img_packs_pred, crop_size, crop_size, 8000//crop_size, 4000//crop_size)
    obtained_image = cv2.resize(obtained_image, (expected_image.shape[1], expected_image.shape[0]))
    save_img('ob.bmp', obtained_image)
    

    d1 = len(list_img_packs)*(max_len+num_index*len_sub_index)
    d2 = 8000*4000*24
    density_score = 1 - d1/d2
    print('density score:{:.4f}'.format(density_score))
    print('ompatibility score:{:.4f}'.format(compatibility_score))

    ssim_value = structural_similarity(expected_image, obtained_image, multichannel=True)
    recovery_score = (ssim_value - 0.84) / 0.16 if ssim_value > 0.84 else 0
    print('ssim:{:.4f}, rec score:{:.4f}'.format(ssim_value, recovery_score))

    temp_score = (0.2 *density_score  + 0.3 * compatibility_score + 0.5 * recovery_score) * 100
    print('socre:', temp_score)

    return temp_score



def eval_img_new(test_num=10, best_flag=False):
    model = Image_Net_vae_mute(max_len, crop_size, pad_size, num_sub).to(device)
    temp_dir = './'+str(crop_size)+'_'+str(max_len)+'_'+str(num_sub)+'_'+ str(len_sub) +'/'
    if best_flag:
        model.load_state_dict(torch.load(temp_dir+'./model_vae_best.pb', map_location='cpu'), strict=True)
    else:
        model.load_state_dict(torch.load(temp_dir+'./model_vae.pb', map_location='cpu'), strict=True)
    model = model.to(device)
    model.eval()

    list_ssims = []
    list_scores = []

    root_dir = './images_0713/'
    list_img_names = os.listdir(root_dir)[:test_num]
    for temp_img_name in list_img_names:
        temp_img_dir = root_dir + temp_img_name
        
        temp_score = eval_img_sim(temp_img_dir, model)
        list_scores.append(temp_score)
        print('socre:', temp_score)
    print('average score:{:.3f}'.format(np.mean(list_scores)))
    return list_ssims




if __name__ == '__main__':

    # # ## train model ==> image -> rna & rna -> image
    train_mute_all()
    finetune_mute_all()


    # # # print('test valid flow:========================================================')
    # eval_img_new(10, True)



    
        
