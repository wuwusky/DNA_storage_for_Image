import numpy as np
import cv2
import torch
import pickle
import re
from torch.utils.data import Dataset, DataLoader 
from evaluation import DefaultCoder
# temp_ss = 'A'*10
# print(temp_ss)
import os
current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'


from models import *

max_len = 138
crop_size = 10
pad_size = 0
num_sub  = 3
len_sub = max_len//num_sub
num_index = 5
len_sub_index = 12
len_index = num_index*len_sub_index

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

def load_img(dir):
    img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def get_img_pad(img, u,d,l,r):
    img_pad = cv2.copyMakeBorder(img, u, d, l, r, cv2.BORDER_REFLECT, dst=None)
    return img_pad

def save_img(dir, img):
    cv2.imencode('.png', img)[1].tofile(dir)

def norm_img(img):
    t_max = 255
    t_min = 0
    img_norm = (img-t_min)/(t_max-t_min)
    return img_norm

def image_split(img, img_size, stride=80, flag_all=False):
    h, w, c = img.shape
    list_img_packs = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            try:
                temp_pack = img[i:i+img_size, j:j+img_size, :]
                if np.sum(temp_pack)<=10*img_size*img_size and not flag_all:
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

def convert_feat2bp(list_feats):
    list_pred_bps = []
    for feat in list_feats:
        feat = feat.reshape(-1, 4)
        feat = np.argmax(feat, axis=1)
        img_bp = ''
        for bp in feat:
            img_bp += dict_id2bp[bp]
        list_pred_bps.append(img_bp)
    return list_pred_bps

def convert_feat2bp_with_index(list_feats):
    list_pred_bps = []
    for feat in list_feats:
        feat = feat.reshape(-1,4)
        feat = np.argmax(feat, axis=1)
        img_bp=''
        for bp in feat[:len_sub]:
            img_bp += dict_id2bp[bp]
        img_bp_index = 'ACCA'+img_bp+'AGGA'+img_bp+'ACGA'+img_bp+'AGCA'+img_bp
        list_pred_bps.append(img_bp_index)
    return list_pred_bps

def convert_bp2tensor(seq, len_max):
    list_bp_oh = []
    for bp in seq:
        list_bp_oh.append(dict_bp2onehot[bp])

    if len(list_bp_oh)<len_max:
        list_bp_oh = list_bp_oh + [[0,0,0,0]]*(len_max-len(list_bp_oh))
    elif len(list_bp_oh)>len_max:
        list_bp_oh = list_bp_oh[:len_max]
    seq_np = np.array(list_bp_oh).reshape(-1,4)
    seq_tensor = torch.from_numpy(seq_np).float()
    return seq_tensor



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset_img_rna(Dataset):
    def __init__(self, list_data, status='train'):
        super(Dataset_img_rna, self).__init__()
        self.list_data = list_data
        self.status = status
    
    def __getitem__(self, index):
        temp_data = self.list_data[index]
        # temp_data = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
        
        temp_data = get_img_pad(temp_data, pad_size, pad_size, pad_size, pad_size)
        # temp_data = np.expand_dims(temp_data, axis=2)
        temp_data = torch.from_numpy(temp_data).permute(2,0,1)
        data= {}
        data['input'] = temp_data.float()
        data['label'] = temp_data.float()
        return data

    def __len__(self):
        return len(self.list_data)

class Dataset_rna_seq(Dataset):
    def __init__(self, list_seq, len_seq):
        super(Dataset_rna_seq, self).__init__()
        self.list_seq = list_seq
        self.len_seq = len_seq

    def __getitem__(self, index):
        temp_seq = self.list_seq[index]
        try:
            seq_tensor = convert_bp2tensor(temp_seq, self.len_seq).to(device)
        except Exception as e:
            print(temp_seq)
        
        data= {}
        data['input'] = seq_tensor.float()
        return data

    def __len__(self):
        return len(self.list_seq)

class Dataset_index(Dataset):
    def __init__(self, list_data, index_len=48):
        super().__init__()
        self.list_data = list_data
        self.index_len = index_len

    def __getitem__(self, index):
        temp_seq = self.list_data[index][:self.index_len]

        seq_tensor = convert_bp2tensor(temp_seq, self.index_len)

        data= {}
        data['input'] = seq_tensor.float()
        return data

    def __len__(self):
        return len(self.list_data)

def acc_mute_rec(list_ori, list_rec):
    acc_num = 0
    for temp_p, temp_l in zip(list_ori, list_rec):
        if temp_p == temp_l:
            acc_num += 1
    
    print('Rec Acc:{:.4f}'.format(acc_num/len(list_ori)))

class Coder(DefaultCoder):
    def __init__(self, team_id: str = "none"):
        super().__init__(team_id=team_id)
        self.address, self.payload = 12, 128
        self.supplement, self.message_number = 0, 0

        self.model = Vae_mute_sim(max_len, crop_size, pad_size, num_sub).to(device)
        model_dir = './'+str(crop_size)+'_'+str(max_len)+'_'+str(num_sub)+'_'+ str(len_sub) +'/model_vae_mute.pb'
        self.model_index = mute_rec_cnn_new(len_sub_index, num_index)


        print(model_dir)
        try:
            self.model.load_state_dict(torch.load(model_dir, map_location='cpu'), strict=True)
        except Exception as e:
            print(e)
        self.model = self.model.to(device)
        self.model.eval()

        self.model_index.load_state_dict(torch.load('./index.pb', map_location='cpu'), strict=True)
        self.model_index = self.model_index.to(device)
        self.model_index.eval()

        self.index2seq = pickle.load(open('index2seq.pkl', 'rb'))
        self.seq2index = pickle.load(open('seq2index.pkl', 'rb'))



    def image_to_dna(self, input_image_path, need_logs=True):
        print('start convert image to seqences')
        expected_image = load_img(input_image_path)
        self.shape_ori = expected_image.shape

        list_img_packs = image_split(expected_image, crop_size, crop_size, True)
        data_test = Dataset_img_rna(list_img_packs, 'valid')
        loader_test = DataLoader(data_test, batch_size=400, shuffle=False, num_workers=2, drop_last=False)
        list_feats = []


        for i, data in enumerate(loader_test):
            input = data['input'].to(device)
            with torch.no_grad():
                feat = self.model.img2seq(input)[0]
            feat = feat.detach().cpu().numpy()
            list_feats.append(feat)
            if need_logs:
                self.monitor(i + 1, len(loader_test))
        feat_pred = np.concatenate(list_feats, axis=0)
        dna_sequences = convert_feat2bp(feat_pred)

        self.dna_seq_ori = dna_sequences

        print('num of dna seqs:', len(dna_sequences))
        dna_sequences_index = []
        self.dna_seq_index = []
        for i, seq in enumerate(dna_sequences):
            temp_index_seq = ''
            temp_index = self.index2seq[i]
            temp_index_seq = temp_index * num_index
            temp_seq = seq + temp_index_seq
            dna_sequences_index.append(temp_seq)
            self.dna_seq_index.append(temp_index_seq)
        
        return dna_sequences_index

    def dna_to_image(self, dna_sequences, output_image_path, need_logs=True):
        print('start recovery image from seqences')
        dna_sequences_index = dna_sequences.copy()
        dna_sequences = []
        dna_index = []
        for i, dna_seq in enumerate(dna_sequences_index):
            dna_sequences.append(dna_seq[:max_len])
            dna_index.append(dna_seq[max_len:])
        

        ## rec index
        dataset_index = Dataset_index(dna_index, index_len=len_index)
        loader_index = DataLoader(dataset_index, batch_size=400, shuffle=False, drop_last=False)
        list_preds = []
        for i, data in enumerate(tqdm(loader_index, ncols=100)):
            input = data['input'].to(device)
            # print(input.shape)
            with torch.no_grad():
                pred_rec = self.model_index(input)
            temp_pred = torch.argmax(pred_rec, dim=-1).detach().cpu().numpy().tolist()
            list_preds += temp_pred

        index_recs = []
        for temp_pred in list_preds:
            temp_seq = ''
            for temp_pred_id in temp_pred:
                temp_seq += dict_id2bp[temp_pred_id]
            index_recs.append(temp_seq)
        print('test index rec:')
        acc_mute_rec(self.dna_seq_index, index_recs)


        ## test
        list_index_guess = []
        for temp_seq_guess in index_recs:
            try:
                list_index_guess.append(self.seq2index[temp_seq_guess[:len_sub_index]])
            except Exception as e:
                # print(e)
                list_index_guess.append(0)
        
        dna_sequences_guess = [0]*len(dna_sequences)
        for index, seq in zip(list_index_guess, dna_sequences):
            try:
                dna_sequences_guess[index] = seq
            except Exception as e:
                # print(e)
                dna_sequences_guess[0] = seq
        
        dna_sequences = []
        for i, seq in enumerate(dna_sequences_guess):
            if seq != 0:
                dna_sequences.append(seq)
            else:
                dna_sequences.append(dna_sequences_guess[-2])


        list_img_packs_pred = []
        data_test = Dataset_rna_seq(dna_sequences, max_len)
        loader_test_s2i = DataLoader(data_test, batch_size=400, shuffle=False, drop_last=False)

        ## test
        list_preds = []
        for i, data in enumerate(tqdm(loader_test_s2i, ncols=100)):
            input = data['input'].to(device)
            with torch.no_grad():
                pred_rec = self.model.dna_mute(input)
            temp_pred = torch.argmax(pred_rec, dim=-1).detach().cpu().numpy().tolist()
            list_preds += temp_pred

        dna_sequences_rec = []
        for temp_pred in list_preds:
            temp_seq = ''
            for temp_pred_id in temp_pred:
                temp_seq += dict_id2bp[temp_pred_id]
            dna_sequences_rec.append(temp_seq)
        print('test seq rec:')
        acc_mute_rec(self.dna_seq_ori, dna_sequences_rec)
        ## test



        for i, data in enumerate(tqdm(loader_test_s2i, ncols=100)):
            input = data['input'].to(device)
            with torch.no_grad():
                out = self.model.generate_mute(input)[-1]

            out = out.detach().permute(0,2,3,1).cpu().numpy()
            list_img_packs_pred.append(out)
            # if need_logs:
            #     self.monitor(i + 1, len(loader_test_s2i))
        img_packs_pred = np.concatenate(list_img_packs_pred, axis=0)
        obtained_image = image_merge(img_packs_pred, crop_size, crop_size, 8000//crop_size, 4000//crop_size)
        h, w = self.shape_ori[:2]
        obtained_image = cv2.resize(obtained_image, (w, h))
        # obtained_image = process_post(obtained_image)
        save_img(output_image_path, obtained_image)