import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

from models import *

from random import seed, shuffle, random, randint, choice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pickle


from coder_new import num_index, len_sub_index
len_index = num_index*len_sub_index

from coder_new import max_len


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


def mutate(source_dna_sequence):
    mutate_number = int(0.0150 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)
    insert_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)
    delete_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)
    # mutate_number = choice([1,2])
    # insert_number = choice([1])
    # delete_number = choice([1])
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
        temp_seq, temp_seq_m = self.list_data[index]

        # if random() < 0.5 and self.status == 'train':
        #     temp_seq = temp_seq[::-1]


        temp_bp = choice(['A','G','T','C'])*max_len
        temp_seq = temp_bp + temp_seq
        if self.status == 'train':
            if random() < 0.99:
                temp_seq_mute = mutate(temp_seq)
            else:
                temp_seq_mute = temp_seq
            
        else:
            temp_seq_mute = mutate(temp_seq)
        
        temp_seq = temp_seq[max_len:]
        temp_seq_mute = temp_seq_mute[max_len:]

        seq_tensor = convert_bp2tensor(temp_seq, len_index)
        seq_label = torch.argmax(seq_tensor, dim=-1)
        seq_mute_tensor = convert_bp2tensor(temp_seq_mute, len_index)
        
        data= {}
        data['seq'] = seq_tensor.float()
        data['seq_mute'] = seq_mute_tensor.float()
        data['seq_label'] = seq_label.long()
        return data

    def __len__(self):
        return len(self.list_data)

def convert_bp2tensor(seq, max_len=48):
    list_bp_oh = []
    for bp in seq:
        list_bp_oh.append(dict_bp2onehot[bp])

    if len(list_bp_oh)<max_len:
        list_bp_oh = list_bp_oh + [[0,0,0,0]]*(max_len-len(list_bp_oh))
    elif len(list_bp_oh)>max_len:
        list_bp_oh = list_bp_oh[:max_len]
    seq_np = np.array(list_bp_oh).reshape(-1,4)
    seq_tensor = torch.from_numpy(seq_np).float()
    return seq_tensor

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
    compatibility_score = h_score + c_score
    return compatibility_score

def train_mute_index():
    batch_size = 2000
    epoch_max = 300
    lr = 1e-3
    num_workser = 4

    
    list_data = []
    data_index = pickle.load(open('index2seq.pkl', 'rb'))
    list_temp = []
    # for k, v in data_index.items():
    for k in range(320000):
        v = data_index[k]
        
        temp_seq = v*num_index
        list_data.append([temp_seq, temp_seq])
        list_temp.append(temp_seq)
    
    list_temp_set = set(list_temp)


    data_train = Dataset_seq_mute(list_data[:320000], scale=10)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workser, drop_last=True)

    data_valid = Dataset_seq_mute(list_data[:320000])
    loader_valid = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    

    model = mute_rec_cnn_new(len_sub_index, num_index).to(device)
    model.load_state_dict(torch.load('./index.pb'))


    loss_fun_l1 = nn.L1Loss(reduction='sum')
    loss_fun_ce = nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*1//2, epoch_max*8//10])

    for epoch in range(epoch_max):
        model.train()
        list_preds = []
        list_labels = []
        with tqdm(loader_train, ncols=150) as tqdmDataLoader:
            for i, data in enumerate(tqdmDataLoader):
                # temp_id = 2000+i%30
                # seed(temp_id)


                seq = data['seq'].to(device)
                seq_mute = data['seq_mute'].to(device)
                seq_label = data['seq_label'].to(device)
                    
                
                pred_rec = model(seq_mute)

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

        acc_seq(list_preds[-100000:], list_labels[-100000:])

        model.eval()
        torch.save(model.state_dict(), './index.pb')
        lr_sh.step()


        list_preds = []
        list_labels = []
        for data in tqdm(loader_valid, ncols=100):
            seq = data['seq'].to(device)
            seq_mute = data['seq_mute'].to(device)
            seq_label = data['seq_label'].to(device)
                
            with torch.no_grad():
                pred_rec = model(seq_mute)

            temp_pred = torch.argmax(pred_rec, dim=-1).detach().cpu().numpy().tolist()
            temp_label = torch.argmax(seq, dim=-1).detach().cpu().numpy().tolist()
            list_preds += temp_pred
            list_labels += temp_label
        acc_seq(list_preds, list_labels)
        print('-----------------------------------------------------------')

        list_seqs = []
        for pred in list_labels:
            temp_seq = ''
            for bp in pred:
                temp_seq += dict_id2bp[bp]
            list_seqs.append(temp_seq)
        
        metric_gc(list_seqs)


if __name__ == '__main__':
    train_mute_index()