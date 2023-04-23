import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from rdatkit import io

class RNASeqDataset(Dataset):
    def __init__(self, data_file, file_type, model_name):
        if file_type == 'rdat':
            rdat_file = io.RDATFile(data_file)
            self.seqs = [r.seq for r in rdat_file.sequences]
        elif file_type == 'parquet':
            self.seqs = pd.read_parquet(data_file)
        elif file_type == 'pickle':
            with open(data_file, 'rb') as f:
                self.seqs = pickle.load(f)
        else:
            raise ValueError('Unsupported file type:', file_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        encoding = self.tokenizer(seq, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids']
        attn_mask = encoding['attention_mask']
        return input_ids, attn_mask

    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


"""
data = RNASeqDataset('rna_sequences.rdat', 'rdat', 'Rostlab/prot_bert_bfd')
loader = data.get_loader(batch_size=32, shuffle=True)

for input_ids, attn_mask in loader:
    # do something with the input IDs and attention mask
    print(f'Input IDs: {input_ids}')
    print(f'Attention Mask: {attn_mask}')"""
