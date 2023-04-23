import logging
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from rdatkit import io

logging.basicConfig(level=logging.INFO)

class RNASeqDataset(Dataset):
    """
    A custom dataset for RNA sequences.
    """
    def __init__(self, data_file, file_type, model_name):
        """
        Initializes the dataset.
        :param data_file: The file containing the RNA sequences.
        :param file_type: The type of file (rdat, parquet, pickle).
        :param model_name: The name of the model to use for tokenization.
        """
        self.data_file = data_file
        self.file_type = file_type
        self.model_name = model_name
        self._load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _load_data(self):
        """
        Loads the data from the file.
        """
        if self.file_type == 'rdat':
            rdat_file = io.RDATFile(self.data_file)
            self.seqs = [r.seq for r in rdat_file.sequences]
        elif self.file_type == 'parquet':
            self.seqs = pd.read_parquet(self.data_file)
        elif self.file_type == 'pickle':
            with open(self.data_file, 'rb') as f:
                self.seqs = pickle.load(f)
        else:
            logging.error('Unsupported file type: %s', self.file_type)
            raise ValueError('Unsupported file type:', self.file_type)

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
The model to use is Rostlab/pro_bert_bfd
data = RNASeqDataset('/data/rmdb/ADD125_STD_0001.rdat', 'rdat', 'Rostlab/prot_bert_bfd')
loader = data.get_loader(batch_size=32, shuffle=True)

for input_ids, attn_mask in loader:
Feed them to transformers model
    """
