import logging
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from rdatkit import io, structure

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
            try:
                rdat_data = io.load_rdat(self.data_file)
            except Exception as e:
                logging.error(f'Error loading RDAT file: {e}')
                raise

            # Create empty lists to store the sequences and structures
            sequences = []
            structures = []

            # Loop through all the RNA sequences in the RDAT file
            for i in range(len(rdat_data.sequences)):
                # Extract the RNA sequence
                rna_sequence = rdat_data.sequences[i]
                sequences.append(rna_sequence)

                # Extract the RNA structure
                rna_structure = rdat_data.annotations[i].structure

                # Handle missing values in the RNA structure
                filled_structure = structure.fill_in_structure(rna_structure)

                # Append the filled structure to the list of structures
                structures.append(filled_structure)

            # Create a pandas dataframe with the sequences and structures as columns
            self.rna_df = pd.DataFrame({'Sequence': sequences, 'Structure': structures})
        else:
            logging.error('Unsupported file type: %s', self.file_type)
            raise ValueError('Unsupported file type:', self.file_type)

    def __len__(self):
        sequences = self.rna_df['Sequence']
        return len(sequences)

    def __getitem__(self, idx):
        seq = self.rna_df['Sequence'][idx]

        encoding = self.tokenizer(seq, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids']
        attn_mask = encoding['attention_mask']
        return input_ids, attn_mask

    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    model_name = 'Rostlab/prot_bert_bfd'
    data = RNASeqDataset('/data/rmdb/ADD125_STD_0001.rdat', 'rdat', model_name)
    loader = data.get_loader(batch_size=32, shuffle=True)

    for input_ids, attn_mask in loader:
        # Feed them to transformers model
        pass
