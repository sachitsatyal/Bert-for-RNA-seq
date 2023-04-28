import logging
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)

class RNADataset(Dataset):
    """
    A custom dataset for RNA sequences.
    """
    def __init__(self,model_name,rna_df):

        
        self.model_name = model_name
        self.rna_df=rna_df
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    

    def __len__(self):
        sequences = self.rna_df['Sequences']
        return len(sequences)

    def __getitem__(self, idx):
      
        sequence = self.rna_df['Sequences'][idx]
        structure = self.rna_df['Structures'][idx]
        encoding = self.tokenizer(sequence, structure, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return encoding["input_ids"][0], encoding["attention_mask"][0]




if __name__ == '__main__':


    sequences=[' CGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAAACUCUUGAUUAUGAAGUCUGUCGCUUUAUCCGAAAUUUUAUAAAGAGAAGACUCAUGAAU','XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX']
    structures=['..(((((((...((((((.........))))))........((((((.......))))))..))))))).......((((((..........))))))...............','..........((((((.(((.((.(.(((...))).).)).)))...))).((((..((((....))))....))))))).........................']

    sequences.append('UUAUAGGCGAUGGAGUUCGCCAUAAACGCUGCUUAGCUAAUGACUCCUACCAGUAUCACUACUGGUAGGAGUCUAUUUUUUUAGGAGGAAGGAUCUAUGA')
    structures.append('....................................................................................................')


    rna_df=pd.DataFrame({'Sequences':sequences,'Structures':structures})

    # create a dataloader for the RNA sequences
    dataset = RNADataset('Rostlab/prot_bert_bfd',rna_df)
    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        print(f'input_ids {input_ids}')
        print(f'attention_mask {attention_mask}')
        
