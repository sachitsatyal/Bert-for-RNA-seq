import numpy as np
import pandas as pd
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from typing import List
import rdatkit
from transformers import AutoTokenizer
from torch.utils.data import Subset

class RNATokenizer:
    """
    Tokenizer class for RNA sequence to numerical representation conversion.
    """
    
    def __init__(self, vocab_file: str = None, max_length: int = 512, special_tokens: dict = None):
        """
        Initializes the Tokenizer object.
        
        Args:
        - vocab_file (str): Optional path to a pre-existing vocabulary file.
        - max_length (int): Maximum sequence length.
        - special_tokens (dict): Optional dictionary of special tokens to add to the vocabulary.
        """
        
        self.max_length = max_length
        self.special_tokens = special_tokens or {}
        self.tokenizer = None
        
        if vocab_file is None:
            self.vocab = self.build_vocab()
            self.save_vocab("vocab.txt")
        else:
            self.vocab = self.load_vocab(vocab_file)
        
        self.tokenizer = self.build_tokenizer()
        
    def build_vocab(self, sequences: List[str]):
        """
        Builds a vocabulary dictionary from RNA sequences.
        """
        
        # Collect unique nucleotides in the sequences
        unique_nucleotides = set("".join(sequences))
        
        # Build vocabulary from unique nucleotides and special tokens
        vocab = {}
        vocab.update(self.special_tokens)
        for i, nuc in enumerate(unique_nucleotides):
            vocab[nuc] = i + len(self.special_tokens)
            
        return vocab
        
    def save_vocab(self, file_path: str):
        """
        Saves the vocabulary to a file.
        """
        
        with open(file_path, "w") as f:
            for token, index in self.vocab.items():
                f.write(f"{token}\t{index}\n")
                
    def load_vocab(self, file_path: str):
        """
        Loads the vocabulary from a file.
        """
        
        vocab = {}
        with open(file_path, "r") as f:
            for line in f:
                token, index = line.strip().split("\t")
                vocab[token] = int(index)
                
        return vocab
    
    def build_tokenizer(self):
        """
        Builds a tokenizer from the vocabulary.
        """
        
        from transformers import PreTrainedTokenizerFast
        
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=None,
            vocab_file=None,
            model_max_length=self.max_length,
            special_tokens=self.special_tokens
        )
        
        tokenizer.get_vocab = lambda: self.vocab
        
        return tokenizer
    
    def encode(self, sequence: str):
        """
        Encodes an RNA sequence into numerical representation.
        """
        
        return self.tokenizer.encode(sequence, add_special_tokens=False)
    
    def batch_encode_plus(self, sequences: List[str]):
        """
        Encodes a list of RNA sequences into a BatchEncoding object.
        """
        
        # Tokenize sequences
        tokenized_sequences = [self.encode(seq) for seq in sequences]
        
        # Pad sequences to max length
        padded_sequences = self.tokenizer.pad(tokenized_sequences, padding=True, truncation=True, max_length=self.max_length)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_sequences["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(padded_sequences["attention_mask"], dtype=torch.long)
        
        # Return BatchEncoding object
        return {"input_ids": input_ids, "attention_mask": attention_mask}



class RNADataset(Dataset):
    """
    Dataset class for RNA structure prediction.
    """
    
    def __init__(self, sequences: List[str], structures: List[str], tokenizer, split: dict = None):
        """
        Initializes the RNADataset object.
        
        Args:
        - sequences (List[str]): List of RNA sequences.
        - structures (List[str]): List of RNA secondary structures.
        - tokenizer: Tokenizer object.
        - split (dict): Dictionary of split ratios for training, validation, and testing sets.
        """
        
        self.sequences = sequences
        self.structures = structures
        self.tokenizer = tokenizer
        
        # Split dataset into train, val, and test sets
        if split is None:
            self.split = {"train": 0.8, "val": 0.1, "test": 0.1}
        else:
            self.split = split
            
        self.indices = self.split_indices()
        
    def split_indices(self):
        """
        Splits dataset indices into train, val, and test sets.
        """
        
        num_samples = len(self.sequences)
        num_train = int(num_samples * self.split["train"])
        num_val = int(num_samples * self.split["val"])
        num_test = num_samples - num_train - num_val
        
        indices = np.arange(num_samples)
        # np.random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train+num_val]
        test_indices = indices[num_train+num_val:]
        
        return {"train": train_indices, "val": val_indices, "test": test_indices}
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        
        # return len(self.indices["train"])
        return len(self.sequences)
    
    def __getitem__(self, index):
     
         # Get sequence and structure
         sequence = self.sequences[index]
         structure = self.structures[index]

         # Tokenize sequence
         inputs = self.tokenizer.batch_encode_plus([sequence])

         input_ids = torch.as_tensor(inputs["input_ids"]).squeeze()
         attention_mask = torch.as_tensor(inputs["attention_mask"]).squeeze()

         # Convert structure to SMILES
         # This should be implemented separately, depending on the format of the structure data

         # Prepare output
         output = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": structure}

         return output
    
    def get_data_loader(self, split: str, batch_size: int = 32, shuffle: bool = True):
        """
        Returns a DataLoader object for a specified dataset split.
        """
        
        indices = self.indices[split]
        dataset = Subset(self, indices)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return data_loader




class RNALoader(DataLoader):
    """
    DataLoader class for RNA structure prediction.
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None):
        """
        Initializes the RNALoader object.
        
        Args:
        - dataset: RNADataset object.
        - batch_size (int): Batch size for the DataLoader.
        - shuffle (bool): If True, shuffle the data for each epoch.
        - collate_fn: Optional function to collate samples into batches.
        """
        
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        
    def collate_fn(self, batch):
        """
        Collate function to create batches from samples.
        
        Args:
        - batch: List of samples from the dataset.
        
        Returns:
        - Dictionary of features and labels for the batch.
        """
        
        # Get input IDs, attention masks, and labels for each sample in the batch
        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        
        # Pad input IDs and attention masks to the same length
        max_length = max([len(ids) for ids in input_ids])
        padded_input_ids = [ids + [0]*(max_length-len(ids)) for ids in input_ids]
        padded_attention_mask = [mask + [0]*(max_length-len(mask)) for mask in attention_mask]
        
        # Convert everything to Tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        # Create dictionary of features and labels for the batch
        batch_dict = {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor, "labels": labels_tensor}
        
        return batch_dict
    






def train(model, train_loader, optimizer, criterion):
    """
    Train function for the RNA structure prediction model.
    """
    
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss/len(train_loader)

def evaluate(model, val_loader, criterion):
    """
    Evaluate function for the RNA structure prediction model.
    """
    
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
    return running_loss/len(val_loader)
"""
if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize tokenizer and device
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
   
    first='../../data/rmdb/ADD125_STD_0001.rdat'
    second='../../data/rmdb/ETRUNM_VN1_0001.rdat'
    third='../../data/rmdb/FLUORSW_BZCN_0028.rdat'
    files=[first,second,third]
    sequences=[]
    structures=[]
    def load_rna(): 
        for file in files:
            # create RDATFile instance
            rdat = rdatkit.RDATFile()
            # load RDAT file
            rdat.load(open(file, 'r'))



            construct_name= list(rdat.constructs.keys())[0]
            print(construct_name)

            # print(
            
            # np.array(rdat.values[construct_name]).shape

            # )



            rdat_sec = rdat.constructs[construct_name]
            sequence=rdat_sec.sequence
            sequences.append(sequence)
            structure=rdat_sec.structure
            structures.append(structure)
       


    load_rna()
   
    
    seqs=['AUGAGUCCUGGUUUGCCAGUUU','GGGAGCUUCGUAGCCUGGGUGG','GGGCCUGAGACCCUGUUCACAGGGC',' UUCGUUUUAGUUACGUUGUCUU']
    structs=['((((((.....))))))',' ((((((.....))))))','(((((((...))))))))',' ((((((.....))))))']
    sequences.extend(seqs)
    structures.extend(structs)
    seq='AUGAGUCCUGGUUUGCCAGUUU'
    struct='((((((.....))))))'

    for i in range(1000):
        sequences.append(seq)
        structures.append(struct)

    if len(sequences) != len(structures):
        raise ValueError("Sequences and structures lists have different lengths")
    dataset = RNADataset(sequences, structures, tokenizer)
  
    if len(dataset) != len(sequences):
        raise ValueError("Dataset size does not match input size")
    
    train_indices = dataset.indices["train"]
    val_indices = dataset.indices["val"]

    if len(val_indices) == 0:
        raise ValueError("Validation set is empty")
    test_indices = dataset.indices["test"]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = RNALoader(train_dataset, batch_size=16)
    val_loader = RNALoader(val_dataset, batch_size=16)
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")
    test_loader = RNALoader(test_dataset, batch_size=16)
    for batch in test_loader:
        print(batch)
    

    model = None # Replace with your own model
    
    # Set optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = None
    
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Evaluate the model on the test set
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}")"""
"""