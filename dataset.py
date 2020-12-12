import torch

class StatementVerificationWithTablesDataset(torch.utils.data.Dataset):
    ''' Statement Verification From Tables Dataset '''

    def __init__(self, encodings, labels):
        #self.root_dir = root_dir
        #self.transform = transform 
        assert len(encodings['input_ids']) == len(labels)
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        sample = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample