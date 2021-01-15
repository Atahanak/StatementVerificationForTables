import torch

class StatementVerificationWithTablesDataset(torch.utils.data.Dataset):
    ''' Statement Verification From Tables Dataset '''

    def __init__(self, encodings, labels):
        #self.root_dir = root_dir
        #self.transform = transform 
        print(labels)
        print(len(encodings['input_ids']), len(labels))
        assert len(encodings['input_ids']) == len(labels)
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        sample = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample


class TableDataset(torch.utils.data.Dataset):
    ''' SequenceClassification Tables '''
    def __init__(self, tables, statements, labels, tokenizer):
        self.tables = tables
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            table= self.tables[idx],
            queries=  self.statements[idx],
            truncation= True,
            padding= "max_length",
            return_tensors= "pt",
            #label=  self.labels[idx]
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["label"] = self.labels[idx]
        return encoding

    def __len__(self):
        return len(self.labels)

