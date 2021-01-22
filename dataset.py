import torch

class StatementVerificationWithTablesDataset(torch.utils.data.Dataset):
    ''' Statement Verification From Tables Dataset '''

    def __init__(self, encodings, labels, meta_data=None):
        #self.root_dir = root_dir
        #self.transform = transform 
        print(labels)
        print(len(encodings['input_ids']), len(labels))
        assert len(encodings['input_ids']) == len(labels)
        self.encodings = encodings
        self.labels = labels
        self.meta_data = meta_data

    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        sample = {key: val[idx].squeeze(0) for key, val in self.encodings.items()}
        if self.labels[idx] != None:
            sample['labels'] = torch.tensor(self.labels[idx])
        if self.meta_data != None:
            sample["file_name"] = self.meta_data["file_names"][idx]
            sample["table_id"] = self.meta_data["table_ids"][idx]
            sample["statement_id"] = self.meta_data["statement_ids"][idx]
        return sample


class TableDataset(torch.utils.data.Dataset):
    ''' SequenceClassification Tables '''
    def __init__(self, tables, statements, labels, tokenizer, meta_data=None):
        self.tables = tables
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.meta_data = meta_data

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
        if self.labels != None:
            encoding["label"] = self.labels[idx]
        if self.meta_data != None:
            encoding["file_name"] = self.meta_data["file_names"][idx]
            encoding["table_id"] = self.meta_data["table_ids"][idx]
            encoding["statement_id"] = self.meta_data["statement_ids"][idx]
        return encoding

    def __len__(self):
        if self.labels == None:
            return len(self.statements)
        else:
            return len(self.labels)

