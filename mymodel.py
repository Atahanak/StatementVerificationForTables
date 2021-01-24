import sys, os
import torch
import xmltodict
from raw_data import TableForVerification

def train(model, optimizer, train_dataset, epochs):
    for ep in range(0, epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        batch = next(iter(train_dataloader))
        assert batch["input_ids"].shape == (4, 512)
        assert batch["attention_mask"].shape == (4, 512)
        assert batch["token_type_ids"].shape == (4, 512, 7)
        number_processed = 0
        total = len(train_dataloader) * batch["input_ids"].shape[0] # number of batches * batch_size
        print(f"Epoch: {ep}")
        for batch in train_dataloader:
            # get the inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            # initialize model parameters
            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            #model_predictions = outputs.logits.argmax(-1)

            # backward pass
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"TRAINING: Processed {number_processed} / {total} examples")
            number_processed += batch["input_ids"].shape[0]

def get_tables_from_xml(xml_file_name, tables, dtype=None):
    ''' dump xml to dictionary '''
    with open(xml_file_name) as file:
        doc = xmltodict.parse(file.read())

    if 'document' in doc and doc['document'] is not None:
        if 'table' in doc['document']:
            table = doc['document']['table']
            if type(table) == list:
                for t in table:
                    tables.append(TableForVerification(t, dtype))
            else:
                tables.append(TableForVerification(table, dtype))

def get_dataset(data_dir, tt=None, dtype=None):
    tables = []
    for filename in os.listdir(data_dir):
        if filename.endswith("52.xml"): 
            file_path = data_dir + filename
            get_tables_from_xml(file_path, tables, dtype) 
        else:
            continue
    print("TTT", len(tables)) 
    text_tables = []
    statements = []
    labels = []
    for table in tables:
        if tt == None:
          table.populate_tables_statements_labels(text_tables, statements, labels)
        else:
          table.populate_tables_statements_labels_tt(text_tables, statements, labels)
          
    print(labels)
    print(len(labels)) 
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
    return TableDataset(text_tables, statements, labels, tokenizer)

train_dir = sys.argv[1]
test_dir = sys.argv[2]

from transformers import TapasTokenizer
from dataset import TableDataset
#train set
train_dataset = get_dataset(train_dir, True, "un") #TableDataset(text_tables, statements, labels, tokenizer)

from transformers import TapasConfig, AdamW, TapasForSequenceClassification
model = TapasForSequenceClassification.from_pretrained("google/tapas-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5) 

#TRANING
print("Starting training...")

epochs = 1
train(model, optimizer, train_dataset, epochs)        

train_dataset2 = get_dataset(train_dir, None, "enre")
model2 = TapasForSequenceClassification.from_pretrained("google/tapas-base")
model2.to(device)
optimizer2 = AdamW(model2.parameters(), lr=3e-5) 
train(model2, optimizer2, train_dataset2, epochs)        

print("Starting evaluation...")
#test set
test_dataset = get_dataset(test_dir) #TableDataset(text_tables, statements, labels, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

from datasets import load_metric
accuracy = load_metric("accuracy")

batch = next(iter(test_dataloader))
assert batch["input_ids"].shape == (1, 512)
assert batch["attention_mask"].shape == (1, 512)
assert batch["token_type_ids"].shape == (1, 512, 7)
number_processed = 0
total = len(test_dataloader) * batch["input_ids"].shape[0] # number of batches * batch_size

for batch in test_dataloader:
    # get the inputs
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch["label"]

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    model_predictions = outputs.logits.argmax(-1)

    print(model_predictions, labels)
    if model_predictions[0] == 1: #enough information
        # forward pass
        outputs = model2(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        model_predictions = outputs.logits.argmax(-1)
        accuracy.add_batch(predictions=model_predictions, references=labels)
    else: #unkown
        accuracy.add_batch(predictions=[2], references=labels)

    print(f"TESTING: Processed {number_processed} / {total} examples")
    number_processed += batch["input_ids"].shape[0]

final_score = accuracy.compute()
print(final_score)