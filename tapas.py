import sys, os
import torch
import xmltodict
from raw_data import TableForVerification

def get_tables_from_xml(xml_file_name, tables):
    ''' dump xml to dictionary '''
    with open(xml_file_name) as file:
        doc = xmltodict.parse(file.read())

    if 'document' in doc and doc['document'] is not None:
        if 'table' in doc['document']:
            table = doc['document']['table']
            if type(table) == list:
                for t in table:
                    tables.append(TableForVerification(t))
            else:
                tables.append(TableForVerification(table))

data_dir = sys.argv[1]
tables = []
samples = []
labels = [] 
file_name = sys.argv[1]
    
for filename in os.listdir(data_dir):
    if filename.endswith(".xml"): 
        file_path = data_dir + filename
        #print(file_path)
        get_tables_from_xml(file_path, tables) 
    else:
        continue

text_tables = []
statements = []
labels = []
for table in tables:
    table.populate_tables_statements_labels(text_tables, statements, labels)

from transformers import TapasTokenizer
from dataset import TableDataset
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
train_dataset = TableDataset(text_tables, statements, labels, tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

from transformers import TapasConfig, AdamW, TapasForSequenceClassification
model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")
optimizer = AdamW(model.parameters(), lr=5e-5) 

#TRAINING


#INFERENCE
batch = next(iter(train_dataloader))
assert batch["input_ids"].shape == (4, 512)
assert batch["attention_mask"].shape == (4, 512)
assert batch["token_type_ids"].shape == (4, 512, 7)
tokenizer.decode(batch["input_ids"][0])

from datasets import load_metric
accuracy = load_metric("accuracy")

print("Starting evaluation...")
number_processed = 0
total = len(train_dataloader) * batch["input_ids"].shape[0] # number of batches * batch_size

for batch in train_dataloader:
    # get the inputs
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]
    labels = batch["label"]

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
    model_predictions = outputs.logits.argmax(-1)

    # add metric
    accuracy.add_batch(predictions=model_predictions, references=labels)

    number_processed += batch["input_ids"].shape[0]
    print(f"Processed {number_processed} / {total} examples")

final_score = accuracy.compute()
print(final_score)
