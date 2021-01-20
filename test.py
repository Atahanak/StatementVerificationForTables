import torch
import sys, os
import xmltodict
from raw_data import TableForVerification
from dataset import TableDataset

def get_tables_from_xml(xml_file_name, tables, meta_data):
    ''' dump xml to dictionary '''
    with open(xml_file_name) as file:
        doc = xmltodict.parse(file.read())

    if 'document' in doc and doc['document'] is not None:
        if 'table' in doc['document']:
            table = doc['document']['table']
            if type(table) == list:
                for t in table:
                    tables.append(TableForVerification(t))
                    meta_data["file_names"].append(xml_file_name)
            else:
                tables.append(TableForVerification(table))
    return doc

xmls = {}
def get_dataset(data_dir):
    tables = []
    meta_data = {
        "file_names": [],
        "table_ids": [],
        "statement_ids": []
    } 
    for filename in os.listdir(data_dir):
        if filename.endswith(".xml"): 
            file_path = data_dir + filename
            xml = get_tables_from_xml(file_path, tables, meta_data) 
            xmls[filename] = xml
        else:
            continue
        
    text_tables = []
    statements = []
    for table in tables:
        table.populate_tables_statements(text_tables, statements, meta_data)
    
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
    return TableDataset(text_tables, statements, None, tokenizer, meta_data)

def save(model, filename, epoch):
        params = {
            'state_dict': {
                'network': model.state_dict(),
            },
            'epoch': epoch
        }
        torch.save(params, filename)

def resume(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    state_dict = checkpoint['state_dict']

    new_state = set(model.state_dict().keys())
    for k in list(state_dict['network'].keys()):
        if k not in new_state:
            del state_dict['network'][k]
    model.load_state_dict(state_dict['network'])
    model.to(device)
    return model

test_dir = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]

from transformers import TapasTokenizer
from dataset import TableDataset
#test set
test_dataset = get_dataset(test_dir) #TableDataset(text_tables, statements, labels, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

from transformers import TapasConfig, AdamW, TapasForSequenceClassification
model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resume(model, model_path, device)
model.to(device)

print("Starting evaluation...")
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
    print(batch["table_id"])
    print(batch["file_name"])
    print(batch["statement_id"])

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    model_predictions = outputs.logits.argmax(-1)
    print(model_predictions)

    print(f"TESTING: Processed {number_processed} / {total} examples")
    number_processed += batch["input_ids"].shape[0]