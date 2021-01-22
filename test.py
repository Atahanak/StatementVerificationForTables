import torch
import sys, os
import json
import xmltodict
from dicttoxml import dicttoxml
from raw_data import TableForVerification
from dataset import TableDataset
from dataset import StatementVerificationWithTablesDataset
from transformers import BertTokenizer

label_map = {
    "0": "refuted",
    "1": "entailed",
    "2": "unknown"
}

def get_tables_from_xml(xml_file_name, tables):
    ''' dump xml to dictionary '''
    with open(xml_file_name) as file:
        doc = xmltodict.parse(file.read(), attr_prefix='-')

    if 'document' in doc and doc['document'] is not None:
        if 'table' in doc['document']:
            table = doc['document']['table']
            if type(table) == list:
                for t in table:
                    tables.append(TableForVerification(t, xml_file_name.split("/")[2]))
            else:
                tables.append(TableForVerification(table, xml_file_name.split("/")[2]))
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
            xml = get_tables_from_xml(file_path, tables) 
            xmls[filename] = xml
        else:
            continue
        
    samples = []
    labels = []
    for table in tables:
        temp = table.get_samples_and_labels(meta_data)
        samples += temp['samples']
        labels += temp['labels']

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = tokenizer(
        samples, 
        padding = True,
        truncation = True,
        return_tensors="pt"
    )
    return StatementVerificationWithTablesDataset(data, labels, meta_data)

def get_dataset_tapas(data_dir):
    tables = []
    meta_data = {
        "file_names": [],
        "table_ids": [],
        "statement_ids": []
    } 
    for filename in os.listdir(data_dir):
        if filename.endswith(".xml"): 
            file_path = data_dir + filename
            xml = get_tables_from_xml(file_path, tables) 
            xmls[filename] = xml
        else:
            continue
        
    text_tables = []
    statements = []
    for table in tables:
        table.populate_tables_statements(text_tables, statements, meta_data)
    
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
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

def populate_result_xml(file_name, table_id, statement_id, label, xmls):
    doc = xmls[file_name]
    if 'table' in doc['document']:
        table = doc['document']['table']
        if type(table) == list:
            for t in table:
                if table_id == t['-id']:
                    if 'statements' in t: 
                        if 'statement' in t['statements']:
                            if type(t['statements']['statement']) == list:
                                for s in t['statements']['statement']:
                                    if s["-id"] == statement_id:
                                        s["-type"] = label
                            else:
                                if t['statements']['statement']["-id"] == statement_id:
                                    t['statements']['statement']["-type"] = label
        else:
            if table_id == table['-id']:
                if 'statements' in table: 
                    if 'statement' in table['statements']:
                        if type(table['statements']['statement']) == list:
                            for s in table['statements']['statement']:
                                if s["-id"] == statement_id:
                                    s["-type"] = label
                        else:
                            if table['statements']['statement']["-id"] == statement_id:
                                table['statements']['statement']["-type"] = label

test_dir = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]

from transformers import TapasTokenizer
from dataset import TableDataset
#test set
test_dataset = get_dataset(test_dir) #TableDataset(text_tables, statements, labels, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

from transformers import TapasConfig, AdamW, TapasForSequenceClassification
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resume(model, model_path, device)
model.to(device)

print("Starting evaluation...")
batch = next(iter(test_dataloader))
#assert batch["input_ids"].shape == (1, 512)
#assert batch["attention_mask"].shape == (1, 512)
#assert batch["token_type_ids"].shape == (1, 512, 7)
number_processed = 0
total = len(test_dataloader) * batch["input_ids"].shape[0] # number of batches * batch_size

##print(xmls)
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
    ##outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    model_predictions = outputs.logits.argmax(-1)
    res = model_predictions.tolist()
    print("Prediction", res[0])
    populate_result_xml(batch["file_name"][0], batch["table_id"][0], batch["statement_id"][0], label_map[str(res[0])], xmls) 

    print(f"TESTING: Processed {number_processed} / {total} examples")
    number_processed += batch["input_ids"].shape[0]

#print(json.dumps(xmls, indent=4, sort_keys=True))

for key in xmls:
    #convert to xml
    temp = xmls[key]#['document']
    xml = json.dumps(temp, indent=4, sort_keys=True)#dicttoxml(temp, custom_root="document")
        
    # write to a file
    f = open(output_path+key, "w")
    f.write(xml)#.decode())
    #print(xml) 