''' transformer example '''
import sys
import os
import xmltodict

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataset import StatementVerificationWithTablesDataset
from sklearn.model_selection import train_test_split

from raw_data import TableForVerification

def compute_metrics(pred):
    ''' detailed metrics for evaluation '''
    print(pred)
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    #print(f"acc: {acc}, f1: {f1}, precision: {precision}, recall: {recall}")
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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

if __name__ == "__main__":
    
    ############################### 
    #     DATA PREPROCESSING      # 
    ############################### 
    data_dir = sys.argv[1]
    tables = []
    samples = []
    labels = [] 
    file_name = sys.argv[1]
    
    for filename in os.listdir(data_dir):
        if filename.endswith("20502.xml"): 
            file_path = data_dir + filename
            #print(file_path)
            get_tables_from_xml(file_path, tables) 
        else:
            continue

    for table in tables:
        temp = table.get_samples_and_labels()
        #print(temp['samples'][1])
        samples += temp['samples']
        labels += temp['labels']
    #get_tables_from_xml(file_name, tables)
    #data = tables[0].get_samples_and_labels() 
    train_samples, val_samples, train_labels, val_labels = train_test_split(samples, labels, test_size=.2)
    print(len(train_samples), len(train_labels), len(val_samples), len(val_labels))    
    ############################### 
    # MODEL TRAINING & EVALUATION # 
    ###############################         
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(type(samples[0][0]))
    train_data = tokenizer(
        train_samples, 
        padding = True,
        truncation = True,
        #return_tensors="pt"
    )
    print(train_data["input_ids"])
    sys.exit()
    train_dataset = StatementVerificationWithTablesDataset(train_data, train_labels)

    test_data = tokenizer(
        val_samples, 
        padding = True,
        truncation = True,
        #return_tensors="pt"
    )
    test_dataset = StatementVerificationWithTablesDataset(test_data, val_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,           # evaluation dataset
        compute_metrics=compute_metrics      # detailed metrics for eval
    )

    trainer.train()
    ev = trainer.evaluate()
    print("here", ev)