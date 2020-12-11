''' transformer example '''
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataset import StatementVerificationWithTablesDataset

def compute_metrics(pred):
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

if __name__ == "__main__":
    print("Learning transformers.")
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    #init tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_data = tokenizer(
        [("ee alaka", "together"), ("what is this mate ", "this is it our chance to understand tables"), ("kel alaka ", "tuple tuple in the corner")],
        padding = True,
        truncation = True,
        #return_tensors="pt"
    )
    train_dataset = StatementVerificationWithTablesDataset(train_data, [1, 0, 1])
    #print(train_data)

    test_data = tokenizer(
        [("ee alaka", "together"), ("what is this mate ", "this is it our chance to understand tables"), ("kel alaka ", "tuple tuple in the corner")],
        padding = True,
        truncation = True,
        #return_tensors="pt"
    )
    test_dataset = StatementVerificationWithTablesDataset(test_data, [0, 1, 0])
    #print(test_data)
    
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
        train_dataset=train_dataset,            # training dataset
        eval_dataset=test_dataset,               # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    ev = trainer.evaluate()
    print("here", ev)