''' transformer example '''
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

if __name__ == "__main__":
    print("Learning transformers.")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sac = tokenizer(
        [("ee alaka", "together"), ("what is this mate ", "this is it our chance to understand tables"), ("kel alaka ", "tuple tuple in the corner")],
        padding = True,
        truncation = True,
        return_tensors="pt"
    )
    print(sac)

    import torch
    pt_outputs = pt_model(**sac)
    print(pt_outputs)

    
    import torch.nn.functional as F
    pt_predictions = F.softmax(pt_outputs[0], dim=-1)

    print(pt_predictions)