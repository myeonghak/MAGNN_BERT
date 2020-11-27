import pathlib
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import transformers
import torch


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def fetch_vectors(string_list, batch_size=8):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    
    DEVICE = torch.device("cuda")
    
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    bert_model.to(DEVICE)
    
    fin_features = []
    max_len = 64

    for data in chunks(string_list, batch_size):
        tokenized = []
        
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:max_len])
        
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        
        attention_mask = np.where(padded != 0, 1, 0)        
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)
        
        with torch.no_grad():
            last_hidden_states = bert_model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)
        
    fin_features = np.vstack(fin_features)
    return fin_features