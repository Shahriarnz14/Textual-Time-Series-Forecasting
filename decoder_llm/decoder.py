
from json_comment_reader import read_json_with_comments
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate
from datasets import load_dataset, Dataset
import os
# import sys
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from io import StringIO
import pandas as pd
import uuid
from torcheval.metrics import BinaryF1Score
from subprocess import run as run_subprocess

def num_eval_lines(eval_folder, header=True):
    print('Computing epoch/batch_size ratio')
    result = run_subprocess(f"cat {eval_folder}/*.csv | wc -l", shell=True, capture_output=True, text=True)
    total_lines = int(result.stdout.strip())
    num_files = sum(1 for f in os.listdir(eval_folder) if f.endswith(".csv"))
    # total_lines = sum(sum(1 for _ in open(file, "r", encoding="utf-8")) for file in folder.glob("*.csv"))
    n_eval_lines = total_lines - num_files * (1 if header else 0)
    return n_eval_lines

def save_checkpoint(model, optimizer, epoch, loss, f1_threshold=None, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1_threshold': f1_threshold
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    
    last_path = os.path.normpath(os.path.join(checkpoint_dir, "..", "last.pt"))
    torch.save(checkpoint_data, last_path)
    
    # Check for best checkpoint
    best_path = os.path.normpath(os.path.join(checkpoint_dir, "..", "best.pt"))
    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path)
        if loss < best_checkpoint['loss']:
            torch.save(checkpoint_data, best_path)
            print(f"New best checkpoint saved at {best_path} with loss {loss:.4f}")
    else:
        torch.save(checkpoint_data, best_path)
        print(f"Initial best checkpoint saved at {best_path} with loss {loss:.4f}")
    
    print(f"Checkpoint saved at {checkpoint_path} and transiently at {last_path}")

def load_checkpoint(model, optimizer, checkpoint_path, optimizer_device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f'Invalid checkpoint path {checkpoint_path}, starting from scratch')
        return 0, 0
    checkpoint = torch.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(optimizer_device)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    f1_threshold = checkpoint['f1_threshold'] if 'f1_threshold' in checkpoint else None
    
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {epoch+1}")
    return epoch, loss, f1_threshold

# Define the MLP head
class MLPHead(nn.Module):
    def __init__(self, hidden_size, num_labels, sep_labels):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 256)      # First fully-connected layer
        self.fc2 = nn.Linear(256, num_labels)       # Second fully-connected layer for output
        self.relu = nn.ReLU()                       # Activation function
        self.dropout = nn.Dropout(0.1)              # Dropout for regularization
        self.sep_labels = sep_labels

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# Combine model with MLP head
class ModelWithMLP(nn.Module):
    def __init__(self, model, mlp_head, mlp_head_device):
        super(ModelWithMLP, self).__init__()
        self.model = model
        self.mlp_head = mlp_head
        self.mh_device = mlp_head_device

    def forward(self, input_ids, attention_mask, use_cache=True):
        # Get the last hidden state from RoBERTa
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        # outputs shape is [B x 512 (seq_len) x vocabsize_or_hidden_size]
        # --> so want to use output_hidden_states=True, and pull from:
        #     [B x seq_len x hidden_size]
        # TODO last_hidden_state is not in CLMs, besides we want to use a series of them.
        # hidden_states is (output_layer_num x [B x seq_len x hidden_size])
        last_output = outputs['hidden_states'][-1][:,-1,:] # use last token in a CLM   #[:, 0, :]  # Use the [CLS] token representation
        return self.mlp_head(last_output.float().to(self.mh_device))

# Load dataset and tokenize

def token_and_df(dataset):
    while True:
        for d in dataset['train']:
            df = Dataset.from_pandas(pd.read_csv(StringIO(d['text'])))
            yield { **d, **df.to_dict() }

# import uuid
class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))
def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")

def get_newline_mask(example, max_len):
    text = example['text']
    # newlines = get_newline_locations(text)
    newlines = []
    offsets = example['offset_mapping']
    for i, (start, end) in enumerate(offsets):
        if text[start:end] == "\n":
            newlines.append(i)
        # else: 
        #     print(text[start:end], " __from__ ", start, ":", end)
    return \
        {'newline_mask':
            torch.zeros(max_len,dtype=torch.bool).scatter_(0, torch.tensor(newlines), True).tolist()
        }


def get_newline_mask2(examples, offsets, max_len):
    texts = examples['text']
    # newlines = get_newline_locations(text)
    newlines = [[] for _ in range(len(texts))]
    for j in range(len(texts)):
        for i, (start, end) in enumerate(offsets[j]):
            if texts[j][start:end] == "\n":
                newlines[j].append(i)
            # else: 
            #     print(text[start:end], " __from__ ", start, ":", end)
    return \
        {'newline_mask':
            [torch.zeros(max_len,dtype=torch.bool).scatter_(0, torch.tensor(newline), True).tolist() 
             for newline in newlines]
        }

def tokenize_text(examples, tokenizer, mysuffix='', max_len=512):
    text = examples['text'] if mysuffix == '' else [t + ' ' + mysuffix for t in examples["text"]]
    encs = tokenizer(text,
                     return_offsets_mapping=True,
                     add_special_tokens=True,  # monitor for offset issues with this set to True (not sure if shifts affect)
                     truncation=True,
                    #  padding=True,
                     max_length=max_len
                     )
    
    # get the newline mask
    newline_mask = get_newline_mask2(examples, encs['offset_mapping'], max_len=max_len)

    # get the event char loc, get the time offset
    event_char_locs = [[] for _ in range(len(examples['text']))]
    time_char_offsets = [[] for _ in range(len(examples['text']))]
    for j in range(len(examples['text'])):  # event offset (event_loc to event_loc+time_offset = event string)
        for k in np.where(newline_mask['newline_mask'][j])[0]:
            event_char_locs[j].append(encs.token_to_chars(j,k)[1])  # skips the header event
    for j in range(len(examples['text'])):  # time char offsets
        for ki, k in enumerate(np.where(newline_mask['newline_mask'][j])[0]):
            if ki==0:
                continue
            else: 
                # N/A issue
                try:
                    # match = list(re.finditer(r",\s*\d+", examples['text'][j][event_char_locs[j][ki-1]:event_char_locs[j][ki]]))[-1].start()+1
                    match = list(re.finditer(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', examples['text'][j][event_char_locs[j][ki-1]:event_char_locs[j][ki]]))[-1].start()+1
                except:
                    match = None 
            time_char_offsets[j].append(match)  # skips the header event
        try:  # last event offset
            # lastmatch = list(re.finditer(r",\s*\d+", examples['text'][j][event_char_locs[j][-1]:]))
            lastmatch = list(re.finditer(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', examples['text'][j][event_char_locs[j][-1]:]))
            if len(lastmatch)>0:
                # time_char_offsets[j].append(list(re.finditer(r",\s*\d+", examples['text'][j][event_char_locs[j][-1]:]))[-1].start()+1)  # get the last event (if there is no newline)
                time_char_offsets[j].append(list(re.finditer(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', examples['text'][j][event_char_locs[j][-1]:]))[-1].start()+1)  # get the last event (if there is no newline)
        except:
            time_char_offsets[j].append(None) 

    # For an example, get the event string
    # [res['text'][r:r+t-1] for r,t in zip(res['event_char_locs'], res['time_char_offsets'])]
    # For an example, get the time vector
    # [res['text'][r0+t:r-1] for r0,r,t in zip(res['event_char_locs'][:-1], res['event_char_locs'][1:], res['time_char_offsets'][:-1])]

    return {**encs, **newline_mask, **{"event_char_locs": event_char_locs, "time_char_offsets": time_char_offsets}}

def event_generator(dataset):  # for now, randomly selects from each example before going to the next
    while True:
        for datum in dataset:
            permis = np.random.permutation(len(datum['event']))
            for i in permis:
                yield datum['event'][i]

def pad_list_with_none(vector, k):
    return vector + [None] * max(0, k - len(vector))

def time_to_concmatrix(vector, length=None):  # NOTE: from [-1 to 1, 0 is equality]
    if length is not None:
        zerov = np.zeros(length)
        zerov[:len(vector)] = vector
        vector = zerov
    # Convert the input to a NumPy array
    else:
        vector = np.array(vector)
    if np.any(np.isnan(vector)):
        raise ValueError
    # Compute the comparison matrix
    matrix = np.sign(vector[:, None] - vector[None, :])
    return matrix

def time_to_concmatrix_torch(vector):
    # Convert the input to a PyTorch tensor
    vector = torch.tensor(vector, dtype=torch.float32)
    # Compute the pairwise difference matrix
    diff_matrix = vector[:, None] - vector[None, :]
    # Use sign to compute the ordering matrix
    matrix = diff_matrix.sign()
    return matrix

# NOTE: this assumes the examples are time ordered (in text and df); if not, order the file by time and re-save;
def make_branching(examples, k = 4, k_synth = 0, w = 24, has_warned=False, **kwargs):
    ### Do universal branch processing

    # dict_keys(['text',
    # 'event', 
    # 'time', 
    # 'input_ids', 
    # 'attention_mask', 
    # 'offset_mapping', 
    # 'newline_mask', 
    # 'event_char_locs', 
    # 'time_char_offsets'])
    ex_events = []
    ex_times = []
    ex_itimes = []
    ex_indices = []
    ex_permis = []
    for iel, event_list in enumerate(examples['event']): # event_list: [a,b,c,d] --> [a,b], [b,c], [c,d], [d, None]
        if len(event_list) == 0:
            ex_events.append([])  # need to keep the right lengths even case of empty event lists
            ex_times.append([])
            ex_itimes.append([])
            ex_indices.append([])
            ex_permis.append([])
            continue
        skip_list = [False]
        if len(event_list) > 1:
            skip_list = np.array(examples['time'][iel][:-1]) == np.array(examples['time'][iel][1:])
        deck = deque(range(len(event_list)))
        deck.extend([None]*(k+1))  # extend so you don't get a circular
        subseqis = []
        times = []
        itimes = []
        events = []
        permis = []
        for ei in range(len(event_list)):
            if (ei == len(event_list)-1) or not skip_list[ei]:
                result = np.array(list(deck)[1:k+1])
                result = result[result != np.array(None)].astype(int)
                # time_array = np.array(pad_list_with_none(examples['time'][iel][result[0]:(result[-1]+1)], k), dtype=float)
                try:
                    time_array = np.array([examples['time'][iel][i] for i in result], dtype=float)
                    if not has_warned:
                        if len(time_array) > 1:
                            if np.any(time_array[1:] < time_array[:-1]):
                                has_warned = True
                                print('WARNING: detected non-time-ordered event lists:\n', 
                                    ', '.join(event_list),
                                    ', '.join([str(t) for t in time_array])
                                )
                            
                    time_array_deck = np.array(examples['time'][iel][deck[0]],dtype=float)
                    subseqis.append(result)  # Append the current state as a list
                    times.append(time_array)
                    itimes.append((time_array-time_array_deck) <= w)  # this is the window_itime determination
                    # we'd prefer to select the set that are in (0,w] filling up to k slots.
                    # given that you get all events at the identical time, just use the (random) last co-ocurring time?
                    # implement via a time_array[:-1] != time_array[1:] skip vector?
                    events.append(np.array([event_list[i] for i in result]))
                    # gen synth events
                    permis.append(np.random.permutation(len(result)+k_synth))
                except ValueError:
                    pass
            deck.rotate(-1)  # Rotate left by 1, regardless of skipping

        ex_events.append(events)
        ex_times.append(times)
        ex_itimes.append(itimes)
        ex_indices.append(subseqis)
        ex_permis.append(permis)
        
        # censorship is just 1-I(window), and 1 for the synths
    

    # should it be window dependent instead # events? --


    # Operationally:
    # (1) collect k next events, times (if <k, request extra synth; if >k, subsample);
        # get exactly k if left; 
    # (2) add k' synthetic events; check synth event are not among subsampled events
    # (3) generate vectors: [Event candidates], [Ordering] [I(window)], [Times], [Censor]
        # done (minus censor which is 1-I(window))
    # (4) generate token suffixes for events and task
    # (5) generate labels for tasks
    # (6) generate token-label pairs, make batches with truncation/padding
    ex_strings = []
    for event_ls, time_ls, perm_ls in zip(ex_events, ex_times, ex_permis):
        event_strings = []
        for event, time, perm in zip(event_ls, time_ls, perm_ls):
            es = event[perm[:len(event)]]
            if len(es) > 0:
                event_strings.append(', '.join(es))
            else:
                event_strings.append("")
        ex_strings.append(event_strings)
        # tokenizer(examples["text"],
        #              return_offsets_mapping=True,
        #              add_special_tokens=True,  # monitor for offset issues with this set to True (not sure if shifts affect)
        #              truncation=True,
        #              padding=True,
        #              max_length=MAX_LEN
        #              )

    labels = {}
    ### Do labeling according to kwargs
    if "concordance" in kwargs:
        ex_label_strings = []
        ex_label_times = []
        for event_strings, event_ls, time_ls, perm_ls in zip(ex_strings, ex_events, ex_times, ex_permis):
            label_strings = []
            label_times = []
            for strings, event, time, perm in zip(event_strings, event_ls, time_ls, perm_ls):
                label_strings.append("The following events occur: " + strings + ". Order the last k=" + str(k+k_synth) + " phrases in time.")
                time_filled = np.where(np.isnan(time), np.maximum.accumulate(np.nan_to_num(time, nan=-1e10)), time)  # Last Value Carry Forward
                if len(time_filled) == 0:
                    pass
                else:
                    if np.isnan(time[0]):
                        time_filled[0] = -1e10  # To avoid np.inf-np.inf --> np.nan
                        time_filled = np.where(np.isnan(time_filled), np.maximum.accumulate(np.nan_to_num(time_filled, nan=-np.inf)), time_filled)  # Last Value Carry Forward
                

                # if np.any(np.isnan(time)):
                    # label_times.append([], length=k+k_synth)    
                label_times.append(time_to_concmatrix(time_filled[perm[:len(time)]], length=k+k_synth)) # re-order by the permuation (string already is), but not that this again assumes times are ordered in the data file.
            ex_label_strings.append(label_strings)
            ex_label_times.append(label_times)
        labels['concordance_label_strings'] = ex_label_strings
        labels['concordance_label_times'] = ex_label_times

    if "window" in kwargs:
        ex_label_strings = []
        ex_label_itimes = []
        for event_strings, event_ls, itime_ls, perm_ls in zip(ex_strings, ex_events, ex_itimes, ex_permis):
            label_strings = []
            label_itimes = []
            for strings, event, itime, perm in zip(event_strings, event_ls, itime_ls, perm_ls):
                label_strings.append("Suppose thee following events occur: " + strings + ". Which of the k=" + str(k+k_synth) + " events occur in the next " + str(w) + " hour(s)?")
                itime_result = np.zeros(k+k_synth)
                itime_result[:len(itime)] = itime[perm[:len(itime)]]
                label_itimes.append(itime_result) # re-order by the permuation (string already is), but not that this again assumes times are ordered in the data file.
            ex_label_strings.append(label_strings)
            ex_label_itimes.append(label_itimes)
        labels['window_label_strings'] = ex_label_strings
        labels['window_label_itimes'] = ex_label_itimes

    if len(examples['text']) != len(ex_events):
        raise ValueError
    # raise NameError
    return {'ex_events':ex_events,
            'ex_times':ex_times,
            'ex_itimes':ex_itimes,
            'ex_indices':ex_indices,
            'ex_permis':ex_permis,
            **labels
            }

def make_tokenized_pairs(examples, tokenizer, data_collator, max_tokens=512, padding=True, truncation=True):
    io_dict = {}

    if 'window_label_strings' in examples.keys():
        window_input_ids = []
        flattened_wls = []
        flattened_input_ids = []
        flattened_wli = []
        for input_id, wls, wli in zip(examples['input_ids'],
                                      examples['window_label_strings'],
                                      examples['window_label_itimes']):
            for string, itime in zip(wls,wli):
                flattened_input_ids.append(input_id)
                flattened_wls.append(string)
                flattened_wli.append(itime)
        encs = tokenizer(
            flattened_wls,
            add_special_tokens=True,  # monitor for offset issues with this set to True (not sure if shifts affect)
            truncation=truncation,
            padding=padding,
            max_length=max_tokens
        )
        window_input_ids = [flattened_input_ids[i] + enc_input_ids
                            for i, enc_input_ids in enumerate(encs['input_ids'])]
        # Define a data collator
        batch = data_collator(
            [{"input_ids": seq} for seq in window_input_ids]
        )
        io_dict['window_input_ids'] = batch['input_ids']
        io_dict['f_window_label_strings'] = flattened_wls
        io_dict['f_window_label_itimes'] = flattened_wli


    if 'concordance_label_strings' in examples.keys():
        concordance_input_ids = []
        flattened_cls = []
        flattened_clt = []
        flattened_input_ids = []
        for input_id, cls, clt in zip(examples['input_ids'],
                                      examples['concordance_label_strings'],
                                      examples['concordance_label_times']):
            for string, time in zip(cls, clt):
                flattened_input_ids.append(input_id)
                flattened_cls.append(string)
                flattened_clt.append(time)

        encs = tokenizer(
            [item for sublist in examples['concordance_label_strings'] for item in sublist],
            add_special_tokens=True,  # monitor for offset issues with this set to True (not sure if shifts affect)
            truncation=truncation,
            # padding=padding,  # padding has to happen later (pads to max seq length), need to pad them with the flattened_input_ids
            max_length=max_tokens
        )
        concordance_input_ids = [flattened_input_ids[i] + enc_input_ids
                                 for i, enc_input_ids in enumerate(encs['input_ids'])]
        # Define a data collator
        batch = data_collator(
            [{"input_ids": seq} for seq in concordance_input_ids]
        )

        io_dict['concordance_input_ids'] = batch['input_ids']
        io_dict['f_concordance_label_strings'] = flattened_cls
        io_dict['f_concordance_label_times'] = flattened_clt
        # raise NameError    
    return io_dict

class Extras_DataCollatorWithPadding(DataCollatorWithPadding):
    # probably slow, but it runs
    def __call__(self, features):
        # sub_features1 = features.map(lambda x: super().__call__(x))  # can't map
        if 'window_input_ids' in features[0]:
            sub_features = [{'input_ids': f['window_input_ids']} for f in features]
            new_window_features = super().__call__(sub_features)
        if 'concordance_input_ids' in features[0]:
            sub_features = [{'input_ids': f['concordance_input_ids']} for f in features]
            new_concordance_features = super().__call__(sub_features)
        for i, f in enumerate(features):
            if 'window_input_ids' in features[0]:
                features[i]['window_input_ids'] = new_window_features['input_ids'][i]
            if 'concordance_input_ids' in features[0]:
                features[i]['concordance_input_ids'] = new_concordance_features['input_ids'][i]
        return features

# extras_data_collator = Extras_DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

def collate_fn_with_device(batch, data_collator, device):  # This sends Extras_DataCollatorWithPadding to cuda
    batch = data_collator(batch)  # Use DataCollatorWithPadding for padding
    for b in batch:
        for key, val in b.items():
            if key != "f_concordance_label_strings" and key != "f_window_label_strings":
                b[key] = val.to(device)
    return default_collate(batch)

# Return m, where rows indexes the time, and columns indexes the future time, asking if future_time is within t of the time.
# unused for now(?)
def events_within_t_mask(times, t):
    import numpy as np
    m = np.tile(times, (len(times),1))
    # print(m - (m.transpose()+t))  
    return np.triu((m - (m.transpose()+t))<=0, 1)

# Dataset
def get_dataloader(data_dir, tokenizer, data_collator, num_labels, device, forecast_window=24, system_text='', batch_size=20, max_len=512):
    ds = load_dataset("text", data_files = [ (data_dir+f) for f in os.listdir(data_dir) if f.endswith(".csv")], sample_by="document", streaming=True)
    d3 = Dataset.from_generator(_DatasetGeneratorPickleHack(token_and_df(ds)), streaming=True)
    # print(system_text)
    tokenized_datasets = d3.map(lambda x: tokenize_text(x, tokenizer, system_text, max_len), batched=True)
    branched_dataset = tokenized_datasets.map(
        lambda x: make_branching(x,concordance=True, window=True, k = num_labels, w=forecast_window),
        batched=True,
        #    batch_size=2
        batch_size=batch_size
    )
    cols_to_remove = next(iter(branched_dataset)).keys()
    tokenized_pairs_dataset = branched_dataset.map(
        lambda x: make_tokenized_pairs(x, tokenizer, data_collator),
        batched=True,
        # batch_size=2,
        batch_size=batch_size,
        remove_columns=cols_to_remove
    ).with_format('torch')
    extras_data_collator = Extras_DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # DataLoader
    data_loader = DataLoader(tokenized_pairs_dataset, 
                            batch_size=batch_size,
                            collate_fn=lambda x: collate_fn_with_device(x, extras_data_collator, device)
    )
    return data_loader

# Training function
def train(model, train_loader, criterion, optimizer, eos_token_id='', step_out=None, accumulation_steps=5):
    model.train()
    total_loss = 0
    steps = 0
    for bi, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        attention_mask = batch['window_input_ids'] != eos_token_id
        # slow, but att_mask is computable via ... == eos_token_id
        ## TODO insert prefix/suffix mods here; also the way the result is parsed needs to be changed;
        ## for the 0-shot, few-shot, it only is done in the eval section?
        outputs = model(input_ids=batch['window_input_ids'], attention_mask=attention_mask)
        if model.mlp_head.sep_labels:
            # print("check this")
            outputs = outputs[:,:batch['f_window_label_itimes'].shape[1]]
        window_loss = criterion(outputs, batch['f_window_label_itimes'])

        # Let's overload the model to predict itimes and label with the same model
        # Perhaps a bad idea
        attention_mask = batch['concordance_input_ids'] != eos_token_id
        c_outputs = model(input_ids=batch['concordance_input_ids'], attention_mask=attention_mask)
        if model.mlp_head.sep_labels:
            # print("check this")
            c_outputs = c_outputs[:,batch['f_window_label_itimes'].shape[1]:]
        concordance_loss = criterion(c_outputs.unsqueeze(1) - c_outputs.unsqueeze(2), (batch['f_concordance_label_times']+1.)/2)

        loss = window_loss + concordance_loss

        # Backward pass and optimization
        loss.backward()
        if torch.isnan(loss):
            print("WINDOW:\n", outputs, "\nCONCORDANCE:\n", c_outputs)
            raise ValueError
        total_loss += loss
        print(bi, "; LOSS: ", loss.item())
        steps = bi
        if(bi % accumulation_steps == accumulation_steps-1) or steps == step_out:
            optimizer.step()
        if steps == step_out:
            break
        # else:
            # print(f'{step_out - steps} steps til SOF')

    return total_loss * 1. / steps

def train_with_checkpoint(model, dataloader, loss_fn, optimizer, writer,
                          eos_token_id='',
                          start_epoch=0, epochs=3, checkpoint_interval=1, f1_threshold=None, step_out_factor=None, accumulation_steps=5, save_prefix=''):
    loss = 0
    running_loss = 0
    best_loss = np.inf
    for si, so_epoch in enumerate(range(start_epoch, start_epoch + epochs)):
        # Log training loss to TensorBoard
        loss = train(model, dataloader, loss_fn, optimizer, eos_token_id=eos_token_id, step_out=step_out_factor, accumulation_steps=accumulation_steps)
        print(f"Epoch {so_epoch+1}, Step {so_epoch*step_out_factor}, Loss: {loss.item()}")
        writer.add_scalars("Loss", {"Train": loss.item()}, so_epoch)

        running_loss += loss
        avg_loss = running_loss / (si + 1)
        writer.add_scalars("Loss/AvgEpoch", {"Train":avg_loss}, so_epoch)

        # Save checkpoint at the specified interval
        if (so_epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model.mlp_head, optimizer, so_epoch + 1, avg_loss, f1_threshold, checkpoint_dir=os.path.join("checkpoints/", save_prefix))
    return avg_loss

def eval_with_writer(model, dataloader, loss_fn, writer, eos_token_id='', device='cuda', tokenizer=None, start_epoch=0, epochs=3, step_out_factor=None, epoch_sof=None, approach="MLP", 
                     threshold=None, eval_mode=False, eval_dir=None, system_text=''):
    loss = 0
    running_loss, avg_loss = 0, 0
    best_loss = np.inf
    for si, so_epoch in enumerate(range(start_epoch, start_epoch+epochs)):
        step_out = min(step_out_factor, epoch_sof) if step_out_factor is not None and epoch_sof is not None else step_out_factor
        # Log training loss to TensorBoard
        loss, concordance, f1, f1_threshold, batch_errors = evaluate(model, dataloader, loss_fn, eos_token_id=eos_token_id, step_out=step_out, approach=approach, device=device,
                                                                     threshold=threshold, tokenizer=tokenizer, system_text=system_text)
        print(f"Epoch {so_epoch+1}, Step {so_epoch*step_out_factor}, Eval loss: {loss.item()}")
        writer.add_scalars("Loss", {"Eval":loss.item()}, so_epoch)
        writer.add_scalar("Concordance", concordance.item(), so_epoch)
        writer.add_scalar("F1", f1.item(), so_epoch)
        writer.add_scalar("F1_Threshold", f1_threshold.item(), so_epoch)
        writer.add_scalar("BatchError", batch_errors, so_epoch)

        if eval_mode:
            result_table = f'Measurement | Values \nEval Directory |{eval_dir} \nConcordance | {concordance.item()}\n F1 | {f1.item()}\nF1_Threshold | {f1_threshold.item()}\nBatchError | {batch_errors}'
            print(result_table)
            writer.add_text(f"Results ({eval_dir})",result_table)
            break

        running_loss += loss
        avg_loss = running_loss / (si + 1)
        writer.add_scalars("Loss/AvgEpoch", {"Eval": avg_loss}, so_epoch)
    return avg_loss, f1_threshold

def concordance_from_pms(yhat, y):
    denom = torch.sum(y.abs() > 1e-10)
    numerator = torch.sum((yhat * y) > 1e-10)
    return numerator, denom

def filter_bar_lines(multiline_str):
    return "\n".join(line for line in multiline_str.split("\n") if '|' in line)

def llama_wrap(text:str):
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + \
            text + "<|eot_id|><|start_header_id|>user<|end_header_id|>" + \
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def llama_wrap_tokens(batch, system_text, tokenizer,device=None):
    return add_prefix_suffix_to_batch(
        batch['window_input_ids'],
        torch.ones_like(batch['window_input_ids']),
        prefix_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + system_text + \
            "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        suffix_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        tokenizer = tokenizer,
        device = device
    )

def add_prefix_suffix_to_batch(
    input_ids, attention_masks, prefix_text, suffix_text, tokenizer, device="cpu"
):
    """
    Adds a prefix and suffix to a batch of tokenized inputs, with device handling, using return_tensors="pt"

    Args:
        input_ids (torch.Tensor): Tensor of input token IDs (batch_size, sequence_length).
        attention_masks (torch.Tensor): Tensor of attention masks (batch_size, sequence_length).
        prefix_text (str): Text to add as a prefix.
        suffix_text (str): Text to add as a suffix.
        tokenizer (AutoTokenizer): The tokenizer used to encode the inputs.
        device (str): "cpu" or "cuda" (if available).
    Returns:
        tuple: A tuple containing the new input_ids and attention_masks with the prefix and suffix added.
    """

    prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)
    suffix_inputs = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).to(device)

    prefix_ids = prefix_inputs["input_ids"][0]  # Get the tensor of prefix IDs
    suffix_ids = suffix_inputs["input_ids"][0]  # Get the tensor of suffix IDs

    new_input_ids = []
    new_attention_masks = []

    for i in range(input_ids.shape[0]):  # Iterate over batch
        current_input_ids = input_ids[i]
        current_attention_mask = attention_masks[i]

        # Remove padding from original input, before adding prefix/suffix
        original_len = torch.sum(current_attention_mask).item()
        current_input_ids = current_input_ids[:original_len]
        current_attention_mask = current_attention_mask[:original_len]

        combined_input_ids = torch.cat((prefix_ids, current_input_ids, suffix_ids))
        combined_attention_mask = torch.ones_like(combined_input_ids)

        new_input_ids.append(combined_input_ids)
        new_attention_masks.append(combined_attention_mask)

    # Pad to the maximum sequence length in the batch.
    max_len = max(seq.shape[0] for seq in new_input_ids)

    padded_input_ids = []
    padded_attention_masks = []

    for seq_ids, mask in zip(new_input_ids, new_attention_masks):
        padding_len = max_len - seq_ids.shape[0]
        padded_seq_ids = torch.cat((seq_ids, torch.full((padding_len,), tokenizer.pad_token_id, device=device)))
        padded_mask = torch.cat((mask, torch.zeros((padding_len,), dtype=torch.long, device=device)))

        padded_input_ids.append(padded_seq_ids)
        padded_attention_masks.append(padded_mask)

    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_attention_masks),
    )

# Evaluation function
def evaluate(model, test_loader, criterion, eos_token_id = '', tokenizer=None, step_out=None, device='cuda',
             approach="MLP", threshold=None, system_text=''):
    model.eval()
    total_loss = 0
    correct = 0
    steps = 0
    # running_preds, running_labels = [], []
    if threshold is None:
        metrics = [BinaryF1Score(threshold=torch.log(torch.tensor((k+1)*1./20))) for k in range(19)]
    else:
        metrics = [BinaryF1Score(threshold=threshold)]

    running_cnum, running_cdenom = torch.zeros(1).to(device), torch.zeros(1).to(device)
    batch_errors = 0
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if steps + batch_errors > step_out:
                break

            if approach == "MLP":
                attention_mask = batch['window_input_ids'] != eos_token_id
                outputs = model(input_ids=batch['window_input_ids'], attention_mask=attention_mask)
                if model.mlp_head.sep_labels:
                    # print("check this")   
                    outputs = outputs[:,:batch['f_window_label_itimes'].shape[1]]
            elif approach.startswith("PROMPT:"):
                wii_inputs, _ = llama_wrap_tokens(batch, system_text, tokenizer, device)
                wii_att = wii_inputs != eos_token_id
                generations = model.generate(input_ids=wii_inputs, attention_mask=wii_att, max_new_tokens=100)
                generations = tokenizer.batch_decode([g[len(b):] for g, b in zip(generations, wii_inputs)], skip_special_tokens=True)
                # generations = model(input_ids=batch['window_input_ids'], attention_mask=attention_mask)
                ## TODO parse generations here to map to outputs
                if approach == "PROMPT:window":
                    try:
                        k = batch['f_window_label_itimes'][0].shape[0]
                        parts = [filter_bar_lines(g).strip().split('|')[:k] for g in generations]
                        outputs = torch.tensor([[float(f) for f in part] for part in parts]).to(device)
                        outputs = torch.log(outputs/outputs.max() + 1e-10)  # F1 is calculated using log thresholds of values between 0 and 1
                        # print(outputs)
                        if outputs.isnan().any():
                            raise ValueError
                    except:
                        print('ERROR in output; skipping')
                        print(generations)
                        batch_errors += 1
                        continue

                elif approach == "PROMPT:ordering":
                    try:
                        parts = [filter_bar_lines(g).strip().split('|') for g in generations]
                        outputs = time_to_concmatrix_torch(torch.tensor([[float(f)] for part in parts for f in part])).to(device)
                        if outputs.isnan().any():
                            raise ValueError
                    except:
                        print(generations)
                else:
                    raise ValueError(f"approach ({approach}) must be 'window' or 'ordering'")
            else:
                raise RuntimeError(f"invalid 'approach' ({approach})")

            if approach in ['MLP', 'PROMPT:window']:
                window_loss = criterion(outputs, batch['f_window_label_itimes'])
                [metric.update(outputs.flatten(), batch['f_window_label_itimes'].flatten()) for metric in metrics]
            else:
                window_loss = 0

            if approach in ['MLP', 'PROMPT:ordering']:
                # Let's overload the model to predict itimes and label with the same model
                # Perhaps a bad idea
                attention_mask = batch['concordance_input_ids'] != eos_token_id
                c_outputs = model(input_ids=batch['concordance_input_ids'], attention_mask=attention_mask)
                if model.mlp_head.sep_labels:
                    # print("check this")
                    c_outputs = c_outputs[:,batch['f_window_label_itimes'].shape[1]:]
                concordance_loss = criterion(c_outputs.unsqueeze(1) - c_outputs.unsqueeze(2), (batch['f_concordance_label_times']+1.)/2)
                cnum, cdenom = concordance_from_pms(
                    c_outputs.unsqueeze(1) - c_outputs.unsqueeze(2), # real-valued pred
                    batch['f_concordance_label_times']  # labels from -1 to 1 (-1: before, 0: incomp, 1: after)
                )
                running_cnum += cnum
                running_cdenom += cdenom
            else:
                concordance_loss = 0
                running_cdenom = 1
            

            loss = window_loss + concordance_loss
            
            # Backward pass and optimization
            if torch.isnan(loss):
                print("WINDOW:\n", outputs, "\nCONCORDANCE:\n", c_outputs)
                raise ValueError
            print(bi, "; EVAL LOSS: ", loss.item())
            total_loss += loss
            steps = bi
            if steps == step_out:
                break
    concordance = running_cnum / (running_cdenom + 1e-10)
    metrics_values = [metric.compute() for metric in metrics]
    f1 = max(metrics_values)
    best_index = max(enumerate(metrics_values), key=lambda x: x[1])[0]
    best_threshold = torch.log(torch.tensor((best_index + 1) * 1.0 / 20)) if threshold is None else threshold
    
    return total_loss * 1. / (steps+1), concordance, f1, best_threshold, batch_errors


### ASSESS

import argparse
from datetime import datetime

class TextForecastModel:
    def __init__(self, 
                 run_dir: str = "", 
                 approach: str = "MLP",
                 systext_file: str = "",
                 infix: str = "v0anysorss",
                 data_dir: str = "", 
                 test_dir: str = "",
                 base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
                 cache_dir: str = "/data/<user>/.cache/huggingface/hub/",
                 dl_version: str = "direct",
                 num_labels: int = 8, 
                 forecast_window: int = 24,
                 sep_labels: bool = False,
                 max_len: int = 512, 
                 learning_rate: float = 1e-3,
                 epochs: int = 3000, 
                 train_epochs_per_eval: int = 10, 
                 next_epoch: int = 0, 
                 checkpoint_interval: int = 20, 
                 checkpoint_loadpath: str = None,
                 sof: int = 20,
                 eval_mode: bool = False,
                 accumulation_steps: int = 5,
                 batch_size: int = 20,
                 device = None):
        self.run_dir = run_dir
        self.approach = approach
        self.systext_file = systext_file
        self.infix = infix
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.base_model = base_model
        self.cache_dir = cache_dir
        self.dl_version = dl_version
        self.num_labels = num_labels
        self.forecast_window = forecast_window
        self.sep_labels = sep_labels
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_epochs_per_eval = train_epochs_per_eval
        self.next_epoch = next_epoch
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_loadpath = checkpoint_loadpath
        self.sof = int(sof)
        self.eval_mode = eval_mode
        self.accumulation_steps= accumulation_steps
        self.batch_size = batch_size
        self.device = device


    def main(self):
        # Initialize TensorBoard writer
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Run directory: {self.run_dir}")
        print(f"Current time: {current_time}")
        print(f"Approach: {self.approach}")
        print(f"System text file: {self.systext_file}")
        print(f"Infix: {self.infix}")
        print(f"Data directory: {self.data_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Base model: {self.base_model}")
        print(f"Cache dir: {self.cache_dir}")
        print(f"Dataloader version: {self.dl_version}")   
        print(f"Number of labels: {self.num_labels}")
        print(f"Forecast window: {self.forecast_window}")
        print(f"Separate conc/window labels: {self.sep_labels}")
        print(f"Max length: {self.max_len}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Train epochs per evaluation: {self.train_epochs_per_eval}")
        print(f"Next epoch: {self.next_epoch}")
        print(f"Checkpoint interval: {self.checkpoint_interval}")
        print(f"Checkpoint path: {self.checkpoint_loadpath}")
        print(f"SOFT: {self.sof}")
        print(f"Eval mode: {self.eval_mode}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")

        run_dir = os.path.join(self.run_dir, f"{self.infix}/{current_time}")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self.device)

        ### DATA
        # Set the working directory to the project root (adjust as needed)
        project_dir = os.path.dirname(os.path.abspath(__file__))  # Automatically detects the running script's location
        os.chdir(project_dir)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, cache_dir=self.cache_dir,
                                                  torch_dtype = torch.bfloat16
                                                  )  # Let's train on smaller than F32?
        tokenizer.pad_token = tokenizer.eos_token
        lmmodel = AutoModelForCausalLM.from_pretrained(self.base_model, cache_dir=self.cache_dir,
                                                       device_map="auto",
                                                    #    device_map="balanced_low_0",
                                                       torch_dtype = torch.bfloat16
                                                       )
        lmmodel.config.output_hidden_states = True
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


        # Freeze LM if desired to only train MLP head
        for param in lmmodel.parameters():
            param.requires_grad = False  # Uncomment to freeze LM layers

        ### INIT
        # Initialize MLP head with desired dimensions
        hidden_size = lmmodel.config.hidden_size  # Typically 768 for 'roberta-base'
        total_labels = self.num_labels * 2 if self.sep_labels else self.num_labels
        mlp_head = MLPHead(hidden_size, total_labels, self.sep_labels)
        # MAX_LEN = 512

        checkpoint_loadpath = self.checkpoint_loadpath
        if checkpoint_loadpath is not None:
            _, _, last_threshold = load_checkpoint(mlp_head, None, checkpoint_loadpath, optimizer_device=device)
        else:
            last_threshold = None

        # Create the combined model
        if self.approach == "MLP":
            model = ModelWithMLP(lmmodel, mlp_head, mlp_head_device=device)
        else:
            model = lmmodel

        if self.systext_file != '':
            try:
                with open(self.systext_file, 'r') as file:
                    system_text = file.read()
            except:
                raise IOError(f"{self.systext_file} not read.")
        else:
            system_text = ''

        if self.dl_version == 'direct':
            train_loader = get_dataloader(self.data_dir, tokenizer, data_collator=data_collator,
                                        num_labels=self.num_labels, forecast_window=self.forecast_window,
                                        device=device, batch_size=self.batch_size, max_len=self.max_len)
            test_loader = get_dataloader(self.test_dir, tokenizer, data_collator=data_collator, system_text=system_text,
                                        num_labels=self.num_labels, forecast_window=self.forecast_window,
                                        device=device, batch_size=self.batch_size, max_len=self.max_len)
        elif self.dl_version == 'load_from_dir':
            load_dir, prep_dir = self.load_dir, self.prep_dir

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loaded_epoch = 0
        if checkpoint_loadpath is not None:
            loaded_epoch, _, _ = load_checkpoint(None, optimizer, checkpoint_loadpath, optimizer_device=device)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        # Set up device (use GPU if available)
        # model.to(device)
        mlp_head.to(device)

        eval_epoch_sof = int((num_eval_lines(self.test_dir)+self.batch_size) / self.batch_size)


        for epoch in range(loaded_epoch, loaded_epoch + int(self.epochs)):
            if epoch < self.next_epoch:
                continue

            # train_loss = train(model, train_loader, criterion, optimizer)
            train_loss=None
            if self.approach == "MLP":
                if not self.eval_mode:
                    train_loss = train_with_checkpoint(
                        model, train_loader, criterion, optimizer, writer, 
                        eos_token_id=tokenizer.eos_token_id,
                        start_epoch=epoch,
                        epochs=self.train_epochs_per_eval,
                        checkpoint_interval=self.checkpoint_interval,
                        f1_threshold=last_threshold,
                        step_out_factor=self.sof,
                        accumulation_steps=self.accumulation_steps,
                        save_prefix=f"{self.infix}/{current_time}"
                    )
                self.next_epoch = epoch + self.train_epochs_per_eval
                test_loss, last_threshold = eval_with_writer(
                    model, test_loader, criterion, writer,
                    eos_token_id=tokenizer.eos_token_id,
                    start_epoch=self.next_epoch,
                    epochs=1,
                    threshold=last_threshold if self.eval_mode else None,
                    eval_mode=self.eval_mode,
                    eval_dir=self.test_dir,
                    step_out_factor=self.sof,
                    epoch_sof=eval_epoch_sof,
                    device=self.device
                )
            elif self.approach.startswith("PROMPT"):
                train_loss = 0
                self.next_epoch = epoch + self.train_epochs_per_eval
                test_loss, last_threshold = eval_with_writer(
                    model, test_loader, criterion, writer,
                    eos_token_id=tokenizer.eos_token_id,
                    start_epoch=self.next_epoch,
                    epochs=1,
                    step_out_factor=self.sof,
                    epoch_sof=eval_epoch_sof,
                    approach = self.approach,
                    threshold=last_threshold if self.eval_mode else None,
                    eval_mode=self.eval_mode,
                    eval_dir=self.test_dir,
                    system_text = system_text,
                    device = self.device,
                    tokenizer = tokenizer
                )

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}") if train_loss is not None else print('')
            print(f"Test Loss: {test_loss:.4f}")
            # print(f"Test Accuracy: {test_accuracy:.4f}")

            if self.eval_mode:
                break
            # log to tensorboard

            # checkpoint
        writer.close()

def parse_args(json_args=None):
    parser = argparse.ArgumentParser(description="Text Forecast Model")
    parser.add_argument("--run_dir", type=str, default="/data/<user>/workspace/", help="Run directory")
    parser.add_argument("--infix", type=str, default="ord0", help="Infix for the run")
    # parser.add_argument("--infix", type=str, default="debug", help="Infix for the run")
    parser.add_argument("--approach", type=str, default="MLP", help="Modeling approach (e.g. MLP, PROMPT:window, PROMPT:ordering)")
    parser.add_argument("--systext_file", type=str, default="", help="System text file to use as system text to the prompt")
    parser.add_argument("--data_dir", type=str, default="/data/<user>/workspace/ordered_l33train/", help="Data directory")
    parser.add_argument("--test_dir", type=str, default="/data/<user>/workspace/ordered_l33test/", help="Test directory")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model")
    parser.add_argument("--cache_dir", type=str, default="/data/<user>/.cache/huggingface/hub/", help="Cache path")
    parser.add_argument("--dataloader_version", type=str, default="direct", help="DL version (direct, load_from_dir)")
    parser.add_argument("--num_labels", type=int, default=8, help="Number of labels")
    parser.add_argument("--forecast_window", type=int, default=24, help="Window to make labeling for binary predictions.")
    # parser.add_argument("--separate_labels", type=bool, default=False, help="Separate labels for concordance and window classification")
    parser.add_argument("--separate_labels", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Separate labels for concordance and window classification")
    parser.add_argument("--max_len", type=int, default=512, help="Max length")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs")
    parser.add_argument("--train_epochs_per_eval", type=int, default=10, help="Train epochs per evaluation")
    # parser.add_argument("--train_epochs_per_eval", type=int, default=1, help="Train epochs per evaluation")
    parser.add_argument("--next_epoch", type=int, default=0, help="Next epoch")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Run evaluation with given threshold (searches if no threshold saved from checkpoint)")
    parser.add_argument("--checkpoint_loadpath", type=str, default=None, help="Checkpoint load path")
    parser.add_argument("--sof", type=int, default=20, help="SOFT")
    parser.add_argument("--eval_mode", action="store_true", default=json_args.get("eval_mode",False) if json_args is not None else False, help="Enable evaluation mode")
    parser.add_argument("--accumulation_steps", type=int, default=5, help="Steps to accumulate before opt.step()")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--json", type=str, default=None, help="Use a json file to load files")
    return parser.parse_args()

if __name__ == "__main__":
    # First pass to get store_action settings (if any)
    args = parse_args()
    if args.json is not None:
        print(f'Using json file arguments {args.json}; note these *overwrite* command line args')
        customargs = read_json_with_comments(args.json)
    else:
        customargs = None

    args = parse_args(customargs)
    if args.json is not None:
        # print(customargs)
        for key, value in customargs.items():
            # print(key, ": ", value)
            setattr(args, re.sub(r'^--','',key), value)

    runner = TextForecastModel(
        run_dir=args.run_dir,
        infix=args.infix,
        approach=args.approach,
        systext_file=args.systext_file,
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        base_model=args.base_model,
        cache_dir = args.cache_dir,
        dl_version = args.dataloader_version,
        num_labels=args.num_labels,
        forecast_window=args.forecast_window,
        sep_labels = args.separate_labels,
        max_len=args.max_len,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_epochs_per_eval=args.train_epochs_per_eval,
        next_epoch=args.next_epoch,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_loadpath=args.checkpoint_loadpath,
        sof=args.sof,
        eval_mode=args.eval_mode,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        device = args.device
    )
    runner.main()
    
