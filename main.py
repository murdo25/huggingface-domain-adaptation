import pandas as pd
import torch

# df = pd.read_csv('data/clean_westworld.csv')
df = pd.read_csv('data/westworldS1_windowed.csv')

print(df)

n_lines = df.head(100)
text = n_lines['input_text'].to_list()

from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch



# from transformers import pipeline
# unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)


# run_mlm("Hello I'm a [MASK] language model.")
# run_mlm("He said that Mozart, Beethoven, and Chopin never died. They simply became [MASK].")
# run_mlm("Cayla wants to die becaus [MASK]")


# TODO: make a batched version of this, join everything back together at the end beofre creating the cutsom dataset obj

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', model_max_length=128)
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
print(inputs)
print(inputs.keys())
inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats in equal dimension to input_ids
rand = torch.rand(inputs.input_ids.shape)
# where the random array is less than 0.15, we set true
# mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
print("MASK arr")
print(mask_arr)
# create selection from mask_arr
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
print('selection')
print(selection)
inputs.input_ids[0, selection] = 103
print("mask over inputs")
print(inputs.input_ids)

model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
outputs = model(**inputs)
print(outputs.keys())
print(outputs.loss)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = CustomDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='out',
    per_device_train_batch_size=2,
    num_train_epochs=2
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)
trainer.train()




from transformers import pipeline
unmasker = pipeline('fill-mask', model=trainer.model, tokenizer=tokenizer, device=0)


def run_mlm(seq):
    # print("SEQ: ", seq)
    response = unmasker(seq)
    for r in response:
        print(r['sequence'])
run_mlm("He said that Mozart, Beethoven, and Chopin never died. They simply became [MASK].")
run_mlm("Late night talking or [MASK]?")



