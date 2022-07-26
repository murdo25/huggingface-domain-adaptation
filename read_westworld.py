import re
import os

# raw_data = 'data/westworld/westS1E1.srt'
# raw_data = 'data/westworld/Westworld S01E06 The Adversary.DVDRip.NonHI.en.HBO.srt'
data_path = 'data/westworld/'



def read_and_clean(raw_data):
    turns = []
    with open(raw_data, 'r') as f:
        full_file = f.read()

        # 01:08:03,112 --> 01:08:04,112
        # splits = re.split("\n[0-9]*\n([0-9]*:)*[0-9]*,[0-9]* --> ([0-9]*:)*[0-9]*,[0-9]*\n", full_file)
        splits = re.split("\n[0-9]*\n[0-9]*:[0-9]*:[0-9]*,[0-9]* --> [0-9]*:[0-9]*:[0-9]*,[0-9]*\n", full_file)

        print("total splits:", len(splits))
        splits.pop()
        splits.pop(0)

        def clean(s):
            s = s.replace("<i>", "")
            s = s.replace("</i>", "")
            # s = s.split('</i>')[0]
            s = s.rstrip()
            s = s.replace("\n", " ")
            return s

        for i,s in enumerate(splits):
            clean_turn = clean(s)
            print(i, clean_turn)
            turns.append(clean_turn)
    return turns




files = [data_path + f for f in os.listdir(data_path)]

clean_turns = []
for f in files:
    clean_turns.extend(read_and_clean(f))


def build_windows(clean_turns, window_len=6):
    # I want to to create windowed copies of my dataset 
    # Iterate over my dataset for len(ds)-n_turn windows 
    windows = []
    
    for i in range(len(clean_turns) -window_len +1):
        # how do I get the first n turn
        windows.append(" ".join(clean_turns[i:window_len+i]))
    return windows


dataset = build_windows(clean_turns, window_len=6)
print("Total dialog windows:", len(dataset))
for i in range(10):
    print("---"*20)
    print(dataset[i])


import pandas as pd
df = pd.DataFrame({"input_text":dataset}) 
df.to_csv('data/westworldS1_windowed.csv')

