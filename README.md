# huggingface-domain-adaptation
Domain adaptation from scratch


Simple huggingface distilbert MLM for domain adaptation, the dataset I decided to try was westworld season 1.  

### Steps
1. Clean and compile the text from each episode with `read_westworld.py`
2. Run `main.py`
3. Additionally if you want you can save your model with `model.save(path)`


### :star: Example outputs :star:
Target:
```
he said that mozart, beethoven, and chopin never died. they simply became music.
```
Input:
```
he said that mozart, beethoven, and chopin never died. they simply became [MASK].
```
Output:
```
he said that mozart, beethoven, and chopin never died. they simply became combatants.
he said that mozart, beethoven, and chopin never died. they simply became boxers.
he said that mozart, beethoven, and chopin never died. they simply became rivals.
```
