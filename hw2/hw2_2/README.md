# 2_1 Video Caption

* use library

```bash
pytorch 0.3.0/ torchvision 0.2.0
numpy 1.14.3
termcolor

standard library:
sys
os
copy
random
pickle
json
```

* Training

```bash
python3 model_seq2seq.py [training data (txt)]
#the model, word2index, index2word will be saved in ./save/
```

* Testing

```bash
./hw2_seq2seq.sh [testing data (txt)] [output file]
```

* File relation
  * Training file : model_seq2seq.py, dataset.py, model.py
  * Testing file : predict.py, dataset.py, model.py
