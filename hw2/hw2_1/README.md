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
python3 model_seq2seq.py [training data directory such as (MLDS_hw2_1_data)]
#the model, word2index, index2word will be saved in ./save/
```

* Testing

```bash
./hw2_seq2seq.sh [testing data dir] [output file]
```

* File relation
  * Training file : model_seq2seq.py, dataset.py, ultra_model.py
  * Testing file : predict.py, dataset.py, ultra_model.py
