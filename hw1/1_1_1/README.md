# 1-1 Simulate a Function

```bash
python3 shallow.py
python3 medium.py
python3 deep.py
```

* After running the task above, the result will generate in

```
result_data/shallow_loss.txt
result_data/shallow_predict.txt

result_data/medium_loss.txt
result_data/medium_predict.txt

result_data/deep_loss.txt
result_data/deep_predict.txt
```

with the format, for example

```
#in shallow_loss.txt
[epoch num] [loss]
[epoch num] [loss]
[epoch num] [loss]
[epoch num] [loss]
.
.
.
```

```
#in shallow_predict.txt
[test_x] [predict_y]
[test_x] [predict_y]
[test_x] [predict_y]
[test_x] [predict_y]
.
.
.
```

* Note : train data and test data are in

* ```bash
  ./data/
  ```

###Plot the result
```bash
python3 pic_loss.py
python3 pic_predict.py
```
