# 1-1 **Train on Actual Tasks**

```bash
python3 shallow.py
python3 median.py
python3 deep.py
```

* After running the task above, the result will generate in

```
#shallow
loss/1
acc/1

#median
loss/2
acc/2

#deep
loss/3
acc/3
```

with the format, for example

```
#in loss/1
[epoch num] [loss]
[epoch num] [loss]
[epoch num] [loss]
[epoch num] [loss]
.
.
.
```

```
#in acc/1
[epoch num] [acc]
[epoch num] [acc]
[epoch num] [acc]
[epoch num] [acc]
.
.
.
```

* Note : train data and test data are in

* ```bash
  ./data/
  ```

##Plot the result
```bash
python3 pic_acc.py
python3 pic_loss.py
```
