# 1-2 **Visualize the optimization process**

```bash
python3 main.py
```

* After running the task above, the result will generate in

```
acc/0 weight/0 weight/layer_0
acc/1 weight/1 weight/layer_1
acc/2 weight/2 weight/layer_2
acc/3 weight/3 weight/layer_3
acc/4 weight/4 weight/layer_4
acc/5 weight/5 weight/layer_5
acc/6 weight/6 weight/layer_6
acc/7 weight/7 weight/layer_7
```

* For example, **weight/0** is whole model's data and **weight/layer_0** is layer2's data

With the format, for example

```
#in acc/0
[acc]
[acc]
[acc]
.
.
.
```

```
#in weight/0
[x] [y]
[x] [y]
[x] [y]
.
.
.
```

```
#in weight/layer_0
[x] [y]
[x] [y]
[x] [y]
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
python3 pic_weight.py
python3 pic_layer2_weight.py
```
