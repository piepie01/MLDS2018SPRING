# 1-2 Bonus

```bash
python3 main.py
```

* After running the task above, the result will generate in

```
./0
```

With the format

```
#in ./0
[x] [y]
[x] [y]
[x] [y]
.
.
.
```

* Note that there will be $4001 \times 8$ data in the file.(8是train到一半後開始打點後的epoch數目，所以data裡面有8筆資料是model在8次epoch中走到的位置，index是第0, 4001, 8002, 12003, 16004, 20005, 24006, 28007筆資料，其他都是隨機打點的結果)

* Note : train data and test data are in

* ```bash
  ./data/
  ```

  