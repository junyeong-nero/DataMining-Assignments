# DataMining Assignment1 Topic2 (FP-growth)

- author : junyeong song (magicwho@unist.ac.kr)
- prof : junghoon kim (Junghoon.kim@unist.ac.kr)

## How to run

```
javac A1_G5_t2
java A1_G5_t2 [fileDir] [minimumSupport] [debug]
```

## Structure

- `A1_G5_t2.java` : a class to implement this assignemnt
- `CSVReader.java` : a class to convert `groceries.csv` to frequency table
- `FPGrowth.java` : a class to implement FP-growth algorithm

## Experiments

- `./test_minsup_size.sh` : test with minimum support
- `./test_transaction_size.sh` : test with transaction size
