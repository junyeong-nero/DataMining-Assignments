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

## Datasets

1. In experiment, we use SMPF datasets in `data/SMPF`
   - original dataset size is 100k, therefore divide dataset into 10k scale.
   - you can check from `10k.csv` to `100k.csv` 

2. In addition, we also test our implemented codes with other datasets
   - online_retail datasets V1 from [UCI](https://archive.ics.uci.edu/dataset/352/online+retail)
   - online_retail datasets V2 from [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
   - we tried to preprocessing this datasets and used it.
     - you can check `data/online_retail_1/` and `data/online_retail_2/`

## Experiments

- `./test_minsup_size.sh`
  - test with minimum support (with 100k dataset, minimum support from 0.04 to 0.5)
- `./test_transaction_size.sh`
  - test with transaction size (10k scales)
