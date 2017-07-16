# PowerGrid


### Installation

Install [Anaconda] Python 2.7

To run the python script:
```sh
$ python anomaly_detection
```

To run the python script with debugging mode:
```sh
$ python anomaly_detection -v
```

To run the python script using different csv files:
```sh
$ python anomaly_detection --train data/train.csv --test data/test_v1.csv
```

To see all acceptable commands:
```sh
$ python anomaly_detection -h
```

> Due to the large size of the csv files contributing to the problem of limited github repo size, the files were removed. The contributor must manually put the test_*.csv files in data/test and train.csv in data/train directory



[//]: # (Reference links used in the doc)

[Anaconda]: <https://www.continuum.io/downloads/>