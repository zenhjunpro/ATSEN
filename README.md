## Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and  Fine-grained Student Ensemble（AAAI 2023）



## ATSEN Framework

![](https://github.com/zenhjunpro/ATSEN/blob/main/image/%E6%A1%86%E6%9E%B6.png)

## Requirements

- apex==0.1
- python==3.7.4
- pytorch==1.6.0
-  tranformers==4.19.3
- numpy==1.21.6
- tqdm==4.64.0
- ...

you can see the enviroment in requirements.txt or you can use `pip -r requirements.txt` to create environment

## Benchmark

The reuslts (entity-level F1 score) are summarized as follows:

|  Method   |  CoNLL03  | OntoNotes5.0 |  Twitter  |
| :-------: | :-------: | :----------: | :-------: |
|   BOND    |   81.48   |    68.35     |   48.01   |
|   SCDL    |   83.69   |    68.61     |   51.09   |
| **ATSEN** | **85.59** |  **68.95**   | **52.46** |

## How to run

```
sh <run_dataset>.sh <GPU ID> <DATASET NAME>
```

e.g.

```
sh run_conll03.sh 0,1 conll03
```

```
sh run_ontonotes5.sh 0,1 ontonotes5
```

```
sh run_twitter.sh 0,1 twitter
```

## Models



## Notes and Acknowledgments

The implementation is based on https://github.com/AIRobotZhang/SCDL
