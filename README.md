## Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and  Fine-grained Student Ensemble（AAAI 2023）



## ATSEN Framework

![](https://github.com/zenhjunpro/ASTEN/blob/main/image/%E6%A1%86%E6%9E%B6.png)

## Requirements

- apex==0.1
- python==3.7.4
- pytorch==1.6.0
-  tranformers==4.19.3
- numpy==1.21.6
- tqdm==4.64.0
- ...

you can see the enviroment in requirements.txt or you can use `pip -r requirements.txt` to create environment

## How to run

```yaml
sh run_conll03.sh <GPU ID> <DATASET NAME>
```

e.g.

```yaml
sh run_conll03.sh 0,1 conll03
```

## Notes and Acknowledgments

The implementation is based on https://github.com/AIRobotZhang/SCDL
