## ATSEN

The source code used for [Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and Fine-grained Student Ensemble](https://arxiv.org/abs/2212.06522),published in AAAI 2023.

## Framework

![](https://github.com/zenhjunpro/ATSEN/blob/main/image/%E6%A1%86%E6%9E%B6.png)

## Requirements

At least one GPU is required to run the code.

enviroment:

- apex==0.1
- python==3.7.4
- pytorch==1.6.0
- tranformers==4.19.3
- numpy==1.21.6
- tqdm==4.64.0
- ...

you can see the enviroment in requirements.txt or you can use `pip3 install -r requirements.txt` to create environment

## Benchmark

The reuslts (entity-level F1 score) are summarized as follows:

|  Method   |  CoNLL03  | OntoNotes5.0 |  Twitter  |
| :-------: | :-------: | :----------: | :-------: |
|   BOND    |   81.48   |    68.35     |   48.01   |
|   SCDL    |   83.69   |    68.61     |   51.09   |
| **ATSEN** | **85.59** |  **68.95**   | **52.46** |

## Motivation

![](https://github.com/zenhjunpro/ATSEN/blob/main/image/2.png)

```
def _update_mean_model_variables(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    m = get_param_momentum(param_momentum,global_step,t_total)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.8:
            pass
        else:
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data
            
def get_param_momentum(param_momentum,current_train_iter,total_iters):

    return 1.0 - (1.0 - param_momentum) * (
        (math.cos(math.pi * current_train_iter / total_iters) + 1) * 0.5
    )
```



## Reproducing the Results

We provide three bash scripts `run_conll03.sh`,`run_ontonotes5.sh`,`run_webpage.sh` for running the model on the three datasets.

you can run the code like:

```
 sh <run_dataset>.sh <GPU ID> <DATASET NAME>
```

e.g.

```
sh run_conll03.sh 0,1 conll03
```

The bash scripts include arguments,they are important and need to be set carefully:

- `GPUID`:It means whice device you will use.We  use two devices in our experiment,you can use more.
- `DATASET` :It means which dataset you will use.You can run your own dataset if you create the dataset as follows.
- `LR` :This parameter refers to the learning rate, adjusted for different data sets.
- `WARMUP` :This parameter also needs to be adjusted according to different datasets.
- `BEGIN_EPOCH` :The number of rounds of training in the first phase is different for different datasets.
- `PERIOD` :The number of rounds of training in the first phase is different for different datasets.
- `THRESHOLD`:This parameter is the threshold mentioned in the text, which is generally set to 0.9.
- `TRAIN_BATCH`:This parameter is the size of the training batch, you can adjust it according to the number of devices you have, and the final result of different training batches is different。
- `EPOCH`:This argument means the number of training times for the entire experiment. The general setting is `50`.
- `LABEL_MODE`:The value of this parameter is `Soft `or `Hard`. In general, choose `Soft `, but choose `Hard`  on the Twitter dataset.
- `SEED:`This can help you get the same result with the same arguments. We usually set this to 0.
- `EVAL_BATCH`:This argument only affects the speed of the algorithm; use as large evaluation batch size as your GPUs can hold.We use `32` as usually.

## Running on New Datasets



## Models

We provide the models in this [page]().You can reproduce the results of the experiment.The result  we do can see in [log.txt](https://github.com/zenhjunpro/ATSEN/blob/main/log.txt)

## Notes and Acknowledgments

The implementation is based on https://github.com/AIRobotZhang/SCDL
