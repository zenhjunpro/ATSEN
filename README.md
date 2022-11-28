## Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and  Fine-grained Student Ensemble（AAAI 2023）



## ATSEN Framework

![https://github.com/zenhjunpro/ATSEN/blob/main/image/Motivation.png]()

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

- `student1_model_name_or_path` :

- `student2_model_name_or_path `:

- `output_dir`:

- `tokenizer_name`:

- `cache_dir `:

- `max_seq_length `:

- `learning_rate`:

- `weight_decay `:

- `adam_epsilon`:

- `adam_beta1 `:

- `adam_beta2 `:

- `max_grad_norm `:

- `num_train_epochs `:

- `warmup_steps `:

- `per_gpu_train_batch_size `:

- `per_gpu_eval_batch_size `:

- `gradient_accumulation_steps `:

- `logging_steps`:

- `save_steps `:

- `evaluate_during_training `:

- `seed `:

- `overwrite_output_dir`:

- `mean_alpha`:

- `self_learning_label_mode`:

- `self_learning_period`:

- `model_type `:

- `begin_epoch `:

- `do_train`:

- `dataset`:

- `threshold `:

- `al `:

- `bate `:

## Models

We provide the models in this page.You can reproduce the results of the experiment.The result  we do can see in [log.txt](https://github.com/zenhjunpro/ATSEN/blob/main/log.txt)

## Notes and Acknowledgments

The implementation is based on https://github.com/AIRobotZhang/SCDL
