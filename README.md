# IE-Net

## This repository is the official implementation of [Eliminating Indefiniteness of Clinical Spectrum for Better Screening COVID-19]. 


### Requirements

To install the environment via Anaconda:

```setup
conda env create -f environment.yaml
```

### Dataset

https://www.kaggle.com/einsteindata4u/covid19

### Training and Evaluation

To train the models in this paper, run this command:

```train
cd ./tools
python main_multiple.py
```


### Best Models

The best performed model is in 
```
./checkpoint_best.pth
```

### Comparison 

The comparison experiments are in

```
cd ./comparison
python comparison_experiments_fill_0_multimetric.py
```

### Results

For 10-fold cross validation, our model achieves the following performance on COVID-19 Clinical  dataset:

|                    |    Accuracy       |    Recall      |
| ------------------ |------------------ | -------------- |
|    Our model       |   94.80±1.98      |  92.79±3.07    |

## gitee
code has also been released in [gitee](https://gitee.com/gyguo95/IE-Net)
