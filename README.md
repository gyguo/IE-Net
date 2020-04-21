# IE-Net
 
## This repository is the official implementation of [Eliminating Indefiniteness of Clinical Spectrum for Better Screening COVID-19]. 


## Requirements

To install the environment via Anaconda:

```setup
conda env create -f environment.yaml
```


## Training

To train the models in this paper, run this command:

```train
cd ./tools
python main_multiple.py
```

## Evaluation

To evaluate the models in this paper, run this command:

```eval
cd ./tools
python eval.py
```


## Pre-trained Models & Visualizations

The best performed model is in 
```
./checkpoint_best.pth
```

The comparison experiments are in
```
./comparison
```

## Results

Our model achieves the following performance on COVID-19 dataset:

|                    |    Accuracy       |    Recall      |
| ------------------ |------------------ | -------------- |
|    Our model       |     94.56%        |    90.48%      |
