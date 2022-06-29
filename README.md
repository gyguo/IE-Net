# IE-Net
official implementation of "Eliminating Indefiniteness of Clinical Spectrum for Better Screening COVID-19", accepted by IEEE Journal of Biomedical and Health Informatics (JBHI2021). 


### Requirements

To install the environment via Anaconda:

```setup
conda env create -f environment.yaml
```

### Prepare Dataset

the original dataset is in https://www.kaggle.com/einsteindata4u/covid19

```
cd ./data
unzip feature_original.zip
```

### Training and Evaluation of IE-Net

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

The comparison experiments are 

```
cd ./comparison
python comparison_experiments_fill_0_multimetric.py
```

### Results

For 10-fold cross-validation, our model achieves high performance on the COVID-19 Clinical dataset. The table below shows the **results in the paper**.

|                  |  Accuracy  |   Recall   | Precision  |    AUC     |
| ---------------- | :--------: | :--------: | :--------: | :--------: |
| Results in paper | 94.80±1.98 | 92.79±3.07 | 92.97±3.06 | 94.93±2.00 |

Recently, we introduce F1-score as the metric for selecting the best model, **the performance in terms of Recall and Precision has improved.** As mentioned in the paper, Recall is the most important metric in this paper.

|            |  Accuracy  |     Recall     |   Precision    |    AUC     |     F1     |
| ---------- | :--------: | :------------: | :------------: | :--------: | :--------: |
| Results F1 | 94.05±2.17 | **95.99±3.69** | **94.42±2.26** | 90.50±3.76 | 93.81±2.52 |


