# ETHZ CIL Text Classification 2021

## Project Setup

Clone git repository from [github](https://github.com/bruggisser/cil-project): 
```
git clone git@github.com:bruggisser/cil-project.git
```
Enter project directory
```
cd cil-project
```

Either run `source ./setup.sh` to set up the project or follow the steps below.

### Manual setup
Download the data from [kaggle](https://www.kaggle.com/c/cil-text-classification-2021/data).

Copy the data into the project directory
```
cp download/location/* data/
```

Download the pretrained glove twitter word vectors from [stanford.edu](https://nlp.stanford.edu/projects/glove/).

Copy the file _glove.twitter.27B.200d.txt_ to `data/glove/`.

(Optional) Add configuration for _comet.ml_ to the directory `config/`.

Setup on leonhard cluster
```
source ./leonhard_project.sh
```

### Execute project
> Some models take very long to execute, especially BERT and SVM take more than 24h each. Available models are listed below.
```
bsub -n 1 -W 24:00 -R "rusage[mem=12288, ngpus_excl_p=1]" python main.py
```
### Create plots
To generate plots included in the paper run
```
source ./leonhard_r.sh
Rscript create_plots.R
```


### Expected format for comet.json
```
{ 
      "api_key": "<your_api_key>",  
      "project_name": "<project_name>",  
      "workspace": "<workspace>"  
}
```

### Models and output files
The following models are available (with corresponding file prefixes in brackets): support vector machines (svm), 
Naive Bayes (nb), convolutional neural network (cnn), Long short term memory (lstm) and bert. 

- The results for kaggle are stored in the file `(modelname)_result.csv`.
- The results are aggregated with one of the following approaches: vote, average (avg) and random forest (rf). 
The corresponding results are stored in the file `(aggregation_name)_result.csv`.
- Intermediate results are stored in the files `(modelname)_result_exact.csv` for the kaggle challenge and 
`(modelname)_validation_set.csv` for the validation set and joined in the files `unified_result_exact.csv`, 
`unified_validation_set.csv` accordingly.

## Models
Six models are implemented. They can be selected in the config (`models`) with the following keys:

Model name | value
--- | ---
BERT | bert
Convolutional Neural Network | cnn
Logistic Regression | lr
Long short-term memory | lstm
Naive Bayes | nb
Support Vector Machine | svm

There are three ways implemented to aggregate the results (`aggregations`):

Aggregation | value
--- | ---
Random Forest | rf
Majority vote | vote
Average | avg


## Logs
* Logs are written to `logs/log.out`.
* For each model, an experiment in [comet.ml](https://www.comet.ml/) is started.

## Result files
All result files are stored in the archive `results.zip`. For each model, three files are included as explained above.