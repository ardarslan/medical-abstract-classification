# Data Location
Create a folder named `data/PubMed_200k_RCT` in the main directory.
Put `train.txt`, `dev.txt`, `test.txt` inside.

# Environment Setup

We will create two separate environments. First one is for running task 1, 2 and 3. Second one is for running additional doc2vec experiment. This is due to a mismatch between gensim and tensorflow versions that we want to use.

```
conda env create -f environment.yml
conda env create -f environment_doc2vec.yml
```

# Run Task 1, 2 or 3
```
conda deactivate
conda deactivate
conda activate ml4hc_project2
cd src
```

## Task 1

The commands in _Round X_ are replicated as we run them with multiple
different random seeds. Each will produce a log with the results that
need to be average over for the results reported in the report.

### Round 1: Lemmatization vs. No Lemmatization

First, run the experiments with lemmatization enabled:

```
python task1.py --classifier lightgbm --retrain_task --train_fraction 0.1 --seed 31 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --train_fraction 0.1 --seed 41 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --train_fraction 0.1 --seed 59 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --train_fraction 0.1 --seed 26 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --train_fraction 0.1 --seed 53 --norm_paper_ids
```

_Then, in the `data/PubMed_200k_RCT` folder, delete all files that end
with `.pickle`._ These are preprocessed versions of the corpus that
are cached to improve the runtime of the experiments. They _must_ be
deleted before running the experiments without lemmatization:

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --norm_paper_ids
```

> The use of `--lemma` above may be counter-intuitive but is correct given
> our code base.

> N.B.: Do **NOT** delete the cached `.pickle` files from here on.

### Round 2: Normalizations

Numbers:

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --number2hashsign

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --number2hashsign

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --number2hashsign

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --number2hashsign

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --number2hashsign
```

Paper IDs:

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --norm_paper_ids

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --norm_paper_ids
```

Units:

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --norm_units

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --norm_units

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --norm_units

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --norm_units

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --norm_units
```

Technical Abbreviations:

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --norm_tech_abbr
```

### Round 3: Normalization Combinations

```
python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 31 --norm_paper_ids --number2hashsign --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 41 --norm_paper_ids --number2hashsign --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 59 --norm_paper_ids --number2hashsign --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 26 --norm_paper_ids --number2hashsign --norm_tech_abbr

python task1.py --classifier lightgbm --retrain_task --lemma --train_fraction 0.1 --seed 53 --norm_paper_ids --number2hashsign --norm_tech_abbr
```

_Now, in the `data/PubMed_200k_RCT` folder, delete all files that end with `.pickle` again._

### Final FCNN Classifier

```
python task1.py --classifier mlp --retrain_task --lemma --seed 31 --norm_paper_ids --number2hashsign --norm_tech_abbr --top_k 2000
```

## Task 2

First deactivate the currently active conda environments (Even the base environment):

```
conda deactivate
conda deactivate
```

Then execute the following lines:
```
module load gcc/8.2.0 python/3.8.5 eth_proxy
conda activate ml4hc_project2_doc2vec
python task2.py --epochs 10
```

## Task 3
First deactivate the currently active conda environments (Even the base environment):
```
conda deactivate
conda deactivate
```

Then execute the following lines:
```
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
conda activate ml4hc_project2
```

### Bert base: distilbert-base-uncased, Bert base output: mean of word embeddings, Fine tune bert base: False
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base distilbert-base-uncased --bert_base_output_mode mean --checkpoint_dir /path/to/checkpoints/directory --epochs 3
```

### Bert base: distilbert-base-uncased, Bert base output: cls embedding, Fine tune bert base: False
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base distilbert-base-uncased --bert_base_output_mode cls --checkpoint_dir /path/to/checkpoints/directory --epochs 3
```

### Bert base: distilbert-base-uncased, Bert base output: mean of word embeddings, Fine tune bert base: True
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base distilbert-base-uncased --bert_base_output_mode mean --checkpoint_dir /path/to/checkpoints/directory --epochs 3 --finetune_bert
```

### Bert base: distilbert-base-uncased, Bert base output: cls embedding, Fine tune bert base: True
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base distilbert-base-uncased --bert_base_output_mode cls --checkpoint_dir /path/to/checkpoints/directory --epochs 3 --finetune_bert
```

### Bert base: emilyalsentzer/Bio_ClinicalBERT, Bert base output: mean of word embeddings, Fine tune bert base: False
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base emilyalsentzer/Bio_ClinicalBERT --bert_base_output_mode mean --checkpoint_dir /path/to/checkpoints/directory --epochs 3
```

### Bert base: emilyalsentzer/Bio_ClinicalBERT, Bert base output: cls embedding, Fine tune bert base: False
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base emilyalsentzer/Bio_ClinicalBERT --bert_base_output_mode cls --checkpoint_dir /path/to/checkpoints/directory --epochs 3
```

### Bert base: emilyalsentzer/Bio_ClinicalBERT, Bert base output: mean of word embeddings, Fine tune bert base: True
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base emilyalsentzer/Bio_ClinicalBERT --bert_base_output_mode mean --checkpoint_dir /path/to/checkpoints/directory --epochs 3 --finetune_bert
```

### Bert base: emilyalsentzer/Bio_ClinicalBERT, Bert base output: cls embedding, Fine tune bert base: True
```
bsub -n 2 -W 24:00 -R "rusage[mem=16384, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=32768]" python task3.py --bert_base emilyalsentzer/Bio_ClinicalBERT --bert_base_output_mode cls --checkpoint_dir /path/to/checkpoints/directory --epochs 3 --finetune_bert
```

# Run Doc2Vec Tasks
First deactivate the currently active conda environments (Even the base environment):
```
conda deactivate
conda deactivate
```

Then execute the following lines:
```
module load gcc/8.2.0 python/3.8.5 eth_proxy
conda activate ml4hc_project2_doc2vec
```

Below, DM should be 0 or 1. EMBEDDING_SIZE should be 100 or 300. WINDOW_SIZE should be 5 or 10.
DM=0 is for using Distributed Bag of Words algorithm to train Doc2Vec. DM=1 is for using Distributed Memory algorithm.

## Doc2Vec + MLP
```
bsub -n 2 -W 24:00 -R "rusage[mem=4096]" python doc2vec.py --classifier mlp --vector_size EMBEDDING_SIZE --doc2vec_epochs 20 --mlp_epochs 20 --dm DM --window_size WINDOW_SIZE --learning_rate 0.001 --checkpoint_dir /path/to/checkpoints/directory
```
