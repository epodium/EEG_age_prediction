#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=example@example.com

#Loading modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

pip install --user pandas
pip install --user tables
pip install --user scikit-learn
pip install --user sklearn-rvm

#Copy input file to scratch
cp -r $HOME/data_processed_ML "$TMPDIR"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/ML_train.py "$TMPDIR"/data_processed_ML $HOME/trained_models
