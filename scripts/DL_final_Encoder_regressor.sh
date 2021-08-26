#!/bin/bash
#Set job requirements
#SBATCH -p gpu_shared
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=example@example.com

#Loading modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/11.0.2-GCC-9.3.0
module load cuDNN/8.0.3.33-gcccuda-2020a
module load TensorFlow/2.3.1-fosscuda-2020a-Python-3.8.2
pip install --user zarr
pip install --user pandas
pip install --user scikit-learn
pip install --user tensorflow
pip install tensorflow-addons

python $HOME/check-gpu.py
if [ $? -ne 0 ]; then
    exit 1
fi

#Copy input file to scratch
cp -r $HOME/data_processed_DL "$TMPDIR"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/DL_final_Encoder_regressor.py "$TMPDIR"/data_processed_DL $HOME/trained_models
