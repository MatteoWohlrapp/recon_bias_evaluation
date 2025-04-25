#!/usr/bin/bash

#SBATCH -J "mri_inr"   # job name
#SBATCH --time=4-00:00:00   # walltime
#SBATCH --output=/vol/miltank/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/vol/miltank/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.err  # Standard error of the script
#SBATCH --mem=64G
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1              # Request 1 specific GPU (e.g., A100)

# load python module
. "/opt/anaconda3/etc/profile.d/conda.sh"

# activate corresponding environment
conda deactivate
conda activate pix

cd "/vol/miltank/projects/practical_SoSe24/mri_inr/matteo/code/dfci_evaluation/prediction"

python prediction.py --opt=options/chex_diffusion_tum.yml
