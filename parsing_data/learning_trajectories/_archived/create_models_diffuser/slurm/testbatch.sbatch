#!/bin/bash
#SBATCH --job-name=my_very_own_job 
#SBATCH --partition=drl
#SBATCH --qos=drl-free-cycles
#SBATCH --time=08:00:00 # 8 hours
#SBATCH --output=/data/scratch/delpreto/job_output_%j.log
#SBATCH --error=/data/scratch/delpreto/job_error_%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M

#load_my_dependencies.py

date
echo "Running my job on $HOSTNAME"

# put the job and data on the local node for better performance
#sbcast /data/scratch/$USER/my_training_data /tmp/my_training_data
#sbcast /data/scratch/$USER/my_training_job.py /tmp/my_training_job.py

python3 /data/scratch/delpreto/hello_slurm.py