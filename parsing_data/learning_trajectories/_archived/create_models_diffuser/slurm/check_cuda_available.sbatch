#!/bin/bash
#SBATCH --job-name=test_cuda_available
#SBATCH --partition=drl
#SBATCH --qos=drl-free-cycles
#SBATCH --time=08:00:00 # 8 hours
#SBATCH --output=/data/scratch/delpreto/ActionSense/slurm/logs/job_output_%j.log
#SBATCH --error=/data/scratch/delpreto/ActionSense/slurm/logs/job_error_%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M

ERROR_LOG="/data/scratch/delpreto/ActionSense/slurm/logs/job_error_${SLURM_JOB_ID}.log"
OUTPUT_LOG="/data/scratch/delpreto/ActionSense/slurm/logs/job_output_${SLURM_JOB_ID}.log"

########################
########################

echo ""
echo ""
date
echo ""
echo "Running my job on $HOSTNAME"
echo "HOME is $HOME"
echo "Current directory:"
pwd
echo ""
echo "Python version and location:"
python3 --version
which python3
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Error  file: ${ERROR_LOG}"
echo "Output file: ${OUTPUT_LOG}"

########################
# Specify folders to use
########################

export HOME=/data/scratch/delpreto/ActionSense
python_packages_dir="/data/scratch/delpreto/ActionSense/python_packages"
python_virtual_environment_path="/data/scratch/delpreto/ActionSense/python_venv_actionsense"

cd "${HOME}"

echo ""
echo "HOME is now $HOME"
echo "Current directory:"
pwd
echo ""
echo "-----------------------------------------"
echo "-----------------------------------------"

########################
# Set up the virtual environment.
########################

echo ""
echo "Installing virtualenv"
time python3 -m pip install --target "${python_packages_dir}" virtualenv
export PATH="$PATH:/data/scratch/delpreto/ActionSense/.local/bin"

echo ""
echo "Creating a virtual environment if needed"
virtualenv "${python_virtual_environment_path}"

echo ""
echo "Sourcing the virtual environment"
source "${python_virtual_environment_path}/bin/activate"

echo ""
echo "-----------------------------------------"

########################
# Install packages if needed.
########################

echo ""
echo "-----------------------------------------"
echo "Installing packages"
echo "-----------------------------------------"
echo ""
time python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

########################
# Run the main script
########################

echo ""
echo ""
echo "================================================="
echo "================================================="
echo "================================================="
echo ""
echo ""

{ time python3 /data/scratch/delpreto/ActionSense/slurm/check_cuda_available.py; } 2> >(tee -a "${ERROR_LOG}") >> "${OUTPUT_LOG}"

echo ""
echo "-----------------------------------------"
echo ""

wait  # Ensure all output is written before proceeding

echo ""
echo "================================================="
echo "================================================="
echo "================================================="
echo ""
echo ""

deactivate # maybe not needed since the script is terminating

echo "Done!"
echo ""
echo ""
