#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=TFrecords   # Job name
#SBATCH --output=TFrecords.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=TFrecords.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=1     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=16:01:00   # walltime
#SBATCH --mem=56G   # memory per NODE
#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa200403    # Replace with your system project

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules

module load gnu/6.4.0    
module load java/1.8.0
module load cuda/9.2.148
module load intel/18
module load intelmpi/2018 
module load proj4/4.9.3
module load openblas/0.3.6/gnu
module load tensorflow/1.10.1gpu

## RUN YOUR PROGRAM ##
srun pip3 list
#srun python prep_splits_BigEarthNet-19.py -r /work2/pa20/ipapout/DownloadedData/BigEarthNet-v1.0 -o /work2/pa20/ipapout/gitSpace/MyWork -n [splits/test3.csv] -l tensorflow

srun python prep_splits_BigEarthNet-19.py -r /users/pa20/ipapout/DownloadedData/BigEarthNet-v1.0 -o /work2/pa20/ipapout/gitSpace/TF1.10.1gpu_Py3/MyWork -n /work2/pa20/ipapout/gitSpace/bigearthnet-19-models/splits/val.csv -l tensorflow
