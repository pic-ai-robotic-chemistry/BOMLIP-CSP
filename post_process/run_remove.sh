#!/bin/sh -l
#An example for GPU job.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J csp_test
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1 -n 8
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_8a100
#SBATCH -t 3-00:00:00


echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo =========================================================   
echo Job output begins                                           
echo -----------------                                           
echo


python duplicate_remove3.py


echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
