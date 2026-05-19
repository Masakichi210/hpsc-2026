#!/bin/bash
#$ -cwd
#$ -V
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N hpsc_bench16
#$ -o bench_logs/qsub_$JOB_ID.out
#$ -e bench_logs/qsub_$JOB_ID.err

module load cuda/12.8.0
bash run_bench.sh 16_dbuf
