#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=Test
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:1
#SBATCH --account=a_smart
#SBATCH -o slurm.output
#SBATCH -e slurm-%A.error

module load miniconda3
conda activate unlimiformer
cd /scratch/project/smart/unlimiformer-07-dec
which python
srun python src/run.py \
    src/configs/training/base_training_args.json \
    src/configs/data/gov_report_kg_comb.json \
    --output_dir output_train_bart_base_local/ \
    --learning_rate 1e-5 \
    --model_name_or_path facebook/bart-base \
    --eval_steps 1000 --save_steps 1000 \
    --per_device_eval_batch_size 1 --per_device_train_batch_size 2 \
    --extra_metrics bertscore \
    --unlimiformer_training \
    --max_source_length 16384 \
    --test_unlimiformer \
    --eval_max_source_length 999999  --do_eval=True \
    > output/output${SLURM_JOB_ID}.txt


##python src/run.py \  
##    src/configs/training/base_training_args.json \
##    src/configs/data/gov_report.json \
##    --output_dir output_train_bart_base_local/ \
##    --learning_rate 1e-5  \ 
##    --model_name_or_path facebook/bart-base \
##    --test_unlimiformer  
##    --eval_max_source_length 999999 \
##    --do_eval=True     --eval_steps 1000 --save_steps 1000 \
##    --per_device_eval_batch_size 1 \
##    --per_device_train_batch_size 2 \
##    --extra_metrics bertscore \
##    --unlimiformer_training \
##    --max_source_length 16384
