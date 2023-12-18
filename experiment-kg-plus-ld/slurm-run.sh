#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --job-name=Test
#SBATCH --time=76:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:1
#SBATCH --account=a_smart
#SBATCH -o slurm.output
#SBATCH -e slurm.error

micromamba activate unlimiformer-final
CWD=${1}
RUNTYPE=_${2}
cd /scratch/project/smart/${CWD}
if [ ! -e output ]; then
  mkdir output
fi

which python
echo ${SLURM_JOB_ID}
srun python src/run.py \
    src/configs/training/base_training_args.json \
    src/configs/data/gov_report${RUNTYPE}.json \
    --output_dir output_train_bart_base_local/ \
    --learning_rate 1e-5 \
    --model_name_or_path facebook/bart-base \
    --per_device_eval_batch_size 1 --per_device_train_batch_size 2 \
    --do_eval=True \
    --eval_steps 1000 --save_steps 1000 \
    --extra_metrics bertscore \
    --unlimiformer_training \
    --max_source_length 16384 \
    --eval_max_source_length 999999 \
    --test_unlimiformer  \
    > output/output${SLURM_JOB_ID}.txt

cp slurm.error slurm-${SLURM_JOB_ID}.txt

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
##    --max_source_length 1024 \
