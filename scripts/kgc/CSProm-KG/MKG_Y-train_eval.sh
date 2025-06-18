#!/bin/bash
#SBATCH --job-name=MKG_Y-train-csprom_kg
#SBATCH --output=/blue/daisyw/ma.haodi/MMKGC/CSProm-KG/logs/MKG_Y/mkg_y-train-better_setting_2-batch_256-ent_v0-epoch_200-1.out
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,REQUEUE
#SBATCH --mail-user=ma.haodi@ufl.edu
#SBATCH --nodes=1 # nodes allocated to the job
#SBATCH --ntasks-per-node=8 # one process per node
#SBATCH --cpus-per-task=1 # the number of CPUs allocated per task, i.e., number of threads in each process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=20gb
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

echo "starting job"

module load conda/24.3.0 cuda/11.4.3
echo “load conda/24.3.0 cuda/11.4.3”
conda activate csprom
echo activated csprom

T1=$(date +%s)
echo ${date}

echo "Train on mkg_y"


echo "Job id:" $SLURM_JOB_ID

######## baseline setting ########
# # train commandline:
# python main.py -dataset MKG_Y \
#                 -batch_size 256 \
#                 -pretrained_model bert-large-uncased \
#                 -epoch 200 \
#                 -desc_max_length 0 \
#                 -lr 5e-4 \
#                 -prompt_length 10 \
#                 -alpha 0.1 \
#                 -n_lar 8 \
#                 -label_smoothing 0.1 \
#                 -embed_dim 144 \
#                 -k_w 12 \
#                 -k_h 12 \
#                 -alpha_step 0.00001 \
#                 -desc_path entityid2description.txt \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-$(date +'%Y-%m-%d-%H-%M-%S')-baseline-batch_256-epoch_200


######## mm setting 2 ########
# train commandline:
python main.py -dataset MKG_Y \
                -batch_size 256 \
                -pretrained_model bert-large-uncased \
                -epoch 200 \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 144 \
                -k_w 12 \
                -k_h 12 \
                -alpha_step 0.00001 \
                -use_mm \
                -desc_path entityid2mm_description-setting_2-concate-0.txt \
                -mm_rel_triples qwen-MKG_Y-rel_context-ent_v0-v2.json \
                -mm_triple_tail_types qwen-MKG_Y-triple_tail-ent_v0-v1.json \
                -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_2-ent_v0-batch_256-epoch_200 \
                # -use_relation_hint \
                # -mm_relation_hints qwen-MKG-Y+_img_context-setting_3+-combine.json \
                # -use_reverse_alias \
                # -use_relation_triples \

######## mm setting 3 ########
# # train commandline:
# python main.py -dataset MKG_Y \
#                 -batch_size 256 \
#                 -pretrained_model bert-large-uncased \
#                 -epoch 200 \
#                 -desc_max_length 40 \
#                 -lr 5e-4 \
#                 -prompt_length 10 \
#                 -alpha 0.1 \
#                 -n_lar 8 \
#                 -label_smoothing 0.1 \
#                 -embed_dim 144 \
#                 -k_w 12 \
#                 -k_h 12 \
#                 -alpha_step 0.00001 \
#                 -use_mm \
#                 -desc_path entityid2mm_description-setting_3-concate-0.txt \
#                 -mm_rel_triples qwen-MKG_Y-rel_context-ent_v0-v2.json \
#                 -mm_triple_tail_types qwen-MKG_Y-triple_tail-ent_v0-v1.json \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_3-ent_v0-batch_256-epoch_200
#                 # -use_reverse_alias \
#                 # -use_relation_triples \
#                 # -use_relation_hint \


# # evaluation commandline:
# python main.py -dataset MKG_Y \
#                 -batch_size 128 \
#                 -pretrained_model bert-large-uncased \
#                 -desc_max_length 40 \
#                 -lr 5e-4 \
#                 -prompt_length 10 \
#                 -alpha 0.1 \
#                 -n_lar 8 \
#                 -label_smoothing 0.1 \
#                 -embed_dim 144 \
#                 -k_w 12 \
#                 -k_h 12 \
#                 -alpha_step 0.00001 \
#                 -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-2025-02-04-17-44-03-epoch_60/MKG_Y-epoch=056-val_mrr=0.2558.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-2025-02-04-10-49-01-epoch_200/MKG_Y-epoch=197-val_mrr=0.3096.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_Y-2025-02-03-18-24-57/MKG_Y-epoch=098-val_mrr=0.2705.ckpt


T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
echo $(date -ud "@$ELAPSED" +'%j days %H hours %M minutes %S seconds' | awk '{print ($1-1) " days " $3 " hours " $5 " minutes " $7 " seconds"}')