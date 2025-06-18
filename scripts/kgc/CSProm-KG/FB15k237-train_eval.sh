######## baseline setting ########
# # train commandline:
# python main.py -dataset FB15k237 \
#                 -batch_size 128 \
#                 -pretrained_model bert-large-uncased \
#                 -epoch 60 \
#                 -desc_max_length 40 \
#                 -lr 5e-4 \
#                 -prompt_length 10 \
#                 -alpha 0.1 \
#                 -n_lar 8 \
#                 -label_smoothing 0.1 \
#                 -embed_dim 144 \
#                 -k_w 12 \
#                 -k_h 13 \
#                 -alpha_step 0.00001 \
#                 -desc_path entityid2description.txt \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k-237-ori-$(date +'%Y-%m-%d-%H-%M-%S')-baseline-epoch_60


######## mm setting 2 ########
# train commandline:
python main.py -dataset FB15k237 \
                -batch_size 128 \
                -pretrained_model bert-large-uncased \
                -epoch 60 \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 156 \
                -k_w 12 \
                -k_h 13 \
                -alpha_step 0.00001 \
                -use_mm \
                -desc_path entityid2mm_description-setting_2-concate-0.txt \
                -mm_rel_triples qwen-fb15k-237-rel_context-ent_v0-v2.json \
                -mm_triple_tail_types qwen-fb15k-237-triple_tail-ent_v0-v1.json \
                -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_2+rel_hint-emb_156-epoch_60 \
                -use_relation_hint \
                -mm_relation_hints qwen-FB237_img_context-small-setting_3+-v1-combine.json \
                # -continue_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-2025-03-12-00-59-30-better_setting_2+x-epoch_60/FB15k237-epoch=026-val_mrr=0.3220.ckpt
                # -use_reverse_alias \
                # -use_relation_triples \

######## mm setting 3 ########
# # train commandline:
# python main.py -dataset FB15k237 \
#                 -batch_size 128 \
#                 -pretrained_model bert-large-uncased \
#                 -epoch 60 \
#                 -desc_max_length 40 \
#                 -lr 5e-4 \
#                 -prompt_length 10 \
#                 -alpha 0.1 \
#                 -n_lar 8 \
#                 -label_smoothing 0.1 \
#                 -embed_dim 156 \
#                 -k_w 12 \
#                 -k_h 13 \
#                 -alpha_step 0.00001 \
#                 -use_mm \
#                 -desc_path entityid2mm_description-setting_3-concate-0.txt \
#                 -mm_rel_triples qwen-fb15k-237-rel_context-ent_v0-v2.json \
#                 -mm_triple_tail_types qwen-fb15k-237-triple_tail-ent_v0-v1.json \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_3-emb_156-epoch_60
#                 # -use_reverse_alias \
#                 # -use_relation_triples \
#                 # -use_relation_hint \
#                 # -mm_relation_hints "path" \


# # evaluation commandline:
# python main.py -dataset FB15k237 \
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
#                 -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-2025-02-04-17-44-03-epoch_60/FB15k237-epoch=056-val_mrr=0.2558.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-2025-02-04-10-49-01-epoch_200/FB15k237-epoch=197-val_mrr=0.3096.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/FB15k237-2025-02-03-18-24-57/FB15k237-epoch=098-val_mrr=0.2705.ckpt