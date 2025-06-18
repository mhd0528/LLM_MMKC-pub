######## baseline setting ########
# # train commandline:
# python main.py -dataset MKG_W \
#                 -batch_size 128 \
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
#                 -desc_path entityid2description.txt \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-$(date +'%Y-%m-%d-%H-%M-%S')-baseline-epoch_200


# ######## mm setting 2 ########
# train commandline:
python main.py -dataset MKG_W \
                -batch_size 128 \
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
                -mm_rel_triples qwen-MKG_W-rel_context-ent_v6-v2.json \
                -mm_triple_tail_types qwen-MKG_W-triple_tail-ent_v6-v1.json \
                -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_2+rel_hint-ent_v6-epoch_200 \
                -use_relation_hint \
                -mm_relation_hints qwen-MKG-W+_img_context-setting_3+-combine.json \
                # -use_reverse_alias \
                # -use_relation_triples \

######## mm setting 3 ########
# # train commandline:
# python main.py -dataset MKG_W \
#                 -batch_size 128 \
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
#                 -mm_rel_triples qwen-MKG_W-rel_context-ent_v6-v2.json \
#                 -mm_triple_tail_types qwen-MKG_W-triple_tail-ent_v6-v1.json \
#                 -save_dir /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-$(date +'%Y-%m-%d-%H-%M-%S')-better_setting_3+x-ent_v6-epoch_200 \
#                 # -use_reverse_alias \
#                 # -use_relation_triples \
#                 # -use_relation_hint \


# # evaluation commandline:
# python main.py -dataset MKG_W \
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
#                 -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-2025-02-04-17-44-03-epoch_60/MKG_W-epoch=056-val_mrr=0.2558.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-2025-02-04-10-49-01-epoch_200/MKG_W-epoch=197-val_mrr=0.3096.ckpt
#                 # -model_path /blue/daisyw/ma.haodi/MMKGC/CSProm-KG/checkpoint/MKG_W-2025-02-03-18-24-57/MKG_W-epoch=098-val_mrr=0.2705.ckpt