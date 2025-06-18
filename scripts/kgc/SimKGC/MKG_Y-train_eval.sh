
######## baseline setting ########
# # train commandline:
# python3 -u main.py \
# --model-dir "${OUTPUT_DIR}" \
# --pretrained-model bert-base-uncased \
# --pooling mean \
# --lr 1e-5 \
# --use-link-graph \
# --train-path "$DATA_DIR/train.txt.json" \
# --valid-path "$DATA_DIR/valid.txt.json" \
# --task ${TASK} \
# --batch-size 1024 \
# --print-freq 20 \
# --additive-margin 0.02 \
# --use-amp \
# --use-self-negative \
# --finetune-t \
# --pre-batch 2 \
# --epochs 10 \
# --workers 4 \
# --max-to-keep 5 "$@"
# # --use_desc


######## mm setting 2 ########
# # train commandline:
# python3 -u main.py \
# --model-dir "${OUTPUT_DIR}" \
# --pretrained-model bert-base-uncased \
# --pooling mean \
# --lr 1e-5 \
# --use-link-graph \
# --train-path "$DATA_DIR/train.txt.json" \
# --valid-path "$DATA_DIR/valid.txt.json" \
# --task ${TASK} \
# --batch-size 1024 \
# --print-freq 20 \
# --additive-margin 0.02 \
# --use-amp \
# --use-self-negative \
# --finetune-t \
# --pre-batch 2 \
# --epochs 10 \
# --workers 4 \
# --max-to-keep 3 \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json" \
# --use_relation_hint \
# --mm_relation_hints "qwen-MKG-Y+_img_context-setting_3+-combine.json" \
# "$@"
# # --use_reverse_alias \
# # -use_relation_triples \

######## mm setting 3 ########
# # train commandline:
# python3 -u main.py \
# --model-dir "${OUTPUT_DIR}" \
# --pretrained-model bert-base-uncased \
# --pooling mean \
# --lr 1e-5 \
# --use-link-graph \
# --train-path "$DATA_DIR/train.txt.json" \
# --valid-path "$DATA_DIR/valid.txt.json" \
# --task ${TASK} \
# --batch-size 1024 \
# --print-freq 20 \
# --additive-margin 0.02 \
# --use-amp \
# --use-self-negative \
# --finetune-t \
# --pre-batch 2 \
# --epochs 10 \
# --workers 4 \
# --max-to-keep 3 \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json" \
# "$@"
# # --use_reverse_alias \
# # -use_relation_triples \
# # -use_relation_hint \


######## Eval commandline ########
# baseline setting
# bash scripts/eval.sh ./checkpoint/mkg_y-baseline/model_best.mdl MKG_Y
# bash scripts/eval.sh ./checkpoint/mkg_y-baseline_desc/model_best.mdl MKG_Y

# mm setting 2 (QWEN_new)
# bash scripts/eval.sh ./checkpoint/mkg_y-better_setting_2-ent_v0/model_best.mdl MKG_Y \
# mm setting 2+x (QWEN_new + gt description)
# bash scripts/eval.sh ./checkpoint/mkg_y-better_setting_2+x-ent_v0/model_best.mdl MKG_Y \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json"
# # --use_reverse_alias
# mm setting 2+rel_hint (QWEN_new + 3+)
bash scripts/eval.sh ./checkpoint/mkg_y-better_setting_2+rel_hint-ent_v0/model_best.mdl MKG_Y \
--use_desc \
--use_mm \
--mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
--mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json" \
--use_relation_hint \
--mm_relation_hints "qwen-MKG-Y+_img_context-setting_3+-combine.json" \
# # --use_reverse_alias

# mm setting 3
# bash scripts/eval.sh ./checkpoint/mkg_y-better_setting_3-ent_v0/model_best.mdl MKG_Y \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json"
# mm setting 3+x (QWEN_new + gt description)
# bash scripts/eval.sh ./checkpoint/mkg_y-better_setting_3+x-ent_v0/model_best.mdl MKG_Y \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG_Y-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-MKG_Y-triple_tail-ent_v0-v1.json"
# # --use_reverse_alias
