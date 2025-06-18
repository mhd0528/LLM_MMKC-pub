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
# --max-to-keep 3 "$@"
# # --use_desc


######## mm setting 2 ########
# # entity context is preprocessed into entity.json file
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
# --mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
# --mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json" \
# --use_relation_hint \
# --mm_relation_hints "qwen-MKG-W+_img_context-setting_3+-combine.json" \
# "$@"
# # --use_reverse_alias \
# # --use_relation_triples \

# ######## mm setting 3 ########
# # entity context is preprocessed into entity.json file
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
# --mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
# --mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json" \
# "$@"
# # --use_reverse_alias \
# # --use_relation_triples \
# # --use_relation_hint \

######## Eval commandline ########
# # baseline setting
# bash scripts/eval.sh ./checkpoint/mkg_w-baseline/model_best.mdl MKG_W
# bash scripts/eval.sh ./checkpoint/mkg_w-baseline_desc/model_best.mdl MKG_W
# mm setting 2 (QWEN_old)
# bash scripts/eval.sh ./checkpoint/mkg_w-mm_setting_2-old/model_best.mdl MKG_W \
# mm setting 2 (new)
bash scripts/eval.sh ./checkpoint/mkg_w-better_setting_2-ent_v6/model_best.mdl MKG_W \
--use_desc \
--use_mm \
--mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
--mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json" \
--use_relation_hint \
--mm_relation_hints "qwen-MKG-W+_img_context-setting_3+-combine.json" \
# --use_reverse_alias \
# mm setting 2 + x (mm desc + gt desc)
# bash scripts/eval.sh ./checkpoint/mkg_w-better_setting_2+x-ent_v6/model_best.mdl MKG_W \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
# --mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json"
# --use_reverse_alias \

# mm setting 3
# bash scripts/eval.sh ./checkpoint/mkg_w-better_setting_3-ent_v6/model_best.mdl MKG_W \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
# --mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json"
# # --use_reverse_alias \
# mm setting 3 + x (mm desc + gt desc)
# bash scripts/eval.sh ./checkpoint/mkg_w-better_setting_3+x-ent_v6/model_best.mdl MKG_W \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-MKG-W-rel_context-ent_v6-v2.json" \
# --mm_triple_tail_types "qwen-MKG-W-triple_tail-ent_v6-v1.json"
# # --use_reverse_alias \
