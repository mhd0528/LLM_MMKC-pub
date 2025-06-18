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
# --max-num-tokens 100 \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
# --use_relation_hint \
# --mm_relation_hints "qwen-FB237_img_context-small-setting_3+-v1-combine.json" \
# "$@"
# # --use_reverse_alias \
# # --use_relation_triples \

######## mm setting 3 ########
# entity context is preprocessed into entity.json file
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
# --mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
# "$@"
# # --use_reverse_alias \
# # --use_relation_triples \
# # --use_relation_hint \

######## Eval commandline ########
# baseline setting
# bash scripts/eval.sh ./checkpoint/fb15k237-baseline/model_best.mdl FB15k237
# bash scripts/eval.sh ./checkpoint/fb15k237-baseline_desc/model_best.mdl FB15k237

# mm setting 2 (QWEN_new)
bash scripts/eval.sh ./checkpoint/fb15k237-better_setting_2+rel_hint-2/model_best.mdl FB15k237 \
--use_desc \
--use_mm \
--mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
--mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
--use_relation_hint \
--mm_relation_hints "qwen-FB237_img_context-small-setting_3+-v1-combine.json" \
# --use_reverse_alias \
# mm setting 2 + x (mm desc + gt desc)
# bash scripts/eval.sh ./checkpoint/fb15k237-better_setting_2+x/model_best.mdl FB15k237 \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
# # --use_reverse_alias \

# # mm setting 3
# bash scripts/eval.sh ./checkpoint/fb15k237-better_setting_3/model_best.mdl FB15k237 \
# --use_desc \
# --use_mm \
# --mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
# --mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
# # --use_reverse_alias \
