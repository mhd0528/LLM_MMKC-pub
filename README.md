# FICHA

This is the repository for the ICDE submission "FICHA: Fusion of Image Context and Human Annotations for Multi-Modal Knowledge Graph Completion"

## Link-Aware Contexts

Leverage [Qwen2-VL-8B](https://github.com/QwenLM/Qwen2.5-VL) to provide link-aware context.
Multimodal contexts for each dataset are provided in the `data/` folder
The converted context for CSProm-KG and SimKGC can be found in `CSProm-KG/data/` and `SimKGC/data/`.

## Getting Started

```
pip install -r requirements.txt
```

### Download Data
Train, valid, and test data are provided. 
Image data can be found at (not necessary): 
 - FB15k-237: Follow [MKGformer](https://github.com/zjunlp/MKGformer)
 - MKG-W/MKG-Y: Provide by [MMRNS](https://github.com/quqxui/MMRNS), [Google Drive](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj?usp=drive_link)


## Reproduction
### Setup
Get [CSProm-KG](https://github.com/chenchens190009/CSProm-KG) and [SimKGC](https://github.com/intfloat/SimKGC). Replace corresponding files with the ones in `CSProm-KG/` and `SimKGC/` to enable MM context.
Install the requirements for `CSProm-KG`, `SimKGC`, and `Qwen` from their repo.

### Training

To train the model on for CSProm-KG, use the following command, with correponding context file for each setting:

```
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
      -save_dir {PATH_TO_SAVE_CHECKPOINTS} \
      -use_relation_hint \
      -mm_relation_hints qwen-FB237_img_context-small-setting_3+-v1-combine.json
```

For SimKGC, first set:
`OUTPUT_DIR={PATH_TO_SAVE_RESULT}, DATA_DIR={PATH_TO_DATA}, TASK={FB15k237, MKG-W, MKG-Y}`
Then using the following command:
```
python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 2 \
--epochs 10 \
--workers 4 \
--max-to-keep 3 \
--use_desc \
--use_mm \
--mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
--mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
"$@"
```

### Testing

To evaluate the model for CSProm-KG, run the following command, with correponding context file for each setting, and corresponding checkpoints, e.g.:
```
python main.py -dataset FB15k237 \
      -batch_size 128 \
      -pretrained_model bert-large-uncased \
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
      -model_path {PATH_TO_CHECKPOINT}
```

For SimKGC, using the following command
```
bash SimKGC/scripts/eval.sh {PATH_TO_CHECK_POINT} FB15k237 \
--use_desc \
--use_mm \
--mm_rel_triples "qwen-fb15k-237-rel_context-ent_v0-v2.json" \
--mm_triple_tail_types "qwen-fb15k-237-triple_tail-ent_v0-v1.json" \
--use_relation_hint \
--mm_relation_hints "qwen-FB237_img_context-small-setting_3+-v1-combine.json" \
```