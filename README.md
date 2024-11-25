# MMKG-T5

This is the repository for the ICDE submission "Transformer-Based Multimodal Knowledge Graph Completion with Link-Aware Contexts"

## Link-Aware Contexts

Leverage [multimodal generation model](https://huggingface.co/BleachNick/MMICL-Instructblip-T5-xxl) to provide link-aware context. 


## Getting Started

```
pip install -r requirements.txt
```

### Download Data
Train, valid, and test data are provided. 
Image data can be downloaded from: 


## Reproduction
### Training

To train the model on FB15-237, run the following command.
Note, this library will automatically use all available GPUs.
You can control the GPUs used with the environment variable `CUDA_VISIBLE_DEVICES=0,1,2,3`

```
python -u main.py --config-name=mm_config dataset.name=fb15k-237 dataset.mm=True \
      context.setting=$mm_setting context.max_size=50 context.file=data/fb15-237/context/t5-FB237_img_context-small-setting_2-v1.json \
      context.relation=true context.relation_file=data/fb15-237/context/t5-fb15k-237-rel_context-v1.json context.rel_max_size=10 \
      valid.tiny=True valid.every=2 train.max_epochs=30
```




