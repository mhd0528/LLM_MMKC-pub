import os
from transformers import (
    InstructBlipProcessor, InstructBlipForConditionalGeneration
)
import torch
from PIL import Image

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xxl"
    )
processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-flan-t5-xxl"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# prompt = "What is unusual about this image?"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

# load id2str dictionary
id2str_file = "/blue/daisyw/ma.haodi/LLM-MMKGC/data/fb15k-237/entity_strings.del"
id2str_dict = {}
with open(id2str_file) as f:
    for line in f.readlines():
        e_id, e_str = line.strip().split('\t')
        id2str_dict[e_id] = e_str

image_folder = "/orange/daisyw/ma.haodi/LLM-MMKGC/data/FB15k-237/image-graph_images"
# for each entity, we generate the caption based on the entity
for name in os.listdir(image_folder):
    ent_path = os.path.join(image_folder, name)
    # if folder， extract id, then check all images inside
    if os.path.isdir(ent_path):
        ent_id = '/' + name.replace('.', '/')
        ent_str = id2str_dict[ent_id]
        print(f">>> Generating captions for ENT {ent_str}, {ent_id}")
        for image_path in os.listdir(ent_path):
            image_full_path = os.path.join(ent_path, image_path)
            with open(image_full_path) as f:
                # print(f"Content of '{image_full_path}'")
                image = Image.open(image_full_path).convert("RGB")
                # prompt = f"This image is about {ent_str}. Summarize what does it show about this entity in one sentence?"
                prompt = f"Describe the image about {ent_str} in one sentence."
                inputs = processor(
                    images=image, text=prompt, return_tensors="pt"
                    ).to(device)

                outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=256,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                )
                generated_text = processor.batch_decode(
                    outputs, skip_special_tokens=True
                    )[0].strip()
                print(f"\t>>>> {image_path}: {generated_text}")
        break
