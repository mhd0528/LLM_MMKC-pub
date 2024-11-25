import argparse
import collections
import json as js
import os
from pathlib import Path
import random

import torch
from model.instructblip import (
    InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel, InstructBlipForConditionalGeneration, InstructBlipProcessor
)

from context_utils import related_ent_img_search, get_useful_images, get_image_relation_prompt, get_summary_context, get_ent_images, ent_combine_summary

seed = 0
torch.manual_seed(seed)
random.seed(seed)

# load model and processor
model_type = "instructblip"
model_ckpt = "BleachNick/MMICL-Instructblip-T5-xxl"
processor_ckpt = "Salesforce/instructblip-flan-t5-xxl"

config = InstructBlipConfig.from_pretrained(model_ckpt)


model = InstructBlipForConditionalGeneration.from_pretrained(
    model_ckpt,
    config=config
)

# define special tokens
image_placeholder = "图"
sp = [image_placeholder] + [f"<image{i}>" for i in range(20)]

processor = InstructBlipProcessor.from_pretrained(processor_ckpt)

sp = sp + processor.tokenizer.additional_special_tokens[len(sp):]
processor.tokenizer.add_special_tokens({'additional_special_tokens': sp})
if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
    model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
replace_token = "".join(32 * [image_placeholder])

sp_token_id = processor.tokenizer.convert_tokens_to_ids(image_placeholder)
processor.tokenizer.img_place_token_id = sp_token_id
print(f"Special tokens id for '{image_placeholder}': {sp_token_id}. Add to processor.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device, dtype=torch.bfloat16)


def kg_context_generation(kg_img_context_folder, data_folder, triple_split, ent2str_file, id2idx_file, log_file, resume=0, stop=-1, model_mod="multi", context_mod=1):
    image_folder = os.path.join(kg_img_context_folder, "image-graph_images")

    # load training triples
    if triple_split == "all":
        triple_splits = ["train", "valid", "test"]
    else:
        triple_splits = triple_split.split(',')
    context_target_triples = []
    for split in triple_splits:
        split_triples_file = f"{split}.del"
        with open(os.path.join(data_folder, split_triples_file)) as triple_f:
            for line in triple_f.readlines():
                subj, rel, obj = line.strip().split("\t")
                context_target_triples.append([subj, rel, obj])
    if stop == -1:
        stop = len(context_target_triples)

    # ent id 2 text
    # load idx2txt into dictionary
    ent2str_dict = {}
    with open(os.path.join(data_folder, ent2str_file)) as e2s_f:
        for line in e2s_f.readlines():
            e_idx, e_str = line.strip().split("\t")
            ent2str_dict[e_idx] = e_str
    # load id2idx into dictionary
    id2idx_dict = {}
    with open(os.path.join(data_folder, id2idx_file)) as e2i_f:
        for line in e2i_f.readlines():
            e_idx, e_id = line.strip().split("\t")
            id2idx_dict[e_idx] = e_id

    # for each triple, we generate the caption based on the entity
    # the idea is to select images that are about the triple
    triple_cnt = 0

    # save generations to a dicitonary
    ent_context_dict = dict()
    # load from file if previously exist
    context_folder = os.path.join(kg_img_context_folder, "context")
    Path(context_folder).mkdir(parents=True, exist_ok=True)
    if os.path.exists(os.path.join(context_folder, log_file)):
        print(f"Loading existing context file: {os.path.join(context_folder, log_file)}")
        with open(os.path.join(context_folder, log_file)) as context_f:
            ent_context_dict = js.load(context_f)

    total_cnt = len(context_target_triples)
    for idx, (subj, rel, obj) in enumerate(context_target_triples):
        if idx < resume:
            continue
        if idx > stop:
            break
        subj_s, obj_s = ent2str_dict[subj], ent2str_dict[obj]
        subj_id, obj_id = id2idx_dict[subj], id2idx_dict[obj]
        subj_path = os.path.join(image_folder, subj_id[1:].replace('/', '.') if subj_id[0] == '/' else subj_id)
        obj_path = os.path.join(image_folder, obj_id[1:].replace('/', '.') if obj_id[0] == '/' else obj_id)
        print("========")
        # skip if we already processed this triple
        if context_mod == 3:
            if (subj in ent_context_dict) and (obj in ent_context_dict):
                print(f"Triple {triple_cnt} (total {idx}/{total_cnt}) between {subj}:{subj_s} and {obj}:{obj_s} already processed... Skip...")
                print("========\n")
                triple_cnt += 1
                continue
            else:
                # initialize context dictionary for subj and obj
                ent_context_dict[subj] = dict()
                ent_context_dict[subj]["ent_str"] = subj_s
                ent_context_dict[subj]["mm_description"] = []
                ent_context_dict[obj] = dict()
                ent_context_dict[obj]["ent_str"] = obj_s
                ent_context_dict[obj]["mm_description"] = []
        else:
            if (subj in ent_context_dict) and (obj in ent_context_dict[subj]["context_dict"]):
                print(f"Triple {triple_cnt} (total {idx}/{total_cnt}) between {subj}:{subj_s} and {obj}:{obj_s} already processed... Skip...")
                print("========\n")
                triple_cnt += 1
                continue
            else:
                # initialize context dictionary for subj and obj
                if subj not in ent_context_dict:
                    ent_context_dict[subj] = dict()
                    ent_context_dict[subj]["ent_str"] = subj_s
                    ent_context_dict[subj]["context_dict"] = collections.defaultdict(list)
                if obj not in ent_context_dict[subj]["context_dict"]:
                    ent_context_dict[subj]["context_dict"][obj] = []
                if obj not in ent_context_dict:
                    ent_context_dict[obj] = dict()
                    ent_context_dict[obj]["ent_str"] = obj_s
                    ent_context_dict[obj]["context_dict"] = collections.defaultdict(list)
                if subj not in ent_context_dict[obj]["context_dict"]:
                    ent_context_dict[obj]["context_dict"][subj] = []

        print(f"Generation about {subj}: {subj_s} and {obj}: {obj_s}, relation: {rel}")
        if context_mod == 1:  # setting 1: for each entity: select related images --> generate context for each image
            subj_prompt = (
                f"Is the entity '{subj_s}' present in or related to this image: <image>{replace_token}? "
                "Respond with 'Yes' or 'No' only."
            )
            obj_prompt = (
                f"Is the entity '{obj_s}' present in or related to this image: <image>{replace_token}? "
                "Respond with 'Yes' or 'No' only."
            )
            context_prompt = (
                f"Analyze this image: <image>{replace_token} and check if it contains representations of both '{subj_s}' and '{obj_s}' from the knowledge graph. "
                f"Explain how '{subj_s}' and '{obj_s}' are depicted in the image or confirm if they are related to the content of the image."
                "Respond in one sentence."
            )
            prompt_list = [subj_prompt, obj_prompt, context_prompt]
        elif context_mod == 2:  # setting 2: for each entity: select related images --> generate context for all images (summary of images)
            subj_prompt = (
                f"Is the entity '{subj_s}' present in or related to the image 0: <image0>{replace_token}? "
                "Respond with 'Yes' or 'No' only."
            )
            obj_prompt = (
                f"Is the entity '{obj_s}' present in or related to the image 0: <image0>{replace_token}? "
                "Respond with 'Yes' or 'No' only."
            )
            context_prompt = ""  # no context prompt for summary
            prompt_list = [subj_prompt, obj_prompt, context_prompt]

        # loop through images, when image path exists
        if context_mod == 1:  # setting 1: for each entity: select related images --> generate context for each image
            subj_context_list, obj_context_list = [], []
            if os.path.exists(subj_path):
                print(f"\t======== Processing subject {subj} images...")
                subj_context_list = related_ent_img_search(subj_path, model, processor, prompt_list, device, model_mod)
            else:
                print(f"\t======== subject {subj} images don't exist (target: {subj_path})")
            if os.path.exists(obj_path):
                print(f"\t======== Processing object {obj} images...")
                obj_context_list = related_ent_img_search(obj_path, model, processor, prompt_list, device, model_mod)
            else:
                print(f"\t======== object {obj} images don't exist (target: {obj_path})")

            triple_context_list = subj_context_list + obj_context_list
            triple_context_cnt = len(triple_context_list)
            print(f">>>> Related images found: {triple_context_cnt}")

            # save generations to dictionary
            if triple_context_cnt > 0:
                ent_context_dict[subj]["context_dict"][obj] += triple_context_list
                ent_context_dict[obj]["context_dict"][subj] += triple_context_list
        elif context_mod == 2:  # setting 2: for each entity: select related images --> generate context for all images (summary of images)
            # subj
            if os.path.exists(subj_path):  # and subj == '/m/01sl1q':
                print(f"\t======== Processing subject {subj} images...")
                subj_images_list = get_useful_images(subj_path, model, processor, prompt_list, device=device, model_mod=model_mod, select_best_one=False)
            else:
                subj_images_list = []
                print(f"\t======== subject {subj} images don't exist")

            # obj
            if os.path.exists(obj_path):
                print(f"\t======== Processing object {obj} images...")
                obj_images_list = get_useful_images(obj_path, model, processor, prompt_list, device=device, model_mod=model_mod, select_best_one=False)
            else:
                obj_images_list = []
                print(f"\t======== object {obj} images don't exist")

            all_useful_images = subj_images_list + obj_images_list

            # Ignore this triple, if no useful images are found
            if len(all_useful_images) == 0:
                ent_context_dict[subj]["context_dict"][obj] += []
                ent_context_dict[obj]["context_dict"][subj] += []
                # save to json file
                with open(os.path.join(context_folder, f"{log_file}-backup"), 'w') as out_f:
                    js.dump(ent_context_dict, out_f, indent=2)
                with open(os.path.join(context_folder, log_file), 'w') as out_f:
                    js.dump(ent_context_dict, out_f, indent=2)
                print(f"No useful images found. Triple {triple_cnt} (total {idx}/{total_cnt}) skipped...")
                print("========\n")
                triple_cnt += 1
                continue

            # all_useful_images
            summary_prompt = get_image_relation_prompt(all_useful_images, subj_s, obj_s, replace_token)
            # print("prompt: ")
            # print(summary_prompt)

            # all_useful_images
            triple_context_list = []
            triple_context_list += get_summary_context(model=model, processor=processor, useful_images=all_useful_images, summary_prompt=summary_prompt)

            print(f">>>> Related images found: {len(all_useful_images)}")

            # save generations to dictionary
            ent_context_dict[subj]["context_dict"][obj] += triple_context_list
            ent_context_dict[obj]["context_dict"][subj] += triple_context_list
        elif context_mod == 3:  # setting 3: for each head entity: all images --> generate context (summary of images)
            subj_valid_images = get_ent_images(subj_path)
            if subj_valid_images is None:
                ent_context_dict[subj]["mm_description"] = ''.join(ent_context_dict[subj]["mm_description"])
                print(f"\t======== subject {subj} images don't exist")
            elif len(subj_valid_images) == 0:
                ent_context_dict[subj]["mm_description"] = ''.join(ent_context_dict[subj]["mm_description"])
                print(f"\t======== subject {subj} images are not valid")
            else:
                # generate combined context (commonsense + image focus)
                ent_context_dict[subj]["mm_description"] = ent_combine_summary(subj_valid_images, subj_s, replace_token, model, processor, device, model_mod=model_mod)

            obj_valid_images = get_ent_images(obj_path)
            if obj_valid_images is None:
                ent_context_dict[obj]["mm_description"] = ''.join(ent_context_dict[obj]["mm_description"])
                print(f"\t======== object {obj} images don't exist")
            elif len(obj_valid_images) == 0:
                ent_context_dict[obj]["mm_description"] = ''.join(ent_context_dict[obj]["mm_description"])
                print(f"\t======== object {obj} images are not valid")
            else:
                ent_context_dict[obj]["mm_description"] = ent_combine_summary(obj_valid_images, obj_s, replace_token, model, processor, device, model_mod=model_mod)
            print(f'\t>>>> Generated context: {ent_context_dict[subj]["mm_description"]}|SEP|{ent_context_dict[obj]["mm_description"]}')

        # if triple_cnt > 3:
        #     break

        # save to json file
        with open(os.path.join(context_folder, f"{log_file}-backup"), 'w') as out_f:
            js.dump(ent_context_dict, out_f, indent=2)
        with open(os.path.join(context_folder, log_file), 'w') as out_f:
            js.dump(ent_context_dict, out_f, indent=2)

        print(f"Triple {triple_cnt} (total {idx}/{total_cnt}) processed...")
        print("========\n")
        triple_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context for KG triples')
    parser.add_argument('-kg', '--kg_img_context_folder', help='Path to KG image context folder',
                        default="/orange/daisyw/ma.haodi/LLM-MMKGC/data/FB15k-237/")
    parser.add_argument('-d', '--data_folder', help='Path to data folder of the KG',
                        default="/blue/daisyw/ma.haodi/LLM-MMKGC/data/fb15k-237/")
    parser.add_argument('-ts', '--triples_split', help='data splits we want to generate context for',
                        default="train")
    parser.add_argument('-e2s', '--ent2str_file', help='Entity id to string file',
                        default="entity_strings.del")
    parser.add_argument('-i2i', '--id2idx_file', help='Entity id to idx file',
                        default="entity_ids.del")
    parser.add_argument('-l', '--log_file', help='Log file to save generations',
                        default="t5-FB237_train_context-v1.json")
    parser.add_argument('-r', '--resume', help='Resume from index',
                        default=0)
    parser.add_argument('-s', '--stop', help='Stop at index',
                        default=-1)
    parser.add_argument('-mm', '--model_mod', help='Model mode we are using, either multi-images or single-image',
                        default="multi")
    parser.add_argument('-cm', '--context_mod', help='context search mode, currently 1: single image generation, 2: multi images summary',
                        default="1")
    args = vars(parser.parse_args())
    torch.set_float32_matmul_precision('medium')
    kg_context_generation(
        args["kg_img_context_folder"], args["data_folder"],
        args["triples_split"], args["ent2str_file"], args["id2idx_file"],
        args["log_file"], int(args["resume"]), int(args["stop"]), args["model_mod"], int(args["context_mod"])
    )
