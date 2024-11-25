import argparse
import collections
import json as js
import os
from pathlib import Path
import random

import torch
from transformers import (
    InstructBlipProcessor, InstructBlipForConditionalGeneration
)

from context_utils import related_ent_img_search

# load model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xxl"
)
processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-flan-t5-xxl"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

seed = 0
torch.manual_seed(seed)
random.seed(seed)


def kg_context_generation(kg_img_context_folder, data_folder, train_triples_file, ent2str_file, id2idx_file, log_file, resume=0, stop=-1):
    image_folder = os.path.join(kg_img_context_folder, "image-graph_images")

    # load training triples
    train_triples = []
    train_ent_dict = collections.defaultdict(list)
    with open(os.path.join(data_folder, train_triples_file)) as triple_f:
        for line in triple_f.readlines():
            obj, rel, subj = line.strip().split("\t")
            train_triples.append([obj, rel, subj])
            train_ent_dict[obj].append([rel, subj])
            train_ent_dict[subj].append([rel, obj])
    if stop == -1:
        stop = len(train_triples)

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
        print(f"Loading existing context file: {log_file}")
        with open(os.path.join(context_folder, log_file)) as context_f:
            ent_context_dict = js.load(context_f)

    for idx, (obj, rel, subj) in enumerate(train_triples):
        if idx < resume:
            continue
        if idx > stop:
            break
        obj_s, subj_s = ent2str_dict[obj], ent2str_dict[subj]
        obj_id, subj_id = id2idx_dict[obj], id2idx_dict[subj]
        obj_path = os.path.join(image_folder, obj_id[1:].replace('/', '.'))
        subj_path = os.path.join(image_folder, subj_id[1:].replace('/', '.'))
        print("========")
        # skip if we already processed this triple
        if (obj in ent_context_dict) and (subj in ent_context_dict[obj]["context_dict"]):
            print(f"Triple {triple_cnt} (total {idx}) between {obj}:{obj_s} and {subj}:{subj_s} already processed... Skip...")
            print("========\n")
            triple_cnt += 1
            continue

        # initialize context dictionary for obj and subj
        if obj not in ent_context_dict:
            ent_context_dict[obj] = dict()
            ent_context_dict[obj]["ent_str"] = obj_s
            ent_context_dict[obj]["context_dict"] = collections.defaultdict(list)
        if subj not in ent_context_dict[obj]["context_dict"]:
            ent_context_dict[obj]["context_dict"][subj] = []
        if subj not in ent_context_dict:
            ent_context_dict[subj] = dict()
            ent_context_dict[subj]["ent_str"] = subj_s
            ent_context_dict[subj]["context_dict"] = collections.defaultdict(list)
        if obj not in ent_context_dict[subj]["context_dict"]:
            ent_context_dict[subj]["context_dict"][obj] = []
        
        print(f"Generation about {obj}: {obj_s} and {subj}: {subj_s}, relation: {rel}")
        obj_prompt = (
            f"Is the entity '{obj_s}' present in or related to this image? "
            "Respond with 'Yes' or 'No' only."
        )
        subj_prompt = (
            f"Is the entity '{subj_s}' present in or related to this image? "
            "Respond with 'Yes' or 'No' only."
        )
        context_prompt = (
            f"Analyze this image and check if it contains representations of both '{obj_s}' and '{subj_s}' from the knowledge graph. "
            f"Explain how '{obj_s}' and '{subj_s}' are depicted in the image or confirm if they are related to the content of the image."
            "Respond in one sentence."
        )
        prompt_list = [obj_prompt, subj_prompt, context_prompt]
        
        # loop through images, when image path exists
        obj_context_list, subj_context_list = [], []
        if os.path.exists(obj_path):
            print(f"\t======== Processing object {obj} images...")
            obj_context_list = related_ent_img_search(obj_path, model, processor, prompt_list, device)
        else:
            print(f"\t======== Object {obj} images don't exist")
        if os.path.exists(subj_path):
            print(f"\t======== Processing subject {subj} images...")
            subj_context_list = related_ent_img_search(subj_path, model, processor, prompt_list, device)
        else:
            print(f"\t======== Subject {subj} images don't exist")

        triple_context_list = obj_context_list + subj_context_list
        triple_context_cnt = len(triple_context_list)
        print(f">>>> Related images found: {triple_context_cnt}")

        # save generations to dictionary
        if triple_context_cnt > 0:
            ent_context_dict[obj]["context_dict"][subj] += triple_context_list
            ent_context_dict[subj]["context_dict"][obj] += triple_context_list

        # if triple_cnt > 3:
        #     break

        # save to json file
        with open(os.path.join(context_folder, f"{log_file}-backup"), 'w') as out_f:
            js.dump(ent_context_dict, out_f, indent=2)
        with open(os.path.join(context_folder, log_file), 'w') as out_f:
            js.dump(ent_context_dict, out_f, indent=2)

        print(f"Triple {triple_cnt} (total {idx}) processed...")
        print("========\n")
        triple_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context for KG triples')
    parser.add_argument('-kg', '--kg_img_context_folder', help='Path to KG image context folder',
                        default="/orange/daisyw/ma.haodi/LLM-MMKGC/data/FB15k-237/")
    parser.add_argument('-d', '--data_folder', help='Path to data folder of the KG',
                        default="/blue/daisyw/ma.haodi/LLM-MMKGC/data/fb15k-237/")
    parser.add_argument('-tr', '--train_triples_file', help='Training triples file',
                        default="train.txt")
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
    args = vars(parser.parse_args())
    torch.set_float32_matmul_precision('medium')
    kg_context_generation(
        args["kg_img_context_folder"], args["data_folder"],
        args["train_triples_file"], args["ent2str_file"], args["id2idx_file"],
        args["log_file"], int(args["resume"]), int(args["stop"])
    )
