import argparse
import collections
import json as js
import os
from pathlib import Path
import random
import re

def kg_context_generation(recover_file, kg_img_context_folder, log_file, resume=0, stop=-1):
    ent_context_dict = collections.defaultdict(list)

    # patterns to match
    # Pattern to extract the triples and contexts
    triple_pattern = r"Generation about (\d+): (.+?) and (\d+): (.+?), relation: (\d+)"
    context_pattern = r">>>> Generated context: (.+)"
    related_images_pattern = r">>>> Related images found: (\d+)"

    with open(recover_file, 'r') as in_f:
        log_data = in_f.read()

    triple_cnt = 0
    for match in re.finditer(triple_pattern, log_data):
        obj_id, obj_str, subj_id, subj_str, rel_id = match.groups()
        triple_key = (obj_id, obj_str, subj_id, subj_str, rel_id)
        print(f"========{match}========")

        # initialize the context dictionary
        if obj_id not in ent_context_dict:
            ent_context_dict[obj_id] = dict()
            ent_context_dict[obj_id]["ent_str"] = obj_str
            ent_context_dict[obj_id]["context_dict"] = collections.defaultdict(list)
        if subj_id not in ent_context_dict[obj_id]["context_dict"]:
            ent_context_dict[obj_id]["context_dict"][subj_id] = []
        if subj_id not in ent_context_dict:
            ent_context_dict[subj_id] = dict()
            ent_context_dict[subj_id]["ent_str"] = subj_str
            ent_context_dict[subj_id]["context_dict"] = collections.defaultdict(list)
        if obj_id not in ent_context_dict[subj_id]["context_dict"]:
            ent_context_dict[subj_id]["context_dict"][obj_id] = []

        # Extract the block related to this triple
        start = match.end()
        end = log_data.find("...\n========", start)
        if end == -1:
            end = len(log_data)
        block = log_data[start:end]
        print(f"Retrieved block: \n{block}")

        # Check if there are related images found
        related_images_match = re.search(related_images_pattern, block)
        if related_images_match:
            related_images_count = int(related_images_match.group(1))
            # Skip if no related images found
            if related_images_count == 0:
                print(f"========Triple {triple_cnt} processed...========")
                triple_cnt += 1
                continue

        # Extract all generated contexts in the block
        contexts = re.findall(context_pattern, block)
        # print(contexts)

        # Add object entity and context
        ent_context_dict[obj_id]["context_dict"][subj_id].extend(contexts)

        # Add subject entity and context
        ent_context_dict[subj_id]["context_dict"][obj_id].extend(contexts)

        # load from file if previously exist
        context_folder = os.path.join(kg_img_context_folder, "context")
        Path(context_folder).mkdir(parents=True, exist_ok=True)
        # save to json file
        with open(os.path.join(context_folder, log_file), 'w') as out_f:
            js.dump(ent_context_dict, out_f, indent=2)

        print(f"========Triple {triple_cnt} processed...========\n")
        triple_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context for KG triples')
    parser.add_argument('-re', '--recover_file', help='The file used to recover the context dictionary',
                        default="/orange/daisyw/ma.haodi/LLM-MMKGC/data/FB15k-237/")
    parser.add_argument('-kg', '--kg_img_context_folder', help='Path to KG image context folder',
                        default="/orange/daisyw/ma.haodi/LLM-MMKGC/data/FB15k-237/")
    parser.add_argument('-l', '--log_file', help='Log file to save generations',
                        default="t5-FB237_train_context-v1.json")
    parser.add_argument('-r', '--resume', help='Resume from index',
                        default=0)
    parser.add_argument('-s', '--stop', help='Stop at index',
                        default=-1)
    args = vars(parser.parse_args())

    kg_context_generation(
        args["recover_file"],
        args["kg_img_context_folder"],
        args["log_file"], int(args["resume"]), int(args["stop"])
    )
