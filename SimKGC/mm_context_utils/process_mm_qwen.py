import os
import re
from collections import defaultdict
import json
import random

def read_entity_strings(file_path):
    entity_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            entity_dict[parts[0]] = parts[1]
    return entity_dict

def main():
    '''
    From MMKGC T5 MM context to CSProm-KG data format (entity, relation desc).
    source:
    1. entity types
    2. relation template
    3. ent+rel --> triples
    '''
    source_dir = '/blue/daisyw/ma.haodi/MMKGC/LLM-MMKGC/data/MKG-W'
    mm_source_dir = '/blue/daisyw/ma.haodi/MMKGC/LLM-MMKGC/data/MKG-W+'
    target_dir = '/blue/daisyw/ma.haodi/CSProm-KG/data/processed/MKG-W'
    mm_context_file = 'qwen-MKG-W+_img_context-setting_2-v1.json'

    train_triples = defaultdict(set)
    train_triple_file = os.path.join(source_dir, 'train.del')
    with open(train_triple_file, 'r') as f:
        for line in f:
            h_id, r_id, t_id = line.strip().split('\t')
            train_triples[h_id].add((r_id, t_id))

    # entity context
    source_ent_type_file = os.path.join(source_dir, 'entity_types-v6.del')
    ent_type_dict = read_entity_strings(source_ent_type_file)

    # entity id2str
    source_ent_id_file = os.path.join(source_dir, 'entity_mentions.del')
    ent_id2str_dict = read_entity_strings(source_ent_id_file)

    # relation id2str
    source_rel_id_file = os.path.join(source_dir, 'relation_mentions.del')
    rel_id2str_dict = read_entity_strings(source_rel_id_file)

    # triple context
    source_triple_context_file = os.path.join(mm_source_dir, 'context', mm_context_file)
    with open(source_triple_context_file, 'r') as f:
        source_triple_dict = json.load(f)

    # create mm description
    mm_desc_file = os.path.join(target_dir, 'entityid2mm_description.del')
    with open(mm_desc_file, 'w') as f:
        # mm description: entity type + triple context (construct on-the-fly in dataset)
        for ent_id, ent_type in ent_type_dict.items():
            ent_str = ent_id2str_dict[ent_id]
            # random sample 5 triples
            if ent_id in train_triples:
                triples = train_triples[ent_id]
                triples = random.sample(list(triples), min(5, len(triples)))
                f.write(f"{ent_id}\t{ent_str.replace('/', ' , ').replace('_', ' ')} is a {ent_type}. ")
                for triple in triples:
                    r_id, t_id = triple
                    r_str = rel_id2str_dict[r_id]
                    t_str = ent_id2str_dict[t_id]
                    # if mm context is available, use it
                    if t_id in source_triple_dict[ent_id]:
                        mm_context = source_triple_dict[ent_id][t_id]
                        f.write(f"{ent_str}|{r_str}|{t_str}|{mm_context}|SEP|")
                    # otherwise, use the triple
                    else:
                        f.write(f"{ent_str}|{r_str}|{t_str}|SEP|")
            else:
                f.write(f"{ent_id}\t{ent_str.replace('/', ' , ').replace('_', ' ')} is a {ent_type}.")
            f.write("\n")

    # relation context (template + (r, tail_type) triples (construct on-the-fly in dataset))
    source_rel_template_file = os.path.join(source_dir, 'relation_template.del')
    with open(source_rel_template_file, 'r') as f:
        rel_template_dict = {}
        for line in f:
            parts = line.strip().split('\t')
            rel_template_dict[parts[0]] = parts[2]

    # create mm relation description
    num_relations = len(rel_template_dict)
    mm_rel_desc_file = os.path.join(target_dir, 'relationid2mm_description.del')
    with open(mm_rel_desc_file, 'w') as f:
        for rel_id, rel_desc in rel_template_dict.items():
            f.write(f'{(rel_id)}\tTemplate: {rel_desc}\n')
        # construct reverse template
        for rel_id, rel_desc in rel_template_dict.items():
            f.write(f"{int(rel_id) + num_relations}\tTemplate: {rel_desc.replace('[A]', '[B-]').replace('[B]', '[A]').replace('[B-]', '[B]')}\n")


if __name__ == "__main__":
    main()
