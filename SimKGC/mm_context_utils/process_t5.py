import os
import re

def read_entity_strings(file_path):
    entity_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            entity_dict[parts[0]] = parts[1]
    return entity_dict

def main():
    ''' From MMKGC T5 data fromat to CSProm-KG data format. '''
    source_dir = '/blue/daisyw/ma.haodi/MMKGC/LLM-MMKGC/data/MKG-W'
    target_dir = '/blue/daisyw/ma.haodi/CSProm-KG/data/processed/MKG-W'

    # ent2id files
    source_ent_id_file = os.path.join(source_dir, 'entity_ids.del')
    target_ent_id_file = os.path.join(target_dir, 'entity2id.txt')
    with open(source_ent_id_file, 'r') as f:
        lines = f.readlines()
        num_entities = len(lines)
        with open(target_ent_id_file, 'w') as f_out:
            f_out.write(str(num_entities) + '\n')
            for i, line in enumerate(lines):
                e_idx, e_id = line.strip().split('\t')
                f_out.write(f'{e_id}\t{e_idx}\n')

    # ent id2 description files
    source_ent_desc_file = os.path.join(source_dir, 'entity_desc.del')
    target_ent_desc_file = os.path.join(target_dir, 'entityid2description.txt')
    with open(source_ent_desc_file, 'r') as f:
        lines = f.readlines()
        num_entities = len(lines)
        with open(target_ent_desc_file, 'w') as f_out:
            f_out.write(str(num_entities) + '\n')
            for i, line in enumerate(lines):
                e_idx, e_desc = line.strip().split('\t')
                f_out.write(f'{e_idx}\t{e_desc}\n')

    # ent id2name files
    source_ent_name_file = os.path.join(source_dir, 'entity_mentions.del')
    target_ent_name_file = os.path.join(target_dir, 'entityid2name.txt')
    ent_str_dict = read_entity_strings(os.path.join(source_dir, 'entity_mentions.del'))
    with open(source_ent_name_file, 'r') as f:
        lines = f.readlines()
        num_entities = len(lines)
        with open(target_ent_name_file, 'w') as f_out:
            f_out.write(str(num_entities) + '\n')
            for i, line in enumerate(lines):
                e_idx, e_name = line.strip().split('\t')
                e_name = e_name.replace('/', ' , ').replace('_', ' ')
                f_out.write(f'{e_idx}\t{e_name}\n')

    # rel2id files
    # could be a quick fix for MMT5, all these datasets have str like relation_id. only MKG ones have ids like `Pxx`
    source_rel_id_file = os.path.join(source_dir, 'relation_mentions.del')
    target_rel_id_file = os.path.join(target_dir, 'relation2id.txt')
    with open(source_rel_id_file, 'r') as f:
        lines = f.readlines()
        num_relations = len(lines)
        with open(target_rel_id_file, 'w') as f_out:
            f_out.write(str(num_relations) + '\n')
            for i, line in enumerate(lines):
                r_idx, r_id = line.strip().split('\t')
                f_out.write(f'{r_id}\t{r_idx}\n')

    # rel id2name files
    source_rel_name_file = os.path.join(source_dir, 'relation_mentions.del')
    target_rel_name_file = os.path.join(target_dir, 'relationid2name.txt')
    rel_str_dict = read_entity_strings(os.path.join(source_dir, 'relation_mentions.del'))
    with open(source_rel_name_file, 'r') as f:
        lines = f.readlines()
        num_relations = len(lines)
        with open(target_rel_name_file, 'w') as f_out:
            f_out.write(str(num_relations) + '\n')
            for i, line in enumerate(lines):
                r_idx, r_name = line.strip().split('\t')
                r_name = r_name.replace('/', ' , ').replace('_', ' ')
                f_out.write(f"{r_idx}\t{r_name}\n")

    # rel id2name (with reverse relations) files
    source_rel_name_file_1 = os.path.join(source_dir, 'relation_mentions.del')
    source_rel_name_file_2 = os.path.join(source_dir, 'reverse_relation_mentions.del')
    target_rel_name_file = os.path.join(target_dir, 'relationid2name_reverse.txt')
    rel_str_dict = read_entity_strings(os.path.join(source_dir, 'relation_mentions.del'))
    with open(source_rel_name_file_1, 'r') as f:
        lines = f.readlines()
        num_relations = len(lines)
        with open(target_rel_name_file, 'w') as f_out:
            f_out.write(str(num_relations * 2) + '\n')
            for i, line in enumerate(lines):
                r_idx, r_name = line.strip().split('\t')
                r_name = r_name.replace('/', ' , ').replace('_', ' ')
                f_out.write(f"{r_idx}\t{r_name}\n")
    with open(source_rel_name_file_2, 'r') as f:
        lines = f.readlines()
        with open(target_rel_name_file, 'a') as f_out:
            for i, line in enumerate(lines):
                r_idx, r_name = line.strip().split('\t')
                r_name = r_name.replace('/', ' , ').replace('_', ' ')
                f_out.write(f"{int(r_idx) + num_relations}\t{r_name}\n")

    # train/valid/test2id files & 2id_name files
    for split in ['train', 'valid', 'test']:
        source_file = os.path.join(source_dir, f'{split}.del')
        target_id_file = os.path.join(target_dir, f'{split}2id.txt')
        target_name_file = os.path.join(target_dir, f'{split}2id_name.txt')
        with open(source_file, 'r') as f:
            lines = f.readlines()
            num_triples = len(lines)
            with open(target_id_file, 'w') as f_out:
                f_out.write(str(num_triples) + '\n')
                for i, line in enumerate(lines):
                    h, r, t = line.strip().split('\t')
                    f_out.write(f'{h} {t} {r}\n')

            with open(target_name_file, 'w') as f_out:
                f_out.write(str(num_triples) + '\n')
                for i, line in enumerate(lines):
                    h, r, t = line.strip().split('\t')
                    # convert id to name
                    h = ent_str_dict[h].replace('/', ' , ').replace('_', ' ')
                    t = ent_str_dict[t].replace('/', ' , ').replace('_', ' ')
                    r = rel_str_dict[r]
                    f_out.write(f'{h} | {t} | {r}\n')


if __name__ == "__main__":
    main()
