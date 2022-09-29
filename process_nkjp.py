import os
import random
import conllu
import spacy
from tqdm import tqdm
from lxml import etree
from spacy.tokens import DocBin, Doc


NAMESPACES = {"x":"http://www.tei-c.org/ns/1.0"}
WORDS_FILE = "ann_morphosyntax.xml"
MORPH_FILE = "ann_morphosyntax.xml"
NER_FILE = "ann_named.xml"
SEGMENTATION_FILE = "ann_segmentation.xml"
CONLLU_PATH = "resources/NKJP_UD/NKJP1M-UD.conllu"
XML_DIR = "resources/NKJP/"
PDB_TEST = "resources/UD_Polish-PDB/pl_pdb-ud-test.conllu"
PDB_DEV = "resources/UD_Polish-PDB/pl_pdb-ud-dev.conllu"


def parse_xml(path):
    if not os.path.exists(path):
        return None
    et = etree.parse(path)
    rt = et.getroot()
    return rt

def get_node_id(node):
    # get the id from the xml node
    return node.get('{http://www.w3.org/XML/1998/namespace}id')


def extract_entities_from_subfolder(subfolder):
    # read the ner annotation from a subfolder, assign it to paragraphs
    # also returns paragraph lengths for purposes of removing misaligned data
    ner_path = os.path.join(XML_DIR, subfolder, NER_FILE)
    rt = parse_xml(ner_path)
    if rt is None:
        return {}, {}
    subfolder_entities = {}
    ner_pars = rt.xpath("x:TEI/x:text/x:body/x:p", namespaces=NAMESPACES)
    for par in ner_pars:
        par_entities = {}
        _, par_id = get_node_id(par).split("_")
        ner_sents = par.xpath("x:s", namespaces=NAMESPACES)
        for ner_sent in ner_sents:
            corresp = ner_sent.get("corresp")
            _, ner_sent_id  = corresp.split("#morph_")
            par_entities[ner_sent_id] = extract_entities_from_sentence(ner_sent)
        subfolder_entities[par_id] = par_entities
    par_id_to_len = assign_target_ids(subfolder, subfolder_entities)
    return subfolder_entities, par_id_to_len


def extract_entities_from_sentence(ner_sent):
    # extracts all the entity dicts from the sentence
    # we assume that an entity cannot span across sentences
    segs = ner_sent.xpath("x:seg", namespaces=NAMESPACES)
    sent_entities = {}
    for i, seg in enumerate(segs):
        ent_id = get_node_id(seg)
        targets = [ptr.get("target") for ptr in seg.xpath("x:ptr", namespaces=NAMESPACES)]
        orth = seg.xpath("x:fs/x:f[@name='orth']/x:string", namespaces=NAMESPACES)[0].text
        ner_type = seg.xpath("x:fs/x:f[@name='type']/x:symbol", namespaces=NAMESPACES)[0].get("value")
        ner_subtype_node = seg.xpath("x:fs/x:f[@name='subtype']/x:symbol", namespaces=NAMESPACES)
        if ner_subtype_node:
            ner_subtype = ner_subtype_node[0].get("value")
        else:
            ner_subtype = None
        entity = {"ent_id": ent_id,
                  "index": i,
                  "orth": orth,
                  "ner_type": ner_type,
                  "ner_subtype": ner_subtype,
                  "targets": targets}
        sent_entities[ent_id] = entity
    cleared_entities = clear_entities(sent_entities)
    return cleared_entities


def clear_entities(entities):
    # eliminates entities which extend beyond our scope
    resolve_entities(entities)
    entities_list = sorted(list(entities.values()), key=lambda ent: ent["index"])
    entities = eliminate_overlapping_entities(entities_list)
    for entity in entities:
        targets = entity["targets"]
        entity["targets"] = [t.split("morph_")[1] for t in targets]
    return entities


def resolve_entities(entities):
    # assign morphological level targets to entities
    resolved_targets = {entity_id: resolve_entity(entity, entities) for entity_id, entity in entities.items()}
    for entity_id in entities:
        entities[entity_id]["targets"] = resolved_targets[entity_id]


def resolve_entity(entity, entities):
    # translate targets defined in terms of entities, into morphological units
    # works recurrently
    targets = entity["targets"]
    resolved = []
    for target in targets:
        if target.startswith("named_"):
            target_entity = entities[target]
            resolved.extend(resolve_entity(target_entity, entities))
        else:
            resolved.append(target)
    return resolved


def eliminate_overlapping_entities(entities_list):
    # we eliminate entities which are entirely contained in another
    subsumed = set([])
    for sub_i, sub in enumerate(entities_list):
        for over in entities_list[:sub_i]:
            if any([target in over["targets"] for target in sub["targets"]]):
                subsumed.add(sub["ent_id"])
    return [entity for entity in entities_list if entity["ent_id"] not in subsumed]

def get_seg_tag(seg):
    interp = seg.xpath("x:fs/x:f[@name='disamb']/x:fs/x:f[@name='interpretation']/x:string", namespaces=NAMESPACES)[0].text
    if interp.startswith("::"):
        # this encodes a ":" occuring in text
        tag = "interp"
    else:
        _, tag = interp.split(":", 1)
    return tag


def is_new_token(seg, tag, prev_tag):
    # based on segmentation, decides whether the segment begins a new token or not
    nps = seg.xpath("x:fs/x:f[@name='nps']/x:binary", namespaces=NAMESPACES)
    if nps:
        no_preceding_space = True
    else:
        no_preceding_space = False
    is_new = not no_preceding_space # default situation
    if tag == "interp":
        #punctuation
        is_new = True
    else:
        if prev_tag == "interp":
            # no space because this word occurs after interp e.g. "(aczkolwiek ...)"
            is_new = True
    return is_new

def assign_target_ids(subfolder, subfolder_entities):
    # translates xml target ids into order-based identifiers
    # additionally returns the information about paragraph lengths
    morph_path = os.path.join(XML_DIR, subfolder, MORPH_FILE)
    rt = parse_xml(morph_path)
    morph_pars = rt.xpath("x:TEI/x:text/x:body/x:p", namespaces=NAMESPACES)
    id_to_position = {}
    par_id_to_len = {}
    for par in morph_pars:
        _, par_id = get_node_id(par).split("_")
        morph_sents = par.xpath("x:s", namespaces=NAMESPACES)
        sent2tokens = {}
        token_ind = -1
        sent_lens = []
        for morph_sent in morph_sents:
            _, morph_id = get_node_id(morph_sent).split("_")
            segs = morph_sent.xpath("x:seg", namespaces=NAMESPACES)
            prev_tag = None
            sent_start = token_ind
            for seg in segs:
                _, seg_id = get_node_id(seg).split("morph_")
                tag = get_seg_tag(seg)
                new_token = is_new_token(seg, tag, prev_tag)
                prev_tag = tag
                if new_token:
                    token_ind += 1
                id_to_position[seg_id] = token_ind
            sent_len = token_ind - sent_start
            sent_lens.append(sent_len)
        par_id_to_len[par_id] = sent_lens

    for par_key in subfolder_entities:
        par_sents = subfolder_entities[par_key]
        for sent_key in par_sents:
            sent_entities = par_sents[sent_key]
            for entity in sent_entities:
                entity["targets"] = sorted(list(set([id_to_position[target] for target in entity["targets"]])))
    return par_id_to_len


def retokenize_conllu_sent(sent):
    # merges subtokens, and transfers their properties onto the result
    id_to_token = {tok["id"]:tok for tok in sent}
    to_subsume = []
    for i, token in enumerate(sent):
        if type(token["id"]) == tuple:
            start, _, end = token["id"]
            sub_ids = range(start, end+1)
            to_subsume.extend([tok for tok in sent if tok["id"] in sub_ids])
            sub_tokens = [id_to_token[sub_id] for sub_id in sub_ids]
            main_sub = [tok for tok in sub_tokens if tok["head"] not in sub_ids][0]
            for sub_token in sub_tokens:
                sub_token["inheritor"] = main_sub["id"]
            keys = [key for key in token.keys() if key not in ["form", "misc", "deps"]]
            token.update({k: main_sub[k] for k in keys})
    non_subsumed = [tok for tok in sent if tok not in to_subsume]
    old_id_to_new_id = {tok["id"]: i+1 for i, tok in enumerate(non_subsumed)}
    for sub_tok in to_subsume:
        old_id_to_new_id[sub_tok["id"]] = old_id_to_new_id[sub_tok["inheritor"]]
    for tok in non_subsumed:
        if tok["head"] != 0:
            tok["head"] = old_id_to_new_id[tok["head"]]
        tok["id"] = old_id_to_new_id[tok["id"]]
    for tok in to_subsume:
        sent.remove(tok)
    return sent


def extract_seg_ids(subfolder):
    # extracts sentence ids for bridging between the ner file and the morph file
    seg_path = os.path.join(XML_DIR, subfolder, SEGMENTATION_FILE)
    rt = parse_xml(seg_path)
    if rt is None:
        return {}
    seg_sents = rt.xpath("x:TEI/x:text/x:body/x:p/x:s", namespaces=NAMESPACES)
    seg_id_to_par_id = {}
    seg_id_to_index = {}
    for i, seg_sent in enumerate(seg_sents):
        _, seg_id = get_node_id(seg_sent).split("_")
        seg_par = seg_sent.getparent()
        _, par_id = get_node_id(seg_par).split("_")
        seg_id_to_par_id[seg_id] = par_id
        seg_id_to_index[seg_id] = i
    return seg_id_to_par_id, seg_id_to_index


def load_xml_nkjp():
    subfolder_to_entities = {}
    seg_id_to_par_id = {}
    seg_id_to_index = {}
    subfolder_to_lens = {}
    for subfolder in tqdm([name for name in os.listdir(XML_DIR) if os.path.isdir(os.path.join(XML_DIR, name))]):
        subfolder_to_entities[subfolder], subfolder_to_lens[subfolder] = extract_entities_from_subfolder(subfolder)
        seg_id_to_par_id[subfolder], seg_id_to_index[subfolder] = extract_seg_ids(subfolder)
    return subfolder_to_entities, seg_id_to_par_id, seg_id_to_index, subfolder_to_lens


def assign_named_entities_to_conllu_sent(sent, subfolder_to_entities, seg_id_to_par_id, seg_id_to_index):
    metadata = sent.metadata
    original_folder, piece_id = metadata["orig_file_sentence"].split("_morph_")
    segmentation_id, _ = piece_id.split("#")
    par_id = seg_id_to_par_id[original_folder][segmentation_id]
    par_position = seg_id_to_index[original_folder][segmentation_id]
    metadata["par_position"] = par_position
    metadata["entities"] = {}
    if original_folder in subfolder_to_entities:
        folder_entities = subfolder_to_entities[original_folder]
        if par_id in folder_entities:
            par_entities = folder_entities[par_id]
            sent_entities = par_entities[segmentation_id]
            metadata["entities"] = sent_entities
    metadata["original_folder"] = original_folder
    metadata["par_id"] = par_id


def triple_split(docs_to_pars, subfolder_to_lens):
    # Split the data in accordance with PDB
    with open(PDB_TEST) as f:
        test_txt = f.read()
    test_sents = conllu.parse(test_txt)
    test_sent_texts = [s.metadata["text"] for s in test_sents]
    with open(PDB_DEV) as f:
        dev_txt = f.read()
    dev_sents = conllu.parse(dev_txt)
    dev_sent_texts = [s.metadata["text"] for s in dev_sents]

    total_paragraphs = 0
    train_pars, dev_pars, test_pars = [], [], []
    for doc_id in tqdm(list(docs_to_pars.keys())):
        par_to_sents = docs_to_pars[doc_id]
        for par_id in par_to_sents:
            sents = par_to_sents[par_id]
            par_lens = [len(sent) for sent in sents]
            total_paragraphs += 1
            if par_id not in subfolder_to_lens[doc_id]:
                # missing annotation
                continue
            if par_lens != subfolder_to_lens[doc_id][par_id]:
                # tokenization is different
                continue
            sections = []
            for sent in sents:
                sent_text = sent.metadata["text"]
                if sent_text in test_sent_texts:
                    sections.append("test")
                elif sent_text in dev_sent_texts:
                    sections.append("dev")
                else:
                    sections.append("train")
            if set(sections) == set(["train"]):
                train_pars.append(sents)
            else:
                if "test" not in sections:
                    # we want to make sure no test par gets seen by the model
                    dev_pars.append(sents)
                else:
                    test_pars.append(sents)
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Filtered paragraphs: {len(train_pars) + len(test_pars) + len(dev_pars)}")
    print(f"Removed paragraphs: {total_paragraphs - (len(train_pars) + len(test_pars) + len(dev_pars))}")
    print(f"Train paragraphs: {len(train_pars)}")
    print(f"Dev paragraphs: {len(dev_pars)}")
    print(f"Test paragraphs: {len(test_pars)}")
    return train_pars, dev_pars, test_pars


def create_doc_objs(par_list, nlp):
    doc_bin = DocBin()
    for sents in par_list:
        words, spaces = [], []
        tags, pos, morphs, lemmas, heads, deps, sent_starts = [], [], [], [], [], [], []
        token_index = 0
        ent_annotations = []
        for sent_i, sent in enumerate(sents):
            if sent.metadata["sent_id"] in ["s45074", "s45075"]:
                # These sents contain cycles
                continue
            sent_start_index = token_index
            entities = sent.metadata["entities"]
            ent_annotations.extend(["O" for _ in sent])
            for token in sent:
                words.append(token["form"])
                misc = token["misc"]
                space_after = True
                if misc and "SpaceAfter" in misc and misc["SpaceAfter"] == "No":
                    space_after = False
                spaces.append(space_after)
                tags.append(token["xpostag"])
                pos.append(token["upostag"])
                if token["feats"]:
                    feats = "|".join([f"{attr}={val}" for attr, val in token["feats"].items()])# filter attrs here?
                else:
                    feats = ""
                morphs.append(feats)
                lemmas.append(token["lemma"])
                head_index = token["head"] - 1 + sent_start_index
                if token["head"] == 0:
                    head_index = token_index
                heads.append(head_index)
                deprel = token["deprel"]
                if deprel == "root":
                    deprel = deprel.upper()
                deps.append(deprel)
                if token_index == sent_start_index:
                    sent_starts.append(True)
                else:
                    sent_starts.append(False)
                token_index += 1
            try:
                for entity in entities:
                    targets = entity["targets"]
                    ent_type = entity["ner_type"].upper()
                    ent_annotations[targets[0]] = f"B-{ent_type}"
                    for inside in targets[1:]:
                        ent_annotations[inside] = f"I-{ent_type}"
            except IndexError:
                print("Indexing error: ", sent)
        doc_obj = Doc(vocab=nlp.vocab,
                      words=words,
                      spaces=spaces,
                      tags=tags,
                      pos=pos,
                      morphs=morphs,
                      lemmas=lemmas,
                      heads=heads,
                      deps=deps,
                      sent_starts=sent_starts,
                      ents=ent_annotations,
                      )
        doc_bin.add(doc_obj)
    return doc_bin


def main():
    # Load XML NKJP
    subfolder_to_entities, seg_id_to_par_id, seg_id_to_index, subfolder_to_lens = load_xml_nkjp()

    # Load UD NKJP
    with open(CONLLU_PATH) as f:
        txt = f.read()
    nkjp_sents = conllu.parse(txt)
    retokenized_nkjp_sents = [retokenize_conllu_sent(sent) for sent in nkjp_sents]

    docs_to_pars = {}
    for sent in tqdm(retokenized_nkjp_sents):
        assign_named_entities_to_conllu_sent(sent, subfolder_to_entities, seg_id_to_par_id, seg_id_to_index)
        original_folder = sent.metadata["original_folder"]
        par_id = sent.metadata["par_id"]
        if original_folder in docs_to_pars:
            if par_id in docs_to_pars[original_folder]:
                docs_to_pars[original_folder][par_id].append(sent)
            else:
                docs_to_pars[original_folder][par_id] = [sent]
        else:
            docs_to_pars[original_folder] = {par_id: [sent]}

    for folder in docs_to_pars:
        for par_id in docs_to_pars[folder]:
            sent_list = docs_to_pars[folder][par_id]
            docs_to_pars[folder][par_id] = sorted(sent_list, key=lambda sent: sent.metadata["par_position"])

    nlp = spacy.blank("pl")
    train_pars, dev_pars, test_pars = triple_split(docs_to_pars, subfolder_to_lens)
    train_docs = create_doc_objs(train_pars, nlp)
    dev_docs = create_doc_objs(dev_pars, nlp)
    test_docs = create_doc_objs(test_pars, nlp)
    train_docs.to_disk("nkjp_train.spacy")
    dev_docs.to_disk("nkjp_dev.spacy")
    test_docs.to_disk("nkjp_test.spacy")
    return train_docs, dev_docs, test_docs

if __name__ == "__main__":
    main()
