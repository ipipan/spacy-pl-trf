import spacy
import json
import re
from pathlib import Path
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Token
from spacy.tokens import Span
from spacy.util import get_model_meta
import platform
if platform.uname().system == "Darwin":
    from . import morf_interface as morfeusz2
else:
    import morfeusz2


SMALL_FREQ_FILE = "freq-nkjp1m.tsv"
BIG_FREQ_FILE = "freq-nkjp300m.tsv"
DIMINUTIVES_FILE = "combined.tab"

Token.set_extension("properness", default=[])
Token.set_extension("style", default=[])
Token.set_extension("disambiguator", default="")
Token.set_extension("is_ign", default=True)
Token.set_extension("freq", default=0)
Token.set_extension("is_diminutive", default=False)
Token.set_extension("diminutive_chain", default=[])

@Language.factory(
        "morfeusz",
        assigns=["token.lemma"],
        default_config={
                "model": None,
                "mode": "lookup",
                "overwrite": False,
                "scorer": {"@scorers": "spacy.lemmatizer_scorer.v1"}
            },
        default_score_weights={"lemma_acc": 1.0})
def make_morfeusz(nlp, model, name, mode, overwrite, scorer):
    return Morfeusz(nlp, name, mode=mode, overwrite=overwrite, scorer=scorer)


class Morfeusz(Pipe):
    def __init__(self, nlp, name, mode, overwrite, scorer):
        self.nlp = nlp
        self.model = mode
        self.name = name
        self.mode = mode
        self.overwrite = overwrite
        self.scorer = scorer
        self.morf = morfeusz2.Morfeusz(expand_tags=True)
        self._flexer = Flexer(self)
        self.freq_dict = self.load_freq_dict()
        self.dim_dict = self.load_diminutives()
        self._qualifier_splitter = re.compile(r"[,|]")

    def load_freq_dict(self, fdict="1M"):
        #fdict is the name of the dictionary
        # "1M" - the basic, hand annotated 1 million token NKJP freq dict
        # "300M" - the extended, automatically annotated 300 million token NKJP freq dict
        model_path = Path(__file__).parent
        meta = get_model_meta(model_path)
        data_dir = f"{meta['lang']}_{meta['name']}-{meta['version']}"
        if fdict == "1M":
            freq_file = SMALL_FREQ_FILE
        elif fdict == "300M":
            freq_file = BIG_FREQ_FILE
        else:
            raise ValueError(f"{fdict} is not supported.")

        data_path = model_path / data_dir / "static" / freq_file
        if not data_path.exists():
            # build mode
            data_path = model_path / "static" / freq_file

        with open(data_path, encoding="utf-8") as f:
            txt = f.read()
        lines = [line.split("\t") for line in txt.split("\n")[:-1]]
        freq_dict = {(lemma, pos): int(freq) for freq, lemma, pos in lines}
        return freq_dict


    def load_diminutives(self):
        model_path = Path(__file__).parent
        meta = get_model_meta(model_path)
        data_dir = f"{meta['lang']}_{meta['name']}-{meta['version']}"
        data_path = model_path / data_dir / "static" / DIMINUTIVES_FILE
        if not data_path.exists():
            # build mode
            data_path = model_path / "static" / DIMINUTIVES_FILE
 
        with open(data_path, encoding="utf-8") as f:
            txt = f.read()

        lines = [line.split("\t") for line in txt.split("\n")[:-1]]
        dim_to_lemmas = {}
        for (lemma, dim) in lines:
            if dim in dim_to_lemmas:
                dim_to_lemmas[dim].append(lemma)
            else:
                dim_to_lemmas[dim] = [lemma]
        return dim_to_lemmas


    def freq_weighting(self, seg):
        freq = self.freq_dict.get((seg["lemma"], seg["pos"]), 0)
        return freq


    def tag_distance(self, tag1, tag2):
        feats1 = set(tag1.split(":"))
        feats2 = set(tag2.split(":"))
        dist = len(feats1.symmetric_difference(feats2))
        return dist


    def generate(self, lemma):
        generated = self.morf.generate(lemma)
        processed = [{"form": g[0], "full_tag": g[2]} for g in generated]
        return processed


    def flex(self, to_inflect, target_feats):
        if isinstance(to_inflect, Token):
            return self._flexer.flex_token(to_inflect, target_feats).strip()

        try:
            _ = to_inflect[0]
        except TypeError:
            raise TypeError("Supplied argument must be either a span, token, or a list of tokens.")

        if isinstance(to_inflect, Span):
            to_inflect = list(to_inflect)
        if not all([isinstance(tok, Token) for tok in to_inflect]):
            raise TypeError("Supplied argument must be either a span, token, or a list of tokens.")

        inflected = self._flexer.flex_tokens(to_inflect, target_feats).strip()
        return inflected


    def lemmatize(self, to_lemmatize):
        try:
            _ = to_lemmatize[0]
        except TypeError:
            raise TypeError("Supplied argument must be either a span, token, or a list of tokens.")

        if isinstance(to_lemmatize, Span):
            to_lemmatize = list(to_lemmatize)

        if not all([isinstance(tok, Token) for tok in to_lemmatize]):
            raise ValueError("Supplied argument must be either a span, token, or a list of tokens.")

        lemmatized = self._flexer.lemmatize_tokens(to_lemmatize).strip()
        return lemmatized


    def split_qualifiers(self, qualifier_list):
        new_qualifiers = []
        for qualifier in qualifier_list:
            new_qualifiers.extend(self._qualifier_splitter.split(qualifier))
        return new_qualifiers


    def unpack_analysis(self, analysis):
        tag = analysis[2][2]
        pos = tag.split(":")[0]
        lemma = analysis[2][1]
        disambiguator = ""
        if ":" in lemma:
            if lemma.startswith(":"):
                pass

            else:
                lemma, disambiguator = lemma.split(":")

        unpacked = {"start": analysis[0],
                    "end": analysis[1],
                    "lemma": lemma,
                    "disambiguator": disambiguator,
                    "tag": tag,
                    "pos": pos,
                    "properness": self.split_qualifiers(analysis[2][3]),
                    "style": self.split_qualifiers(analysis[2][4])}
        return unpacked


    def get_best_analysis(self, tok):
        if isinstance(tok, Token):
            orth = tok.orth_
            tag = tok.tag_
        else:
            # for internal use only
            orth = tok
            tag = ""
        graph = self.morf.analyse(orth)
        unpacked = [self.unpack_analysis(an) for an in graph]
        initial_segments = [seg for seg in unpacked if seg["start"] == 0]
        # sorting by two factors, first frequency, and then by tag proximity
        # therefore frequency serves only as tiebreaker
        for seg in initial_segments:
            seg["freq"] = self.freq_weighting(seg)
        ranking = sorted(initial_segments, key=lambda seg: seg["freq"], reverse=True)
        ranking = sorted(ranking, key=lambda seg: self.tag_distance(tag, seg["tag"]))
        best = ranking[0]
        return best


    def dediminutivize(self, seg):
        to_reduce = seg["lemma"]
        chain = []
        is_diminutive = False
        while to_reduce in self.dim_dict:
            is_diminutive = True
            fuller_forms = self.dim_dict[to_reduce]
            if len(fuller_forms) == 1:
                fuller_form = fuller_forms[0]

            else:
                pos = seg["pos"]
                hypotheses = [(ff, self.freq_dict.get((ff, pos), 0)) for ff in fuller_forms]
                ranking = sorted(hypotheses, key=lambda x:x[1], reverse=True)
                fuller_form = ranking[0][0]
            to_reduce = fuller_form
            chain.append(fuller_form)

        return is_diminutive, chain


    def generate_forms(self, token):
        return self.morf.generate(token.lemma_)


    def annotate(self, doc):
        for tok in doc:
            if tok.orth_.isspace():
                #skipping whitespace tokens, e.g. "\t"
                continue

            best = self.get_best_analysis(tok)
            tag = best["tag"]
            tok._.is_ign = (tag == "ign")
            tok._.properness = best["properness"]
            tok._.style = best["style"]
            tok._.freq = best["freq"]

            is_diminutive, chain = self.dediminutivize(best)
            tok._.is_diminutive = is_diminutive
            tok._.diminutive_chain = chain

            if tok.lemma_ == "" or self.overwrite:
                tok.lemma_ = best["lemma"]
                tok._.disambiguator = best["disambiguator"]


    def __call__(self, doc):
        self.annotate(doc)
        return doc


MORPHOLOGY_FILE = "pl_morph.json"
# we use rules induced from PDB as opposed to NKJP, as these seem more reasonable

class Flexer():
    def __init__(self, morf_component):
        self.morf_component = morf_component
        data = self.load_morphology()
        self.attr2feats = data["ATTR2FEATS"]
        self.val2attr = data["VAL2ATTR"]
        self.governing_deprels = data["GOVERNING_DEPRELS"]
        # a list of deprels with inverted dependency structure (i.e. governing children)
        self.accomodation_rules = data["ACCOMODATION_RULES"]
        # A deprel -> agreement attrs dict

    def load_morphology(self):
        model_path = Path(__file__).parent
        meta = get_model_meta(model_path)
        data_dir = f"{meta['lang']}_{meta['name']}-{meta['version']}"
        data_path = model_path / data_dir / "static" / MORPHOLOGY_FILE
        if not data_path.exists():
            # build mode
            data_path = model_path / "static" / MORPHOLOGY_FILE

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        return data


    def get_case_fun(self, token_string):
        if token_string.isupper():
            case_fun = lambda s: s.upper()
        elif token_string.islower():
            case_fun = lambda s: s.lower()
        elif token_string.istitle():
            case_fun = lambda s: s.capitalize()
        else:
            case_fun = lambda s: s
        return case_fun


    def tag_to_feats(self, tag_string):
        split_tag = tag_string.split(":")
        if len(split_tag) > 1:
            return split_tag[1:]
        return []


    def dict_flex(self, lemma, current_tag, target_feats):
        split_target_feats = target_feats.split(":")
        lexeme = self.morf_component.generate(lemma)
        satisfactory = [g for g in lexeme if all([f in self.tag_to_feats(g["full_tag"]) for f in split_target_feats])]
        if not satisfactory:
            return None
        for entry in satisfactory:
            entry["score"] = self.morf_component.tag_distance(current_tag, entry["full_tag"])
        srt = sorted(satisfactory, key=lambda g:g["score"])
        inflected_form = srt[0]["form"]
        return inflected_form


    def flex_token(self, token, target_feats):
        token_string = token.orth_
        case_fun = self.get_case_fun(token_string)

        if target_feats in ["", None]:
            inflected_form = token.orth_
        else:
            lemma = token.lemma_
            if token._.disambiguator:
                lemma = f"{lemma}:{token._.disambiguator}"
            current_tag = token.tag_
            inflected_form = self.dict_flex(lemma, current_tag, target_feats)
            if inflected_form is None:
                inflected_form = token.orth_

        inflected = case_fun(inflected_form) + token.whitespace_
        return inflected


    def flex_subtree(self, head, target_feats):
        ind_to_inflected = {}
        children = list(head.children)
        children_to_inflect = [child for child in children if child.dep_ not in self.governing_deprels]
        governing_children = [child for child in children if child.dep_ in self.governing_deprels]

        if governing_children:
            inflected_head = head.orth_ + head.whitespace_
            governor = governing_children[0]
            inflected_governor_subtree = self.flex_subtree(governor, target_feats)
            ind_to_inflected.update(inflected_governor_subtree)

        else:
            inflected_head = self.flex_token(head, target_feats)

        ind_to_inflected[head.i] = inflected_head
        for child in children_to_inflect:
            child_deprel = child.dep_
            if child_deprel in self.accomodation_rules:
                accomodable_attrs = self.accomodation_rules[child_deprel]
            else:
                accomodable_attrs = []

            # we're not limiting ourselves to target pattern,
            # but rather propagate all the features of the new tag, which go through the rule
            head_feats_dict = {self.val2attr[val]: val for val in self.tag_to_feats(head.tag_) if val in self.val2attr}# current head feats
            target_feats_dict = {self.val2attr[val]: val for val in target_feats.split(":") if val in self.val2attr}
            head_feats_dict.update(target_feats_dict) # overwriting current features with desired ones
            target_child_feats = list(head_feats_dict.values())

            accomodable_feats = [feat for feat in target_child_feats if self.val2attr[feat] in accomodable_attrs]
            child_pattern = ":".join(accomodable_feats)
            inflected_subtree = self.flex_subtree(child, child_pattern)
            ind_to_inflected.update(inflected_subtree)
        return ind_to_inflected


    def flex_tokens(self, tokens, target_feats):
        ind_to_inflected = {}
        independent_subtrees = [tok for tok in tokens if tok.head not in tokens or tok.head == tok]
        for independent_head in independent_subtrees:
            ind_to_inflected.update(self.flex_subtree(independent_head, target_feats))
        token_inds = [tok.i for tok in tokens]
        inflected_seq = sorted([(i, tok) for i, tok in ind_to_inflected.items() if i in token_inds])
        # we restrict the inflected tokens, to ones in the original list
        inflected_string = "".join([tok for i, tok in inflected_seq])
        return inflected_string


    def lemmatize_subtree(self, head):
        # The algorithm recurrently goes through each child and inflects it into the pattern
        # corresponding to the base form of the head of the phrase.
        ind_to_lemmatized = {}
        children = list(head.children)
        children_to_lemmatize = [child for child in children if child.dep_ not in self.governing_deprels]
        governing_children = [child for child in children if child.dep_ in self.governing_deprels]
        if governing_children:
            governor = governing_children[0]
            lemmatized_governor_subtree = self.lemmatize_subtree(governor)
            ind_to_lemmatized.update(lemmatized_governor_subtree)
            ind_to_lemmatized[head.i] = head.orth_
            target_pattern = ""

        else:
            # BASIC:
            lemmatized_head = head.lemma_ + head.whitespace_
            ind_to_lemmatized[head.i] = lemmatized_head
            target_tag = self.morf_component.get_best_analysis(head.lemma_)["tag"]

        for child in children_to_lemmatize:
            child_deprel = child.dep_
            if child_deprel in self.accomodation_rules:
                    accomodable_attrs = self.accomodation_rules[child_deprel]
                    feats = [feat for feat in self.tag_to_feats(target_tag) if feat in self.val2attr]
                    # limiting to supported features

                    accomodable_feats = [feat for feat in feats if self.val2attr[feat] in accomodable_attrs]
                    child_pattern = ":".join(accomodable_feats)
                    lemmatized_subtree = self.flex_subtree(child, child_pattern)
            else:
                    lemmatized_subtree = {tok.ind: tok.orth_ + tok.whitespace_ for tok in child.subtree}
            ind_to_lemmatized.update(lemmatized_subtree)
        return ind_to_lemmatized


    def lemmatize_tokens(self, tokens):
        ind_to_lemmatized = {}
        independent_subtrees = [tok for tok in tokens if tok.head not in tokens or tok.head == tok]
        for independent_head in independent_subtrees:
            ind_to_lemmatized.update(self.lemmatize_subtree(independent_head))
        token_inds = [tok.i for tok in tokens]
        lemmatized_seq = sorted([(i, tok) for i, tok in ind_to_lemmatized.items() if i in token_inds])
        # we restrict the lemmatized tokens, to ones in the original list
        lemmatized_string = "".join([tok for i, tok in lemmatized_seq])
        return lemmatized_string


