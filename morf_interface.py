import re
import json
from pathlib import Path
from itertools import product
import pexpect
from spacy.util import get_model_meta

model_path = Path(__file__).parent
meta = get_model_meta(model_path)
data_dir = f"{meta['lang']}_{meta['name']}-{meta['version']}"
style_path = model_path / data_dir / "static" / "style_qualifiers.json"
properness_path = model_path / data_dir / "static" / "properness_qualifiers.json"
MORF_BIN_PATH = Path("/usr/local/bin")
analyser_path = str(MORF_BIN_PATH / "morfeusz_analyzer")
generator_path = str(MORF_BIN_PATH / "morfeusz_generator")

class MorfInterface():
    # This is a substitute for Python bindings to Morfeusz2 it is intended mainly for Mac OS.
    # It works by starting an interactive process using the analyzer and generator binaries
    # and then sending data and parsing outputs on demand.
    # Morfeusz binaries should be placed in MORF_BIN_PATH
    def __init__(self, expand_tags=True):
        #expand_tags is just for API compatibility with bindings
        self.analyser = pexpect.spawn(analyser_path, encoding="utf-8")
        self.analyser.delaybeforesend = None
        self.generator = pexpect.spawn(generator_path, encoding="utf-8")
        self.generator.delaybeforesend = None

        self.qualifier_splitter = re.compile(r"[,|]")
        with open(style_path, encoding="utf-8") as f:
            self.style_qualifiers = json.load(f)

        with open(properness_path, encoding="utf-8") as f:
            self.properness_qualifiers = json.load(f)

    def dedotify(self, tag):
        # this is for handling the packed notation for tags (which uses dots)
        elements = tag.split(":")
        dimensions = []
        for el in elements:
            dimensions.append(el.split("."))
        combinations = product(*dimensions)
        possible_tags = [":".join(combination) for combination in combinations]
        return possible_tags

    def parse(self, output, mode):
        # parse the output string into structured data
        output = output.strip()
        if mode == "analysis":
            start, end, form, lemma, tag, qualifiers = output.split(",", 5)
            start = int(start)
            end = int(end)
        else:
            form, lemma, tag, qualifiers = output.split(",", 3)
            form = form.strip()

        qualifiers = self.qualifier_splitter.split(qualifiers)
        style = [q for q in qualifiers if q in self.style_qualifiers]
        properness = [q for q in qualifiers if q in self.properness_qualifiers]
        tag_combinations = self.dedotify(tag)
        interpretations = [(form, lemma, tag_interp, ["|".join(properness)], ["|".join(style)]) for tag_interp in tag_combinations]
        if mode == "analysis":
            interpretations = [(start, end, interp) for interp in interpretations]
        return interpretations

    def analyse(self, form):
        # provide analyses for a single token
        if form == ",":
            #special case as comma is used for splitting fields
            return [(0, 1, (",", ",", "interp", [], []))]

        self.analyser.sendline(form)
        self.analyser.expect("\r\n\[")
        self.analyser.expect("]\r\n$")
        analyses = self.analyser.before.replace("]\r\n[", "\r\n").split("\r\n")
        parsed = []
        for analysis in analyses:
            parsed.extend(self.parse(analysis, "analysis"))
        return parsed


    def generate(self, lemma):
        # generate forms for a single lemma
        if lemma == ",":
            #special case as comma is used for splitting fields
            return [(",", ",", "interp", [], [])]

        self.generator.sendline(lemma)
        self.generator.expect("\r\n\[")
        self.generator.expect("]\r\n$")
        syntheses = self.generator.before.replace("]\r\n[", "\r\n").split("\r\n")
        parsed = []
        for synthesis in syntheses:
            parsed.extend(self.parse(synthesis, "generation"))
        return parsed

Morfeusz = MorfInterface
