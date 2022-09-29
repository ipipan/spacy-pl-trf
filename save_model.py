import json
import spacy
import click
from morf import *
from vectorizer import *

@click.command()
@click.option("--version", prompt="model version")
@click.option("--model_dir", prompt="trained model directory")
@click.option("--model_name", prompt="out dir")
def main(version, model_dir, model_name):
    nlp = spacy.load(model_dir)
    nlp.add_pipe("morfeusz")
    trf = nlp.get_pipe("transformer")
    trf.set_extra_annotations = trf_to_vec_annotation_setter

    meta = nlp.meta
    meta["name"] = model_name
    meta["version"] = version
    meta["author"] = "Ryszard Tuora, ICS PAS"
    meta["email"] = "ryszardtuora@gmail.com"
    meta["url"] = "http://zil.ipipan.waw.pl/RyszardTuora"

    out_path = f"{meta['lang']}_{meta['name']}-{meta['version']}"
    nlp.to_disk(out_path)

if __name__ == '__main__':
    main()
