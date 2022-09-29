0. Install the build requirements:
`python3 -m pip install -r requirements.txt`

1. Download the resources:
`./get_resources.sh`
NKJP and NKJP1M-UD have to be downloaded manually from http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=view&target=NKJP-PodkorpusMilionowy-1.2.tar.gz and http://git.nlp.ipipan.waw.pl/alina/PDBUD/blob/master/NKJP1M-UD/ respectively. Extract these into their respective folders in `resources` in accordance with the paths specified in **process_nkjp.py**.

2. Convert the resources by:
`python3 process_nkjp.py`

this will produce three files:
`nkjp_train.spacy, nkjp_dev.spacy, nkjp_test.spacy`

These will correspond to files mentioned in the config file **config.cfg**. Editing the training parameters is done via editing the config file.

3. Train the model via:
`./train.sh -c config.cfg -o model_out`

4. Generate the package via:
`./generate_package.sh -m model_out/model-best -v 0.0.6`

The package will be saved in the tar.gz archive referenced in the output. This archive can be instaled via pip, e.g.
`python3 -m pip install pkgmodel/pl_nask-0.0.6/dist/pl_nask-0.0.6.tar.gz`

5. Run the test script, to verify whether it works.
`python3 test.py`

6. Run the evaluation script on the test data:
`./evaluate.sh >> evaluation_results.md`
