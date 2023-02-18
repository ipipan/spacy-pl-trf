#!/bin/bash
mkdir -p resources/UD_Polish-PDB/
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-train.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-dev.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-dev.conllu 
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-test.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-test.conllu

mkdir -p resources/NKJP
wget 'http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz' -O resources/NKJP/nkjp.tgz
tar -xvf resources/NKJP/nkjp.tgz -C resources/NKJP/
rm resources/NKJP/nkjp.tgz

mkdir -p resources/NKJP_UD
wget 'http://git.nlp.ipipan.waw.pl/alina/PDBUD/repository/archive.zip?ref=master' -O resources/NKJP_UD/nkjp_ud.zip
unzip resources/NKJP_UD/nkjp_ud.zip -d resources/NKJP_UD/
mv resources/NKJP_UD/PDBUD-master-f7a92fb7ceeee5b3f806eb2394aa19d37675fade/NKJP1M-UD/NKJP1M-UD.conllu resources/NKJP_UD/
