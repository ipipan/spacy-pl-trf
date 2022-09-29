#!/bin/bash
mkdir -p resources/UD_Polish-PDB/
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-train.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-dev.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-dev.conllu 
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/master/pl_pdb-ud-test.conllu -O resources/UD_Polish-PDB/pl_pdb-ud-test.conllu
mkdir -p resources/NKJP
mkdir -p resources/NKJP_UD
