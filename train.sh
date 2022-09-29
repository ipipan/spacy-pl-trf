#!/bin/bash
# example usage: ./train.sh -c config.cfg -o model_out
while getopts c:o: flag
do
	case "${flag}" in
		c) config=${OPTARG};;
		o) output_path=${OPTARG};;
	esac
done

if [ -z "$config" ]; then echo "no config file (-c) provided"; exit 1; fi
if [ -z "$output_path" ]; then echo "no output path (-o) given"; exit 1; fi

python3 -m spacy train $config --output $output_path
