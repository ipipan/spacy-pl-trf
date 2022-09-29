#!/bin/bash
# example usage: ./generate_package.sh -m trained/trf_newer -v 0.0.6
set -e
while getopts m:v: flag
do
	case "${flag}" in
		m) trained_model_dir=${OPTARG};;
		v) model_version=${OPTARG};;
	esac
done

if [ -z "$trained_model_dir" ]; then echo "no model path (-m) given"; exit 1; fi
if [ -z "$model_version" ]; then echo "no model version (-v) given"; exit 1; fi

echo Generating model package from $trained_model_dir, version $model_version
cp $trained_model_dir/meta.json .
python3 save_model.py --version $model_version --model_dir $trained_model_dir --model_name nask
sed -i 's/spacy-transformers.null_annotation_setter.v1/spacy-transformers.trf_to_vec_annotation_setter.v1/g' pl_nask-$model_version/config.cfg
mkdir -p pkgmodel
mkdir pl_nask-$model_version/static
cp -r static/* pl_nask-$model_version/static
python3 -m spacy package pl_nask-$model_version pkgmodel --code morf.py,vectorizer.py,morf_interface.py
echo Model saved to pkgmodel/pl_nask-$model_version/dist/pl_nask-$model_version.tar.gz
