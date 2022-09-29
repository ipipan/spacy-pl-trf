#!/bin/bash
# example usage: ./test_package.sh -p pkgmodel/pl_nask-$version/dist/pl_nask-0.0.5.tar.gz
while getopts p: flag
do
        case "${flag}" in
                p) package_path=${OPTARG};;
        esac
done

if [ -z "$package_path" ]; then echo "no package path given"; exit 1; fi

echo Testing model package from $package_path

python3 -m pip uninstall pl_nask
python3 -m pip install $package_path
python3 test.py
