#!/bin/bash

while getopts c:r: flag
do
    case "${flag}" in
        c) copy_cartopy=${OPTARG};;
        r) clean_env=${OPTARG};; 
    esac
done

if [ -z clean_env ]
	then
	echo "Hello 1";
	conda env create -f environment.yaml;
	source activate sevir_challenge;
	python -m ipykernel install --user --name sevir_challenge --display-name "Python 3 (sevir_challenge)";

	if [ -v copy_cartopy ]
		then
		echo "Hello 2";
		cp /home/gridsan/groups/EarthIntelligence/engine/working/cartopy_shapefiles/.local/share/cartopy $HOME/.local/share;
	fi
fi
if [ -v clean_env ]
	then
	echo "Hello 3";
	jupyter kernelspec uninstall 'sevir_challenge';
	conda env remove -n 'sevir_challenge';
fi
