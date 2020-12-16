import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clean', action='store_true', help='remove sevir_challenge conda environment')
parser.add_argument('--cc', action='store_true', help='copy cartpty files on grid')

args = parser.parse_args()

clean = args.clean
copy_cartopy = args.cc

##Remove the python environment (both from Jupyter notebook and terminal)
if clean:
	subprocess.run("jupyter kernelspec uninstall 'sevir_challenge'", shell=True)
	subprocess.run("conda env remove -n 'sevir_challenge'", shell=True)
##Install the environment
else:
	subprocess.run("conda env create -f environment.yml", shell=True)
	subprocess.run("source activate sevir_challenge && python -m ipykernel install --user --name sevir_challenge --display-name 'Python 3 (sevir_challenge)' && source deactivate", shell=True)
	#subprocess.run('python -m ipykernel install --user --name sevir_challenge --display-name "Python 3 (sevir_challenge)"', shell=True)

	if copy_cartopy:
		subprocess.run('cp -r /home/gridsan/groups/EarthIntelligence/engine/working/cartopy_shapefiles/.local/share/cartopy $HOME/.local/share', shell=True)

