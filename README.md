# State-Space Modeling Pipeline: NanoNeuroTechnology lab

**Author**: Bhavya Surapaneni

lfads-env is for lfads code from LDNS repository

state-space-neuro uv env is for everything else

to switch to lfads-env run:

deactivate

source lfads-env/bin/activate

to switch back run:

deactivate

source .venv/bin/activateu

to run things in this other env use --active to target the correct venv


all code that uses SSM relies on the most updated version of SSM, which has not been uploaded to PyPI. ssm is included as a local folder in the external/ folder. in order to compile the cython, you need to cd external/ssm and run 'uv run python setup.py build_ext --inplace --force' in order to run the compilation workflow and generate the necessary cstats binary file for everything to compile.
