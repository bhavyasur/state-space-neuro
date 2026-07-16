# rSLDS Modeling Pipeline: NanoNeuroTechnology lab

**Author**: Bhavya Surapaneni

This GitHub repository is for the NanoNeuroTechnology lab at Purdue University, in the Department of Biomedical Engineering. This pipeline is specifically to run datasets through an rSLDS pipeline that  runs the model and produces a number of useful plots based on the data structure and type. The pipeline is meant to be easily adaptable to work with various DataTypes from different people/experiments. This README.md will describe how to use the pipeline and how to tailor it to your own needs.

Requirements are stored in the pyproject.toml.

There are various parts of this repository that are not strictly necessary to run a rSLDS modeling of your data. This code is stored in archive. There is code there for dimensionality reduction methods PCA, jPCA, and GPFA. Much of this is specific to a Lorenz Attractor and spiking data rather than calcium imaging data, but can be easily tailored for certain needs. Still, most of this code can be ignored unless these specific methods are of interest.

For rSLDS, all source code is located in the 'src' folder. The 'src' folder is structured into datatype folders and a 'rslds' folder that contains the run pipeline and any utilities and helper functions that are general to the pipeline as a whole. Other folders in the 'src' folder are datatype specific folders, where there is a utilities file contained loading functions, plotting functions, and any other helper functions that are specific to a certain datatype/ animal type / experiment type. All function definitions for loading are written in each datatype's utils folder. Then, the loading functions are called DIRECTLY in the pipeline function, run_rslds_pipeline, located within rSLDS.py in the rslds folder. The pipeline is called individually for each datatype in the __datatype___rslds.py function within each datatype folder.







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