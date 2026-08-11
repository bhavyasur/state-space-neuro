# rSLDS Modeling Pipeline: NanoNeuroTechnology lab

**Author**: Bhavya Surapaneni

### Overview

This GitHub repository is for the NanoNeuroTechnology lab at Purdue University, in the Department of Biomedical Engineering. This pipeline is specifically to run datasets through an rSLDS pipeline that  runs the model and produces a number of useful plots based on the data structure and type. The pipeline is meant to be easily adaptable to work with various DataTypes from different people/experiments. This README.md will describe how to use the pipeline and how to tailor it to your own needs.

A document describing the intricacies of the model applications for the NNT lab is linked here within the repository: [rSLDS description](rSLDS_description.pdf).

### Setup

Requirements are stored in the `pyproject.toml`, as well as `uv.lock`. I highly recommend using a uv virtual environment to deal with requirements, as the repository is already configured to work well with uv. Please make sure the Python version you work with is **3.10.** Certain dependencies fail at higher versions, so it's critical to manually set the version to 3.10 when setting up your uv venv.

Note that if you are working on one of the lab computers, you may have to update the `settings.json`file to your Purdue username so uv works correctly:

```JSON
"terminal.integrated.env.windows": {
    "PATH": "C:\\Users\\__YOUR PURDUE USERNAME__\\.local\\bin"
},
```

### Repository Structure

There are various parts of this repository that are not strictly necessary to run a rSLDS modeling of your data. This code is stored in archive. There is code there for dimensionality reduction methods PCA, jPCA, and GPFA. Much of this is specific to a Lorenz Attractor and spiking data rather than calcium imaging data, but can be easily tailored for certain needs. Still, most of this code can be ignored unless these specific methods are of interest.

Data should be stored in a folder titled `data`, though this is easily reconfigurable based on the file paths you call for `run_rslds_pipeline`within the datatype-specific run files. As per the current paradigm, output plots are stored in a folder called`output`at the root of the repository. The path that plots are stored to is automatically generated based on the `plot_key`and other parameters you pass to `run_rslds_pipeline`.

For rSLDS, all source code is located in the `src` folder. The `src` folder is structured into datatype-specific folders and a 'rslds' folder that contains the run pipeline and any utilities and helper functions that are general to the pipeline as a whole. Other folders in the 'src' folder are datatype specific folders, where there is a utilities file contained loading functions, plotting functions, and any other helper functions that are specific to a certain datatype/ animal type / experiment type.

Each datatype-specific folder has, at minimum, a `_load_util.py` file and a `_rslds.py`file. All function definitions for loading are written in that datatype's `_load_util.py`. Then, those loading functions are called DIRECTLY in the main pipeline function, `run_rslds_pipeline`, which is located within `rSLDS.py` in the `src/rslds` folder. This pipeline is called and executed within each datatype's `_rslds.py`file - this makes it easy to only deal with the datatype you are concerned with.

`run_rslds_pipeline`, in the file `src/rslds/rSLDS.py`, is a very lengthy function. The reason is that it contains conditionals for every datatype so you can run the pipeline for any datatype and it accommodates the specific loading and any other nuances with plotting, model fitting, etc. All data loading is done within this function, also within conditionals based on your datatype.

Within `src/rslds` there is also `rslds_util.py`, which contains helper functions that can be applied regardless of the datatype you are working with. Primarily, these are plotting functions and necessary helpers for complex plots.

### Best Approaches

The goal of this repository is to be highly versatile and reconfigurable. To approach working with your data, first check heck if one of the existing datatypes is similar to the data you are working with. Here is a breakdown of existing datatypes:

| folder path          | `src/dual_shank`                                                                                                                                                                                                                                                                                                                                                             | `src/gcamp8`                                                                                                                                                        | `src/l23`                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `src/m2`                                                                                                                                                                                                         | `src/rbp_cre`                                                                                       |
| -------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| datatype description | electrophysiology spiking data (from Om). thus uses Poisson Orthogonal spiking emissions within the rSLDS model.                                                                                                                                                                                                                                                              | 2P calcium imaging data (from Shulan).                                                                                                                                | both normal 2P calcium imaging data and Bessel data (from Shivam).                                                                                                                                                                                                                                                                                                                                                                                                          | Freely-moving Mini2P calcium imaging data. Gaussian Orthogonal emissions. Inconsistent trial structure, so lots of plots are written differently for this function t                                               | 2P calcium imaging data (from Shulan). Gaussian Orthogonal emissions. Consistent trial structure.     |
| other useful notes   | this is the only datatype that is not fully clean, it does not have a`_rslds.py`file because it was used purely as a test. this is probably not the most useful datatype to pull from. HOWEVER, note that if you are working with EPhys data, your `emissions` parameter in the model call inside `run_rslds_pipeline`should be set to `emissions = "poisson_orthog"`. | We tried a lot of variations of cleaning this data since we were getting strange results, so there are some strange parameters for certain loading types and testing. | This is the datatype that is most complex and has been worked on the most - this means there are a lot of functions and code you can take and reuse, but some of the conditionals inside`run_rslds_pipeline`are very complicated (sometimes unnecessarily so). There are also some functions that only work and have been tested on this datatype, such as `state_probablity_plot`in `rslds_util.py`, and you can take these functions and adapt them for your needs. | If you are doing anything with**freely-moving data**, start with this folder. Your loading will likely change based on the structure of your .mat files, but a lot of the general functions can be reused.  | This is different from the GCamP8 since the structure was slightly different (this one included ROI). |

If you are making a new datatype folder for your data, here are some important steps you need to take so the pipeline runs smoothly:

1) Around the top of `src/rslds/rSLDS.py`, there is a class declaration:

```python
class DataType(Enum):
  DualShank = "dualshank"
  RbpCre = "rbpcre"
  L23 = "l23"
  GCaMP8 = "gcamp8"
  M2 = "m2"
```

Whatever your datatype is, add it to the class declaration. This will allow you to set the `type`parameter when you call `run_rslds_pipeline`, since the datatype for `type`is the DataType Enum. For example, if you created a datatype and wanted to refer to it as "monkey", you would update DataType as such:

```python
class DataType(Enum):
  DualShank = "dualshank"
  RbpCre = "rbpcre"
  L23 = "l23"
  GCaMP8 = "gcamp8"
  M2 = "m2"
  Monkey = "monkey"
```

Then, in your run_rslds_pipeline calls, you would set `type=DataType.Monkey`. This ensures that the pipeline will use all the conditionals for your datatype instead of a different datatype, which would load data differently. Additionally, in all of your datatype-specific `_rslds.py` run files, you will need to import the `DataType`enum so you can use it within those pipeline function calls.

2) Write your loading functions for your specific data.
   * The best way to structure this is to load the data in whatever form it's located in within MatLab (whether it's a list of trials per neuron, or just a full continuous time series). Then, if necessary, write a function to manipulate the data into a time-series, write a slicing function to get GO and NOGO trials only, etc. You will likely need to bin your data, so make sure one of the loading functions does this.
   * You will also need some sort of data visualization sanity check plot, and possibly a most likely state plot that uses the zhat_lem. There is a `most_likely_state_plot`plotting function within `src/rslds/rslds_util.py`that works for most datatypes that have a **consistent trial structure** - that is, all the trials are the exact same number of frames. That function works for `l23`, `rbp_cre`, and `gcamp8`. If you do not have a consistent trial structure, such as in the `m2`case, I would write the code for a datatype-specific most likely state function within that datatype's `_load_util.py`file, and then call it in the pipeline conditional. The logic for dealing with the changes in `zhat_lem`and plot specifics can be easily copied from `most_likely_state_plot`.
3) Add your specific conditionals to the `run_rslds_pipeline`function. You can figure out the general structure of when you need a conditional just from reading through `run_rslds_pipeline`. The main conditionals are for loading the data, training the model (different emissions types based on EPhys vs Calcium), and for certain plots. The plots with major conditionals are the **trace sanity check** and **most likely state.** There are a couple other plots that may change based on your datatype. You can also always alter the structure of `run_rslds_pipeline`to make it as easy as possible for you.
4) Save all your changes and run the model from your datatype's `_rslds.py` run file within your datatype-specific folder. Make sure to set the `plot_key`so the save path is automatically set based on the parameters you pass to the pipeline function call.

Overall, use the existing structure of the repository and datatypes to add more datatypes and code. Happy modeling!

### Contact Information

If you have any questions about this specific repository or how best to work with your specific datatype, you can reach Bhavya Surapaneni at [bhavyasp@engineering.upenn.edu](mailto:bhavyasp@engineering.upenn.edu) or [720-366-1828](tel:7203661828).
