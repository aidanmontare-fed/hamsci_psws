# hamsci_psws
Plotting data from Grape V1 receivers.

In terminal, clone repository and install with `pip install .`.

To begin, download the most recent dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6622112.svg)](https://doi.org/10.5281/zenodo.6622111) and unzip it into the `scripts/data/` directory. You can then run the Jupyter notebooks in the `scripts` directory to produce plots. 


## Installation


### With pip

The hamsci_psws package is distributed on the Python Package Index: https://pypi.org/project/hamsci_psws/

`pip install hamsci_psws`


## Getting Started


## Folder Structure

- `hamsci_psws` -  the python package
- `scripts` - example analysis scripts and Jupyter notebooks
- `deprecate`, `scripts/deprecate`, etc. - files that will be removed in future versions


## Contributing

Contributions to HamSCI projects are welcome.

### Releasing the Package

To upload a release, you will need an account at https://pypi.org/, and an API token for that account.

1. Make sure you have the latest version of [pip](https://pip.pypa.io/en/stable/):

`pip install --upgrade pip`

2. Make sure you have the latest version of [build](https://pypa-build.readthedocs.io/en/stable/index.html) and [twine](https://twine.readthedocs.io/en/latest/):

`pip install --upgrade build twine`

3. Build the package (from the root directory of the project):

`python -m build`

4. Upload to [PyPI](https://pypi.org/):

`python -m twine upload dist/*`

Enter `__token__` as the user name, and your token as the token, including the `pypi-` prefix.

5. Test package installation, preferrably in a separate environment from where you are developing.

`pip install hamsci_psws`


## Citing this Project
