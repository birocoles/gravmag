# GravMag

A simple Python Package for processing and interpreting gravity and magnetic
data in Geophysics. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8284770.svg)](https://doi.org/10.5281/zenodo.8284770)


#### General description

The idea is to use mostly standard Python libraries like [`numpy`](https://numpy.org/) and [`scipy`](https://scipy.org/) 
for building everything, or at least most of it. 
We don't focus too much on plotting, but if we need to, we usually rely on the standard 
[`matplolib`](https://matplotlib.org/) library.

Sticking to these standard libraries makes things easier when it comes to installing and updating the code.

We might use **`gravmag`** for teaching classes, courses, and doing research. 
The aim is to create a foundation that can be possibly adapted to handle big data and super tricky problems.

By 'reinventing the wheel,' we're basically learning how to create new stuff from scratch.

All references cited in the code and examples are shown in the file `references.md`.


#### Project Structure

gravmag/
├── Makefile
├── pyproject.toml
├── setup.cfg
├── README.md
├── src/
│ └── gravmag/
│ ├── init.py
│ └── core.py
|── tests/
├── examples/
└── docs/
├── init.py
└── test_core.py


#### Installing, testing, uninstalling, ...

Create a copy of the present repository at your computer. 
The recomendation here is using the `Makefile` to install, test and also uninstall the `gravmag`.
In the project root directory `gravmag/`, [`Ubuntu`](https://ubuntu.com/) users may execute 
the following commands in terminal:

- **Install editable:** `make install`
- **Run tests:** `make test`
- **Open coverage report:** `make report`
- **Format code:** `make style`
- **Clean artifacts:** `make clean`
- **Unistall**: `make uninstall`