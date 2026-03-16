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


#### Installing, testing, uninstalling, ...

Everything here is tested in Ubuntu operating system!

I recommend installing `gravmag` as follows:

1. [Create a copy of the present repository at your computer via `git`]
	`git clone https://github.com/birocoles/gravmag.git`

Then, in the root directory `gravmag`, execute the following command lines:

2. [Create a conda environment] 
	`conda env create -f environment.yml`
3. [Activate de conda environment]
	`conda activate gravmag-env`
4. [Install `gravmag`] 
	`make install`
5. [Test the code]
	`make test`

