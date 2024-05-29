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

The recomendation here is using the `Makefile` to install, test and also uninstall the `gravmag`.
[`Ubuntu`](https://ubuntu.com/) users may execute the steps below:
1. Create a copy of the present repository at your computer. 
2. Open the terminal in the root directory `gravmag` at your computer and execute the command `make install`.
3. Also at the root directory `gravmag`, execute `make test` in terminal to test the code. The command 
`make report` creates the file `index.html` at `gravmag/gravmag/tests/htmlcov`. Open this file in your preferred 
browser to see a report of the tests.
4. To uninstall **`gravmag`**, execute the command `make uninstall` in the root directory `gravmag`.