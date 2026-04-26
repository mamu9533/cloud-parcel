This repo uses pyrcel as a backend model to compute microphysical and optical properties of a low cloud. Because pyrcel does not account for entrainment or coalescence, this best functions as a toy package / proof of concept, although you can still limit cloud thickness to emulate a real low cloud more closely.

Note: the monte_carlo module functions well on its own for 2D simulations and can be used as such.

For a detailed example of how to use the package, see "example.ipynb"

REQUIREMENTS FOR INSTALL

Assimulo must be installed first via conda or mamba for correct installation of pyrcel, then pip install -e . should work.

Python version must be >= 3.8. 

pyrcel, joblib also required