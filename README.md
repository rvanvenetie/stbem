# An adaptive space-time boundary element method for the heat equation
This project provides an implementation of
an adaptive space-time boundary element for the heat equation. The
implementation is done in Python3 and uses multithreading.

## Run instructions
The main file is example.py, see also the run flags inside the source code.
For example, running our smooth example
on the unit square with anisotropic refinement, is done by:
```bash
python3 example.py --problem Smooth --domain UnitSquare --refinement anisotropic --theta 0.9
```

The tests can be run using pytest.
```bash
pytest
```

## License
This project is licensed under the terms of the [MIT license](LICENSE.md).
