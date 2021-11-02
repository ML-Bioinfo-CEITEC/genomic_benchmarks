# README for developers of this package

We currently develop this package using Python 3.8.9, venv, pylint, flake8, pytest, pytest-cov, Black, isort, and other tools. This document should guide you to install and set up the devel environment on a new machine. It is not set in stone and is likely to evolve with the package.

## Basics

Install Python 3.8.9, create a new virtual environment and activate it. Clone this repository and do editable `pip` installation. Finally, install TensorFlow and PyTorch.

```bash
   git clone git@github.com:ML-Bioinfo-CEITEC/genomic_benchmarks.git
   cd genomic_benchmarks
   pip install --editable .

   pip install tensorflow>=2.6.0
   pip install typing-extensions --upgrade  # fixing TF installation issue

   pip install torch>=1.10.0
   pip install torchtext
```

## VS Code Settings

### Python

Install VS Code [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python): use Ctrl-Shift-x to open MarketPlace, search for Python Extension, and install it.

To select Python interpreter, activate Command Palette (`Cmd-Shift-P` or `View -> Command Palette`) and select `Python: Select Interpreter`. You should see a virtual environment you created in previous step.

### Linters

We use `pylint` (PEP8) and `flake8`. VS Code setting is provided in [settings.json](.vscode/settings.json) file. 

If `pylint` and `flake8` were not yet installed in your Python  environment, after restarting VS Code and opening any Python file, you should see the following dialog in the bottom right corner:

![Linters Missing](/assets/img/Screenshot_missing_linters.png)

After clicking on Install buttons, linters will be automatically installed. To check that you have multiple linters selected, go to Command Palette (`Cmd-Shift-P` or `View -> Command Palette`), select `Python: Select Linter` and you should get `current: multiple selected` as below

![Linters Selection](/assets/img/Screenshot_linters_selection.png)

### Formatters

We use `black` and `isort` to auto-format on `Save`. VS Code setting is provided in [settings.json](.vscode/settings.json) file. (You will need to install those two packages.)

### Docstrings

We use [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) extension to help us with writing useful and well formated documentation. Install this extension from VS Code Extensions Marketplace (Ctrl-Shift-x). VS Code settings are already provided in [settings.json](.vscode/settings.json) file. For now, we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring format. Keyboard shortcut: Ctrl+Shift+2.