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

To select Python interpreter, activate Command Palette (`Cmd-Shift-P` or `View -> Command Palette`) and select `Python: Select Interpreter`. You should see a virtual environment you created in previous step.

### Linters

We use `pylint` (PEP8) and `flake8`. To use multiple linters, it is easier to edit `settings.json` file directly. 

On linux, the file is located at `$HOME/.config/Code/User/settings.json`, for other systems find the location [here](https://code.visualstudio.com/docs/getstarted/settings#_settings-file-locations).

Add the following block at the beginning

```json
{
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=150",
        "--ignore=E402,F841,F401,E302,E305"
    ],
    ...
```

If `pylint` and `flake8` were not yet installed in your Python  environment, after restarting VS Code and opening any Python file, you should see the following dialog in the bottom right corner:

![Linters Missing](/assets/img/Screenshot_missing_linters.png)

After clicking on Install buttons, linters will be automatically installed. To check that you have multiple linters selected, go to Command Palette (`Cmd-Shift-P` or `View -> Command Palette`), select `Python: Select Linter` and you should get `current: multiple selected` as below

![Linters Selection](/assets/img/Screenshot_linters_selection.png)