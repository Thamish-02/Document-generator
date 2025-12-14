## AI Summary

A file named packaging.py.


### Function: is_conda_environment(func)

### Function: _get_conda_like_executable(command)

**Description:** Find the path to the given executable

Parameters
----------

executable: string
    Value should be: conda, mamba or micromamba

## Class: PackagingMagics

**Description:** Magics related to packaging & installation

### Function: wrapper()

**Description:** Return True if the current Python executable is in a conda env

### Function: pip(self, line)

**Description:** Run the pip package manager within the current kernel.

Usage:
  %pip install [pkgs]

### Function: _run_command(self, cmd, line)

### Function: conda(self, line)

**Description:** Run the conda package manager within the current kernel.

Usage:
  %conda install [pkgs]

### Function: mamba(self, line)

**Description:** Run the mamba package manager within the current kernel.

Usage:
  %mamba install [pkgs]

### Function: micromamba(self, line)

**Description:** Run the conda package manager within the current kernel.

Usage:
  %micromamba install [pkgs]

### Function: uv(self, line)

**Description:** Run the uv package manager within the current kernel.

Usage:
  %uv pip install [pkgs]
