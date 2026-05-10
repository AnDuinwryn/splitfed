# Zipimport Tutorial

This folder contains an interactive notebook that explains why Python can import
modules from a `.zip` file, and how that relates to normal imports from the
project root.

The tutorial is self-contained. It creates temporary modules and zip archives
under a temporary directory while the notebook runs.

## Run On Ubuntu

From the repository root:

```bash
uv run --with jupyter --no-sync jupyter lab zipimport_tutorial/zipimport_step_by_step.ipynb
```

If Jupyter is already installed in the active environment:

```bash
jupyter lab zipimport_tutorial/zipimport_step_by_step.ipynb
```

You can also use classic notebook:

```bash
jupyter notebook zipimport_tutorial/zipimport_step_by_step.ipynb
```

Run the notebook from top to bottom. Each section builds on variables created by
earlier cells.
