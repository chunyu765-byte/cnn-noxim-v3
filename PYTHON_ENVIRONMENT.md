# Python Environment

This v2 directory intentionally does not copy the full conda environment from
`cnn-noxim`.

Reason:

1. The existing `ptq` conda environment is about 5 GB.
2. Conda environments usually contain absolute paths, so copying them into a
   new project directory is fragile.
3. Keeping the Python environment separate from the project files makes
   `cnn-noxim` and `cnn-noxim-v2` less likely to interfere with each other.

The original working environment was observed as:

```text
python==3.10.20
torch==2.11.0+cu130
numpy==2.2.6
Pillow==12.2.0
```

For later residual-network export scripts, the expected Python packages are:

```text
torch
numpy
Pillow
```

You can either reuse the existing local interpreter:

```bash
/home/camila/cnn-noxim/.conda_envs/ptq/bin/python your_export_script.py
```

or create an independent environment for v2 with the packages listed in
`requirements.txt`.
