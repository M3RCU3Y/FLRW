# Generated Figures And Tables

This directory contains regenerated outputs from:

```bash
python scripts/generate_cosmology_figures.py
```

The PDF files are vector figures. The `.tex` files are generated tables used by
the manuscript and retained here as reproducible numerical outputs.

These files should not be edited by hand. Update the Python script, rerun the
generator, then commit the changed outputs. CI checks generated `.tex` tables by
exact diff and verifies that each PDF figure is rebuilt; exact PDF binary diffs
are intentionally avoided because renderer and font metadata can differ across
platforms.
