"""
run_notebook.py — Executes notebooks/02_eda_and_feature_engineering.ipynb
and populates all execution outputs, tables, and matplotlib chart images.
"""

import base64
import io
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd

nb_path = os.path.join("notebooks", "02_eda_and_feature_engineering.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# Execution namespace
execution_env = {
    "__name__": "__main__",
    "__file__": os.path.abspath(nb_path),
}

# Change directory to notebooks directory so relative paths work
orig_cwd = os.getcwd()
notebook_dir = os.path.abspath("notebooks")
os.chdir(notebook_dir)

execution_count = 1

for cell_idx, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue

    source_code = cell.source
    cell.outputs = []
    cell.execution_count = execution_count

    # Capture stdout
    old_stdout = sys.stdout
    captured_stdout = io.StringIO()
    sys.stdout = captured_stdout

    # Custom display function
    def custom_display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string())
        else:
            print(repr(obj))

    execution_env["display"] = custom_display

    # Intercept plt.show to capture figure
    captured_figures = []

    def custom_show():
        fig = plt.gcf()
        if fig and fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            captured_figures.append(img_b64)
            plt.close(fig)

    plt.show = custom_show
    execution_env["plt"] = plt

    try:
        exec(source_code, execution_env)
        stdout_text = captured_stdout.getvalue()
        if stdout_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type="stream",
                name="stdout",
                text=stdout_text
            ))

        for img_b64 in captured_figures:
            cell.outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": img_b64,
                    "text/plain": "<Figure size>"
                }
            ))

    except Exception as exc:
        sys.stdout = old_stdout
        os.chdir(orig_cwd)
        print(f"[ERROR in cell {cell_idx}]: {exc}", file=sys.stderr)
        raise
    finally:
        sys.stdout = old_stdout

    execution_count += 1

os.chdir(orig_cwd)

# Save executed notebook
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook executed successfully and saved to: {nb_path}")
