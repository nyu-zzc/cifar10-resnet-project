"""Execute notebook cells sequentially using IPython; no kernel server required.

Captures the model checks, recorded results, and plots in the notebook.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault('IPYTHONDIR', str(root / '.ipython'))
    path = root / 'notebooks/reproduction.ipynb'
    notebook = nbformat.read(path, as_version=4)
    InteractiveShell.instance()
    namespace = {'__name__': '__main__'}
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            exec(compile(cell.source, f'reproduction.ipynb:cell-{index}', 'exec'), namespace)
        cell.execution_count = count
        cell.outputs = []
        for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        print(f'Passed code cell {count}', flush=True)
    notebook.metadata['execution_note'] = 'Executed with scripts/execute_notebook.py using sequential IPython cells: CPU model checks, saved results, and training-log plots.'
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    result = {'status': 'passed', 'method': 'sequential_ipython_cell_execution', 'python': sys.version.split()[0],
              'code_cells_passed': count, 'training_started': False, 'competition_image_inference': False,
              'synthetic_cpu_forward_only': True, 'notebook_valid': True}
    (root / 'results/notebook_execution.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
