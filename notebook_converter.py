from nbconvert import PythonExporter
import nbformat
import re

notebooks = ['project', 'analysis']
for ntb in notebooks:
    with open(f"{ntb}.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
    nb.cells = [cell for cell in nb.cells if cell.cell_type != 'markdown']

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    source = re.sub(r'\n{2,}\Z', '\n', source)

    with open(f"{ntb}.py", 'w') as f:
        f.write(source)
print("done")