"""Выполнение тетрадей в отдельных ядрах, сохранение вывода и HTML-копий."""
from pathlib import Path
import os
import sys
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parent
kernel_root = root / '.jupyter'
kernel = kernel_root / 'kernels' / 'coursework'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Coursework Python', 'language': 'python',
    'env': {'PYTHONPATH': os.environ.get('PYTHONPATH', '')}
}), encoding='utf-8')
template_root = root.parent / '.coursework_deps' / 'share' / 'jupyter'
os.environ['JUPYTER_PATH'] = os.pathsep.join([str(kernel_root), str(template_root)])
os.environ['JUPYTER_RUNTIME_DIR'] = str(kernel_root / 'runtime')
os.environ['IPYTHONDIR'] = str(kernel_root / 'ipython')
os.environ['MPLCONFIGDIR'] = str(kernel_root / 'matplotlib')
html_dir = root / 'notebook_html'
html_dir.mkdir(exist_ok=True)
paths = [root / name for name in sys.argv[1:]] if len(sys.argv) > 1 else sorted(root.glob('0[1-4]_*.ipynb'))
for path in paths:
    print('Executing', path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=180, kernel_name='coursework',
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    body, _ = HTMLExporter().from_notebook_node(nb)
    (html_dir / f'{path.stem}.html').write_text(body, encoding='utf-8')
    print('Saved', path.name, flush=True)
