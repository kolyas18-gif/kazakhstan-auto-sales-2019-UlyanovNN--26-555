"""Независимые контрольные суммы через стандартный csv/Decimal и проверка тетрадей."""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import re
import nbformat

root = Path(__file__).resolve().parent
with (root / 'data/autokz2019_clean.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
internal = [r for r in rows if r['is_export'] == 'False']
mercur = [r for r in internal if r['company'] == 'Mercur Auto']
def count(records):
    return sum(int(r['quantity']) for r in records)
def revenue(records):
    return sum((Decimal(r['total_sales_usd']) for r in records), Decimal(0))
assert len(rows) == 16141 and len(rows[0]) == 20
assert count(rows) == 35165 and count(internal) == 34725 and count(mercur) == 394
assert abs(revenue(internal) - Decimal('801697865.966864')) < Decimal('.01')
assert abs(revenue(mercur) - Decimal('15121529.04097')) < Decimal('.01')
assert sum(int(r['source_records']) for r in rows) == 21262
assert sum(int(r['quantity']) for r in rows if int(r['quantity']) < 0) == -1
assert all(r['fuel_type'] in {'', 'F', 'D', 'E', 'HYB'} for r in rows)
assert not any(r['brand'] == 'Volkswagen' and r['sale_date'] >= '2019-06-01' for r in mercur)
summary = json.loads((root / 'data/cleaning_summary.json').read_text(encoding='utf-8'))
source_hash = hashlib.sha256((root / 'data/autokz2019_raw.csv').read_bytes()).hexdigest()
assert source_hash == summary['source_sha256']
notebooks = []
image_count = 0
for path in sorted(root.glob('0[1-4]_*.ipynb')):
    notebook = nbformat.read(path, 4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert [c.execution_count for c in code] == list(range(1, len(code)+1))
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    image_count += images
    notebooks.append({'file': path.name, 'executed_code_cells': len(code), 'embedded_png': images})
assert len(notebooks) == 4 and image_count == 16
report = (root / 'report.md').read_text(encoding='utf-8')
words = len(re.findall(r'\b[\w-]+\b', report))
assert words >= 350
result = {'status': 'passed', 'csv_rows': len(rows), 'csv_columns': len(rows[0]),
          'all_net_units': count(rows), 'domestic_net_units': count(internal),
          'mercur_net_units': count(mercur), 'domestic_sales_usd': str(revenue(internal)),
          'mercur_sales_usd': str(revenue(mercur)), 'notebooks': notebooks,
          'report_word_count': words, 'source_sha256': source_hash,
          'visual_qa': 'All 16 PNG charts inspected; Word report rendered to 5 pages and inspected. HTML browser preview unavailable: file URL policy.',
          'submission': 'Repository URL included in report; uploading files, public access check and LMS submission remain.'}
(root / 'verification_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
