"""Общие определения метрик и оформление графиков для тетрадей проекта."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE, GOLD = '#285F82', '#B77622'


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'figure.figsize': (10, 5), 'figure.dpi': 120,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.18, 'axes.axisbelow': True})
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', lambda x: f'{x:,.2f}')


def metrics(df):
    positive = df.loc[df.quantity.gt(0)]
    return pd.Series({'net_units': int(df.quantity.sum()),
                      'gross_units': int(positive.quantity.sum()),
                      'returned_units': int(-df.loc[df.quantity.lt(0), 'quantity'].sum()),
                      'net_sales_usd': float(df.total_sales_usd.sum()),
                      'gross_sales_usd': float(positive.total_sales_usd.sum()),
                      'avg_vehicle_usd': float(positive.total_sales_usd.sum() / positive.quantity.sum())})


def breakdown(df, column):
    work = df.copy()
    work[column] = work[column].astype('string').fillna('Неизвестно')
    out = work.groupby(column, dropna=False).agg(units=('quantity', 'sum'), sales_usd=('total_sales_usd', 'sum'))
    out['share_pct'] = 100 * out.units / out.units.sum()
    return out.sort_values('units', ascending=False)


def compare(market, company, column):
    total = breakdown(market, column).rename(columns={'units': 'market_units', 'sales_usd': 'market_sales_usd', 'share_pct': 'market_mix_pct'})
    focal = breakdown(company, column).rename(columns={'units': 'mercur_units', 'sales_usd': 'mercur_sales_usd', 'share_pct': 'mercur_mix_pct'})
    out = total.join(focal, how='left').fillna(0)
    out['market_share_pct'] = 100 * out.mercur_units / out.market_units
    out['mix_gap_pp'] = out.mercur_mix_pct - out.market_mix_pct
    return out.sort_values('mercur_units', ascending=False)


def expanded_values(df, column):
    """Распределение по автомобилям, а не по строкам; возвраты исключены."""
    selected = df.loc[df.quantity.gt(0) & df[column].notna()]
    return np.repeat(selected[column].to_numpy(dtype=float), selected.quantity.to_numpy(dtype=int))


def save_figure(name):
    Path('figures').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path('figures') / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def bar(series, title, xlabel, name, color=BLUE):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.43 * len(series) + 1.5)))
    values = series.iloc[::-1]
    ax.barh(values.index.astype(str), values.astype(float), color=color)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(left=0, right=float(values.max()) * 1.18)
    for i, value in enumerate(values):
        label = f'{value:.2f}%' if '%' in xlabel else f'{value:,.0f}'.replace(',', ' ')
        ax.text(float(value) + float(values.max()) * .012, i, label, va='center', fontsize=10)
    save_figure(name)
