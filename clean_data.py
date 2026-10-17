"""Воспроизводимая очистка autokz2019.csv. Python >=3.10, pandas >=2.2, numpy.

Запуск: python clean_data.py --input data/autokz2019_raw.csv --output data
Оригинал не изменяется. Подробные решения и ограничения — в тетрадях и README.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

RENAME = {
    'Год': 'year', 'Месяц': 'month', 'Компания': 'company', 'Бренд': 'brand',
    'Модель': 'model', 'Модификация': 'modification', 'Год выпуска': 'manufacture_year',
    'Страна-производитель': 'country', 'Вид топлива': 'fuel_type',
    'Объём двиг, л,': 'engine_l', 'Коробка передач': 'transmission',
    'Тип привода': 'drive_type', 'Сегмент': 'segment_old', 'Регион': 'region',
    'Наименование дилерского центра': 'dealer', 'Тип клиента': 'client_type',
    'Форма расчета': 'payment_type', 'Количество': 'quantity', 'Цена, USD': 'unit_price_usd',
    'Продажа, USD': 'total_sales_usd', 'Область': 'area',
    'Сегментация 2013': 'segment_2013', 'Класс 2013': 'class_2013',
    'Сегментация Eng': 'segment_eng', 'Локализация производства': 'localization',
}
DROP = ['dealer', 'payment_type', 'segment_old', 'segment_eng', 'localization',
        'client_type', 'modification']
CATEGORIES = ['fuel_type', 'transmission', 'drive_type', 'segment_2013', 'class_2013']
COUNTRIES = dict(zip(
    ['Республика Казахстан', 'Российская Федерация', 'Япония', 'Турция', 'Таиланд',
     'США', 'Германия', 'UK', 'Китай', 'Узбекистан', 'Австрия', 'Швеция', 'Корея',
     'Нидерланды', 'Испания', 'Польша', 'Бельгия', 'Венгрия', 'Белоруссия'],
    ['KAZ', 'RUS', 'JPN', 'TUR', 'THA', 'USA', 'DEU', 'GBR', 'CHN', 'UZB', 'AUT',
     'SWE', 'KOR', 'NLD', 'ESP', 'POL', 'BEL', 'HUN', 'BLR']))
COMPANIES = {
    'Mercur Autos': 'Mercur Auto', 'Меркур Авто': 'Mercur Auto',
    'Каспиан Моторс': 'Caspian Motors', 'Автокапитал': 'Autokapital',
    'ММС Рус': 'MMC RUS', 'Равон Моторс Казахстан': 'Ravon Motors Kazakstan',
    'Хино Моторс Казахстан': 'Hino Motors',
}
MONTHS = dict(zip(['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                   'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'], range(1, 13)))


def load_raw(path):
    """Разделитель и десятичный знак заданы явно; неожиданные поля не игнорируются."""
    raw = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig', low_memory=False)
    if set(raw.columns) != set(RENAME):
        raise ValueError(f'Изменилась схема: {set(raw.columns) ^ set(RENAME)}')
    return raw


def numeric(series):
    return pd.to_numeric(series.astype('string').str.replace(r'\s+', '', regex=True)
                         .str.replace(',', '.', regex=False), errors='coerce')


def transmission_code(value):
    if pd.isna(value):
        return pd.NA
    s = re.sub(r'\s+', '', str(value).upper()).translate(str.maketrans('АМТ', 'AMT'))
    if any(t in s for t in ['DSG', 'DCT', 'PDK', 'TRONIC', 'CVT', 'AMT',
                            'POWERSHIFT', 'АКП', 'AКП', 'РЕДУКТОР', 'AT', 'A/T']):
        return 'Автомат'
    if re.search(r'^\d*A$', s):
        return 'Автомат'
    if any(t in s for t in ['MT', 'M/T', 'МКП', 'MКП', 'МЕХ', 'MЕХ']):
        return 'Механика'
    if re.search(r'^\d+M$', s):
        return 'Механика'
    return pd.NA


def drive_code(value):
    if pd.isna(value):
        return pd.NA
    s = re.sub(r'\s+', '', str(value).lower()).replace('х', 'x')
    if s in {'4wd', 'awd', 'полный', 'полныйпривод', 'quattro', '4motion', '4x4'}:
        return '4WD'
    if s in {'fwd', 'ff', 'передний', 'передний(ff)'}:
        return 'FWD'
    if s in {'rwd', 'задний'}:
        return 'RWD'
    if s in {'2wd', '4x2', '4x2.2'}:
        return '2WD'  # Ведущая ось не установлена: не подменяем передним приводом.
    return pd.NA


def normalize(raw):
    """Очистка на уровне исходных записей с журналом существенных исправлений."""
    empty = raw.isna().all(axis=1)
    duplicates = raw.duplicated() & ~empty
    df = raw.loc[~empty & ~duplicates].rename(columns=RENAME).copy()
    df['source_row'] = df.index + 2  # Номер строки CSV с учётом заголовка.
    changes = []

    def replace(mask, column, value, reason):
        mask = mask.fillna(False)
        if not mask.any():
            return
        values = value.reindex(df.index) if isinstance(value, pd.Series) else pd.Series(value, index=df.index)
        for idx in df.index[mask]:
            changes.append({'source_row': int(df.at[idx, 'source_row']), 'field': column,
                            'before': str(df.at[idx, column]), 'after': str(values.at[idx]),
                            'reason': reason})
        df.loc[mask, column] = values.loc[mask]

    for c in df.columns:
        if c != 'source_row' and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object):
            df[c] = df[c].astype('string').str.strip().replace({'': pd.NA, '#Н/Д': pd.NA})

    df['company'] = df['company'].replace(COMPANIES)
    unknown_countries = set(df['country'].dropna()) - set(COUNTRIES)
    if unknown_countries:
        raise ValueError(f'Новые страны требуют проверки: {unknown_countries}')
    df['country'] = df['country'].map(COUNTRIES)

    # Сдвиг Renault подтверждается одновременно числом в топливе и AT/MT/0 в объёме.
    shifted = (df['brand'].eq('Renault') & df['fuel_type'].isin(['2', '1,6'])
               & df['engine_l'].isin(['AT', 'MT', '0']))
    old_fuel, old_engine = df['fuel_type'].copy(), df['engine_l'].copy()
    replace(shifted, 'transmission', old_engine.replace('0', pd.NA), 'Восстановление сдвига полей Renault')
    replace(shifted, 'engine_l', old_fuel, 'Объём восстановлен из поля топлива Renault')
    replace(shifted, 'fuel_type', pd.NA, 'Исходный тип топлива при сдвиге утрачен; не угадываем')

    hybrid = df['modification'].str.contains(r'e-hybrid|\bhybr\b', case=False, na=False)
    replace(hybrid, 'fuel_type', 'гибрид', 'Явное обозначение гибрида в модификации')
    ipace = df['brand'].eq('Jaguar') & df['model'].str.casefold().eq('i-pace')
    replace(ipace & df['fuel_type'].isna(), 'fuel_type', 'Электро',
            'Модель Jaguar I-Pace: остальные строки явно обозначены как электрокар')
    fuel_raw = df['fuel_type'].copy()
    df['fuel_type'] = df['fuel_type'].str.casefold().map(
        {'бензин': 'F', 'дизель': 'D', 'электро': 'E', 'электричество': 'E', 'гибрид': 'HYB'})

    engine_original = df['engine_l'].copy()
    # Извлечение только из допустимого формата объёма, а не любых первых цифр.
    engine = df['engine_l'].str.upper().str.replace(',', '.', regex=False)
    valid_engine = engine.str.match(r'^\d+(?:\.\d+)?\.?\s*(?:L\.?|MPI|GDI|G|T|T-GDI|TURBO|CRDI(?: VGT| WGT)?|H)?$', na=False)
    df['engine_l'] = pd.to_numeric(engine.where(valid_engine).str.extract(r'^(\d+(?:\.\d+)?)')[0], errors='coerce')
    niva = df['brand'].eq('Chevrolet') & df['model'].eq('Niva')
    replace(niva & df['engine_l'].notna() & df['engine_l'].ne(1.7), 'engine_l', 1.7,
            'Chevrolet Niva: последовательная ошибка 2.7–26.7; в остальных строках модели 1.7')
    electric = df['fuel_type'].eq('E')
    replace(electric, 'engine_l', 0.0,
            'Электромобиль: рабочий объём ДВС отсутствует; 400 л.с. и 88 KWH не являются литрами')
    df.loc[df['engine_l'].eq(0) & ~electric.fillna(False), 'engine_l'] = np.nan
    df['engine_l'] = df['engine_l'].astype('Float64')

    gear_original = df['transmission'].copy()
    drive_original = df['drive_type'].copy()
    df['transmission'] = df['transmission'].map(transmission_code).astype('string')
    cayenne8 = df['brand'].eq('Porsche') & df['model'].eq('Cayenne') & gear_original.eq('8')
    reference = df.loc[df['brand'].eq('Porsche') & df['model'].eq('Cayenne'), 'transmission'].dropna()
    if len(reference) >= 2 and set(reference) == {'Автомат'}:
        replace(cayenne8, 'transmission', 'Автомат', 'Число 8 — число ступеней; все распознанные Cayenne имеют автомат')
    replace(electric & df['transmission'].isna(), 'transmission', 'Автомат',
            'Электропривод отнесён к автомату в требуемой двухкатегорийной схеме')
    df['drive_type'] = df['drive_type'].map(drive_code).astype('string')
    for c in ['region', 'area']:
        df[c] = df[c].str.title()

    for c in ['quantity', 'unit_price_usd', 'total_sales_usd', 'manufacture_year', 'year']:
        df[c] = numeric(df[c])
    df['manufacture_year'] = df['manufacture_year'].astype('Int64')
    date_parts = pd.DataFrame({'year': df['year'], 'month': df['month'].map(MONTHS), 'day': 1})
    df['sale_date'] = pd.to_datetime(date_parts, errors='raise') + pd.offsets.MonthEnd(0)
    if df['sale_date'].isna().any():
        raise ValueError('Не распознаны даты продаж')

    invalid = (df['quantity'].isna() | df['unit_price_usd'].isna() | df['total_sales_usd'].isna()
               | df['quantity'].mod(1).ne(0) | df['unit_price_usd'].le(0))
    quarantine = df.loc[invalid].copy()
    quarantine['reason'] = 'Неизвестное количество или некорректное числовое поле; исключено из KPI'
    invalid_tokens = []
    for col, old in [('fuel_type', fuel_raw), ('engine_l', engine_original),
                     ('transmission', gear_original), ('drive_type', drive_original)]:
        for token, n in old.loc[old.notna() & df[col].isna()].value_counts().items():
            invalid_tokens.append({'field': col, 'unresolved_token': token, 'rows': int(n)})
    df = df.loc[~invalid].copy()
    df['quantity'] = df['quantity'].astype('Int64')
    df['is_return'] = df['quantity'].lt(0)
    df['is_export'] = df['region'].eq('Экспорт')
    df['source_records'] = 1
    for c in CATEGORIES:
        df[c] = df[c].astype('category')
    changes_df = pd.DataFrame(changes, columns=['source_row', 'field', 'before', 'after', 'reason'])
    meta = {'input_rows': len(raw), 'empty_rows': int(empty.sum()),
            'exact_duplicates_removed': int(duplicates.sum()), 'rows_after_exact_dedup': int((~empty & ~duplicates).sum()),
            'quarantine_rows': len(quarantine), 'valid_source_rows': len(df),
            'renault_shift_rows': int(shifted.sum()), 'niva_corrected_rows': int((niva & engine_original.ne('1,7')).sum()),
            'source_quantity_raw': float(numeric(raw['Количество']).sum()),
            'quantity_after_cleaning': int(df['quantity'].sum()),
            'sales_after_cleaning_usd': float(df['total_sales_usd'].sum())}
    return df, quarantine, changes_df, pd.DataFrame(invalid_tokens), meta


def aggregate_clean(df):
    """Сохраняем массу продаж после удаления атрибутов и нормализации.

    Зерно: месяц × компания × характеристики автомобиля × география × цена × возврат.
    Совпадение на этом уровне не означает дубликат исходной записи.
    """
    work = df.drop(columns=DROP + ['year', 'month', 'source_row']).copy()
    measures = ['quantity', 'total_sales_usd', 'source_records']
    keys = [c for c in work if c not in measures]
    clean = work.groupby(keys, dropna=False, observed=True, as_index=False, sort=False)[measures].sum()
    clean['quantity'] = clean['quantity'].astype('Int64')
    clean['manufacture_year'] = clean['manufacture_year'].astype('Int64')
    clean['source_records'] = clean['source_records'].astype('int64')
    clean = clean.sort_values(['sale_date', 'company', 'brand', 'model'], kind='stable').reset_index(drop=True)
    validate(clean)
    assert clean['quantity'].sum() == df['quantity'].sum()
    assert np.isclose(clean['total_sales_usd'].sum(), df['total_sales_usd'].sum(), atol=0.01)
    assert clean['source_records'].sum() == len(df)
    return clean


def validate(df):
    assert not df.duplicated().any(), 'Остались точные дубликаты'
    assert df['sale_date'].dt.is_month_end.all()
    assert df['sale_date'].notna().all()
    assert df['quantity'].notna().all() and df['quantity'].mod(1).eq(0).all()
    assert df['unit_price_usd'].gt(0).all()
    assert df['country'].str.fullmatch('[A-Z]{3}').all()
    assert set(df['fuel_type'].dropna()) <= {'F', 'D', 'E', 'HYB'}
    assert set(df['transmission'].dropna()) <= {'Автомат', 'Механика'}
    assert set(df['drive_type'].dropna()) <= {'4WD', '2WD', 'FWD', 'RWD'}
    assert df['engine_l'].dropna().between(0, 15).all()
    assert df.loc[df['fuel_type'].eq('E'), 'engine_l'].eq(0).all()
    assert df['manufacture_year'].dropna().between(1900, df['sale_date'].dt.year.max()).all()
    assert df['is_return'].eq(df['quantity'].lt(0)).all()
    assert df.loc[df['quantity'].lt(0), 'total_sales_usd'].lt(0).all()
    expected = df['unit_price_usd'] * df['quantity']
    assert np.isclose(df['total_sales_usd'].astype(float), expected.astype(float), atol=0.02, rtol=1e-7).all()
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in CATEGORIES)


def read_clean(path):
    df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['sale_date'],
                     dtype={'quantity': 'Int64', 'manufacture_year': 'Int64', 'engine_l': 'Float64'})
    for c in CATEGORIES:
        df[c] = df[c].astype('category')
    validate(df)
    return df


def run(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(input_path)
    detailed, quarantine, corrections, unresolved, meta = normalize(raw)
    clean = aggregate_clean(detailed)
    meta.update({'output_rows': len(clean), 'output_columns': len(clean.columns),
                 'source_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest()})
    for name, table in [('autokz2019_clean', clean), ('quarantine', quarantine),
                        ('corrections', corrections), ('unresolved_tokens', unresolved),
                        ('clean_source_records', detailed)]:
        table.to_csv(output_dir / f'{name}.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    schema = pd.DataFrame({'column': clean.columns, 'dtype': [str(clean[c].dtype) for c in clean],
                           'missing_rows': clean.isna().sum().values})
    schema.to_csv(output_dir / 'schema.csv', index=False, encoding='utf-8-sig')
    (output_dir / 'cleaning_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    reread = read_clean(output_dir / 'autokz2019_clean.csv')
    assert len(reread) == len(clean) and reread['quantity'].sum() == clean['quantity'].sum()
    return clean, detailed, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'data/autokz2019_raw.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    _, _, result = run(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
