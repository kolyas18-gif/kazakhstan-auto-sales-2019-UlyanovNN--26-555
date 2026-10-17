"""Сборка учебных тетрадей с читаемыми ячейками; выполнение — execute_notebooks.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    nb.cells = [nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text)
                for kind, text in cells]
    nbf.validate(nb)
    nbf.write(nb, ROOT / name)


SETUP = '''from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display

# Тетради можно открывать из папки проекта или из coursework.
ROOT = Path.cwd() if Path('clean_data.py').exists() else Path.cwd() / 'coursework'
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from clean_data import load_raw, normalize, aggregate_clean, read_clean, run, validate, CATEGORIES, DROP, COMPANIES, MONTHS
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
'''

ANALYSIS = SETUP + '''
import matplotlib.pyplot as plt
from analysis_utils import setup, metrics, breakdown, compare, expanded_values, save_figure, bar, BLUE, GOLD
setup()
df = read_clean('data/autokz2019_clean.csv')
market = df.loc[~df.is_export].copy()
mercur = market.loc[market.company.eq('Mercur Auto')].copy()
positive = market.loc[market.quantity.gt(0)].copy()
Path('tables').mkdir(exist_ok=True)
'''

notebook('01_initial_review.ipynb', [
('md', '''# 1. Задача и первичная диагностика

**Контекст.** Аналитик автодилерского холдинга готовит данные о продажах автомобилей в Казахстане за январь–сентябрь 2019 года для руководства «Меркур Авто».

**Техническая цель:** получить очищенный датасет и воспроизводимый скрипт обработки. **Аналитическая цель:** описать рынок, оценить долю и структуру продаж «Меркур Авто», определить обоснованные направления дальнейшей проверки и роста.

Источник — предоставленный файл `data/autokz2019_raw.csv`, копия исходника без изменений. Ссылка задания: https://code.s3.yandex.net/high-education_higher-data-analytics/autokz2019.csv . Интернет-версия не подменяет пользовательский файл.

**Ключевые ограничения:** нет уникального идентификатора сделки, данных о марже, расходах, запасах, лидах и целевых KPI. Количество в строке может превышать единицу. «Компания» содержит разные типы участников, поэтому рейтинг компаний — сравнение записей источника, а не доказанный рейтинг сопоставимых дилерских центров. Все рыночные доли относятся к охвату этого датасета.'''),
('code', SETUP + "\nraw = load_raw('data/autokz2019_raw.csv')\nprint(f'Строк: {len(raw):,}; столбцов: {raw.shape[1]}')\ndisplay(raw.head(5))"),
('md', '## Структура, пропуски и периоды\nПустая строка и неизвестное значение отдельного признака — разные случаи. Не удаляем все строки, в которых есть хотя бы один пропуск.'),
('code', '''profile = pd.DataFrame({'type': raw.dtypes.astype(str), 'missing': raw.isna().sum(),
                        'missing_pct': raw.isna().mean() * 100, 'unique': raw.nunique()})
display(profile)
display(raw.groupby('Месяц').agg(rows=('Год', 'size'), units=('Количество', 'sum')).reindex(MONTHS).dropna())
print('Полностью пустых строк:', raw.isna().all(axis=1).sum())
print('Годы:', raw['Год'].unique())'''),
('md', '''## Явные дубликаты и риск потери продаж
Удаляем полные повторы **по всем исходным полям** согласно заданию. Данные не позволяют отличить повторную загрузку от двух одинаковых сделок; поэтому далее обязательно сравниваем результаты с вариантом без удаления. Удаление столбцов до этой проверки дополнительно стирает различия между строками и не используется как основание для удаления продаж.'''),
('code', '''drop_ru = ['Наименование дилерского центра', 'Форма расчета', 'Сегмент', 'Сегментация Eng',
           'Локализация производства', 'Тип клиента', 'Модификация']
variants = {'Исходник': raw, 'Без полных повторов': raw.drop_duplicates(),
            'Если сначала удалить поля (не используем)': raw.drop(columns=drop_ru).drop_duplicates()}
display(pd.DataFrame({name: {'rows': len(z), 'units': z['Количество'].sum(), 'sales_usd': z['Продажа, USD'].sum()}
                      for name, z in variants.items()}).T)'''),
('md', '## Проверка аномалий\nВозврат и оптовая партия не являются автоматически ошибками. Причину или покупателя оптового заказа источник не раскрывает.'),
('code', '''mask = raw['Количество'].isna() | raw['Количество'].lt(0) | raw['Количество'].ge(100)
display(raw.loc[mask, ['Компания', 'Бренд', 'Модель', 'Количество', 'Цена, USD', 'Продажа, USD']])
expected = raw['Количество'] * raw['Цена, USD']
known = expected.notna()
print('Максимальное расхождение цена × количество и сумма, USD:',
      (raw.loc[known, 'Продажа, USD'] - expected[known]).abs().max())'''),
('md', '## Скрытые проблемы форматов\nПосмотрим на содержательные примеры, а не только на типы столбцов.'),
('code', '''display(raw.loc[raw['Вид топлива'].isin(['2', '1,6']),
        ['Бренд', 'Модель', 'Вид топлива', 'Объём двиг, л,', 'Коробка передач', 'Тип привода']].drop_duplicates())
display(raw.loc[raw['Объём двиг, л,'].isin(['400 Л.С.', '88 KWH']),
        ['Бренд', 'Модель', 'Вид топлива', 'Объём двиг, л,']].drop_duplicates())
print('Компания — вариантов:', raw['Компания'].nunique())
print('Коробка передач — вариантов:', raw['Коробка передач'].nunique())
print('Тип привода — вариантов:', raw['Тип привода'].nunique())'''),
('md', '''## Вывод этапа и сверка с заданием
Файл содержит 39 966 строк, 25 полей и девять месяцев 2019 года; полностью пустых строк нет. Найдено 18 698 повторов сверх первого экземпляра. Есть 6 неизвестных количеств при нулевой сумме, один возврат и две крупные партии по 100 и 115 автомобилей. При известных количествах суммы согласуются с ценой с точностью округления.

**Выполнено:** загрузка с `sep=';'`, `decimal=','`; размер, типы, пропуски, дубликаты и числовые аномалии проверены. Следующий этап — тетрадь 02; все исходные сведения сохранены.''')])

notebook('02_cleaning.ipynb', [
('md', '''# 2. Очистка и проверка результата

Очистка реализована в [clean_data.py](clean_data.py). Тетрадь показывает последовательность, правила, журнал исправлений и контрольные итоги. Скрипт запускается самостоятельно на исходном CSV.

**Порядок:** сначала проверяем повторы исходных записей и используем модификацию для исправления гибридов; затем удаляем ненужные поля. Это осознанное уточнение порядка задания, предотвращающее потерю информации при очистке.'''),
('code', SETUP + "\nraw = load_raw('data/autokz2019_raw.csv')\ndetailed, quarantine, corrections, unresolved, meta = normalize(raw)\ndisplay(pd.Series(meta))"),
('md', '''## Соответствие названий заданию
`Дилер` → «Наименование дилерского центра»; `Тип платежа` → «Форма расчета»; `Локализация` → «Локализация производства». Буквального поля «Сегмент ЕПР» в CSV нет: удалено дополнительное поле «Сегментация Eng» как ближайший аналог по назначению. «Сегментация 2013» и «Класс 2013» сохранены.

«Стоимость» → `unit_price_usd`, «Количество» → `quantity`, «Итоговая стоимость» → `total_sales_usd`. Все итоговые имена английские.'''),
('code', '''from clean_data import RENAME, COUNTRIES
display(pd.DataFrame({'Исходное поле': list(RENAME), 'Английское имя': list(RENAME.values())}))
display(pd.Series(COUNTRIES, name='ALPHA-3').to_frame())'''),
('md', '''## Правила нормализации

- Страны: явный словарь ALPHA-3; «Корея» трактуется как KOR в контексте автомобильных брендов источника. Неизвестная новая страна останавливает скрипт.
- Топливо: F, D, E, HYB. Обозначение `E-Hybrid`/`HYBR` в модификации имеет приоритет над ошибочным бензином.
- Привод: 4WD, FWD, RWD, 2WD. У 2WD неизвестна ведущая ось; приравнивать его к FWD нельзя. `4х2.2` трактуется как двухколёсный привод без определения оси; «Астана» и `0` становятся пропусками.
- Коробка: «Автомат» включает вариатор, робот и электрический редуктор; «Механика» — механическую коробку. TDI не обозначает коробку: оставлен пропуск. Для Cayenne число 8 классифицировано по согласующимся распознанным строкам модели; это документированное восстановление.
- Renault: из смещённых столбцов восстановлены объём и коробка; утраченное топливо не угадывается.
- Двигатель: десятичная точка, удаление суффиксов L/MPI/GDI и др.; Niva исправлена на 1.7 по устойчивому значению остальных строк и последовательности ошибки. У электромобилей объём ДВС 0; мощность и ёмкость батареи не считаются литрами.
- Год выпуска: удалены неразрывные пробелы, тип Int64 допускает пропуски. Регионы и области приведены к `title()`; исторические названия не заменены современными.
- Дата: календарный конец месяца, включая 28 февраля; исходные год и месяц удаляются.'''),
('code', '''display(corrections.groupby(['field', 'reason']).size().rename('rows').reset_index())
display(corrections.head(12))
display(unresolved)
print('Исправленные значения объёма Niva:')
display(detailed.loc[detailed.brand.eq('Chevrolet') & detailed.model.eq('Niva'), 'engine_l'].value_counts())'''),
('md', '''## Пропуски и правдоподобные аномалии
Шесть записей без количества исключены из готовой аналитической таблицы и сохранены в `quarantine.csv`. Нулевая сумма не доказывает нулевое количество. Остальные неизвестные характеристики сохранены как пропуски: среднее или мода могли бы приписать автомобилю неверную комплектацию. В разрезах EDA они показаны как «Неизвестно».

Отрицательное количество сохраняется как возврат. Партии 100 Skoda и 115 Lada сохраняются. Большие двигатели коммерческой техники не удаляются по порогу, подходящему легковым машинам. Значение 2.5 л у автобусов Daewoo вызывает вопрос, но внутреннего подтверждения правильного объёма нет; оно оставлено без выдуманной замены и учитывается как ограничение анализа объёма двигателя.'''),
('code', '''display(quarantine[['source_row', 'brand', 'model', 'quantity', 'total_sales_usd', 'reason']])
display(detailed.loc[detailed.quantity.lt(0) | detailed.quantity.ge(100),
                     ['source_row', 'brand', 'model', 'quantity', 'unit_price_usd', 'total_sales_usd']])'''),
('md', '''## Удаление полей и объединение совпадающих аналитических групп
После нормализации и удаления семи полей строки с одинаковыми оставшимися характеристиками, ценой, месяцем и признаком возврата суммируются. Это **агрегация, а не повторное удаление продаж**. `source_records` сохраняет число представленных исходных записей. Возвраты не смешиваются с положительными продажами.

Итоговое зерно: месяц × компания × характеристики автомобиля × регион/область × цена × признак возврата. Поле `quantity` остаётся весом строки.'''),
('code', '''clean = aggregate_clean(detailed)
display(pd.DataFrame({'До агрегации': [len(detailed), detailed.quantity.sum(), detailed.total_sales_usd.sum()],
                      'После агрегации': [len(clean), clean.quantity.sum(), clean.total_sales_usd.sum()]},
                     index=['Строк', 'Автомобилей нетто', 'Продажи, USD']))
display(clean.dtypes.astype(str).to_frame('dtype'))
display(clean.isna().sum().to_frame('missing_rows'))'''),
('md', '## Экспорт и повторное чтение\nCSV не хранит category/datetime: функция `read_clean()` восстанавливает типы. Контролируем также суммы после повторного чтения.'),
('code', '''clean, detailed, summary = run('data/autokz2019_raw.csv', 'data')
restored = read_clean('data/autokz2019_clean.csv')
pd.testing.assert_frame_equal(clean.reset_index(drop=True), restored.reset_index(drop=True),
                              check_dtype=False, check_categorical=False, rtol=1e-10, atol=1e-6)
validate(restored)
assert not set(DROP + ['year', 'month']) & set(restored.columns)
assert set(restored.sale_date.dt.month) == set(range(1, 10))
assert restored.loc[restored.sale_date.dt.month.eq(2), 'sale_date'].dt.day.eq(28).all()
assert all(isinstance(restored[c].dtype, pd.CategoricalDtype) for c in CATEGORIES)
print('Экспорт, повторное чтение, календарь, категории и контрольные суммы проверены.')
display(pd.Series(summary))'''),
('md', '''## Вывод этапа
Получено 16 141 аналитических строк из 21 262 пригодных исходных записей после удаления точных повторов и карантина. Сохранено 35 165 автомобилей нетто, включая экспорт. Все требуемые преобразования выполнены; неизвестные характеристики не подменены догадками. Протокол исправлений, карантин, схема и промежуточные записи сохранены рядом с итоговым CSV.

**Сверка:** семь лишних полей удалены, английские названия присвоены, повторы обработаны, страны/топливо/привод/компании/двигатель/география/коробки нормализованы, числа и даты приведены к нужным типам. Ограничения: нет идентификатора сделки, часть технических характеристик неизвестна, «Сегмент ЕПР» сопоставлен по смыслу.''')])

notebook('03_market_eda.ipynb', [
('md', '''# 3. Исследовательский анализ рынка

**Популяция:** январь–сентябрь 2019 года; внутренний рынок — записи, где регион не «Экспорт». Все доли относятся к очищенному источнику, а не к внешней официальной статистике. Полная база с экспортом показана отдельно.

**Метрики:** продажи нетто = сумма количества с возвратами; продажи в USD = сумма итоговой стоимости с возвратами. Средняя стоимость проданного автомобиля = сумма положительных продаж / количество положительно проданных автомобилей. Это прокси среднего чека на автомобиль: идентификатора заказа нет. Для распределений веса равны количеству, возвраты исключены. Неизвестные категории включены в знаменатель.

Девять месяцев одного года позволяют описать динамику внутри периода, но не доказать сезонность или годовой рост.'''),
('code', ANALYSIS + '''
kpi = pd.DataFrame({'Вся база': metrics(df), 'Внутренний рынок': metrics(market),
                    'Экспорт': metrics(df.loc[df.is_export])}).T
display(kpi)
kpi.to_csv('tables/market_kpi.csv', encoding='utf-8-sig')
'''),
('md', '## Динамика по месяцам\nПоказываем одновременно объём в автомобилях и сумму продаж; это разные показатели.'),
('code', '''monthly = market.groupby('sale_date').agg(units=('quantity', 'sum'), sales_usd=('total_sales_usd', 'sum'))
monthly['mom_pct'] = monthly.units.pct_change() * 100
display(monthly)
monthly.to_csv('tables/market_monthly.csv')
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
labels = monthly.index.strftime('%m.%Y')
axes[0].plot(labels, monthly.units, marker='o', color=BLUE)
axes[0].set(title='Внутренний рынок: январь–сентябрь 2019', ylabel='Автомобили, нетто', ylim=(0, None))
axes[1].plot(labels, monthly.sales_usd / 1e6, marker='s', color=GOLD)
axes[1].set(ylabel='Продажи, млн USD', xlabel='Месяц', ylim=(0, None))
save_figure('01_market_monthly')
print(f'Пик: {monthly.units.idxmax():%m.%Y}, {monthly.units.max():,} автомобилей; изменение сентябрь/январь: {(monthly.units.iloc[-1]/monthly.units.iloc[0]-1)*100:.1f}%.')'''),
('md', '## Бренды и модели\nРанжирование строится по сумме количества, а не по числу записей.'),
('code', '''brands = breakdown(market, 'brand')
display(brands.head(12))
brands.to_csv('tables/market_brands.csv', encoding='utf-8-sig')
bar(brands.units.head(12), 'Ведущие бренды внутреннего рынка', 'Автомобили, нетто', '02_brands')
print(f'Три крупнейших бренда дают {brands.share_pct.head(3).sum():.1f}% объёма.')'''),
('code', '''model_data = market.assign(brand_model=market.brand + ' ' + market.model)
models = breakdown(model_data, 'brand_model')
display(models.head(12))
models.to_csv('tables/market_models.csv', encoding='utf-8-sig')
bar(models.units.head(12), 'Ведущие модели внутреннего рынка', 'Автомобили, нетто', '03_models')'''),
('md', '## География и сегменты\n«Регион» в этом источнике преимущественно обозначает город. «Область» рассматривается отдельно; названия сохранены в историческом виде.'),
('code', '''regions = breakdown(market, 'region')
areas = breakdown(market, 'area')
segments = breakdown(market, 'segment_2013')
for name, table in [('market_regions', regions), ('market_areas', areas), ('market_segments', segments)]:
    table.to_csv(f'tables/{name}.csv', encoding='utf-8-sig')
display(areas)
bar(regions.units.head(12), 'Крупнейшие региональные рынки', 'Автомобили, нетто', '04_regions')
display(segments)
bar(segments.units, 'Структура рынка по сегментам 2013', 'Автомобили, нетто', '05_segments')'''),
('md', '## Топливо, коробка и привод\nНеизвестные характеристики отображаются отдельной категорией. Они не исчезают из общего объёма.'),
('code', '''for column, label, name in [('fuel_type', 'Вид топлива: F — бензин, D — дизель, E — электро, HYB — гибрид', '06_fuel'),
                            ('transmission', 'Тип коробки передач', '07_transmission'),
                            ('drive_type', 'Тип привода', '08_drive')]:
    table = breakdown(market, column)
    display(table)
    table.to_csv(f'tables/market_{column}.csv', encoding='utf-8-sig')
    bar(table.units, label, 'Автомобили, нетто', name)'''),
('md', '## Распределения цен, объёмов и года выпуска\nВеса равны числу положительно проданных автомобилей. На графике цены логарифмическая ось X позволяет показать весь диапазон без удаления дорогих автомобилей.'),
('code', '''distribution = []
for column in ['unit_price_usd', 'engine_l', 'manufacture_year']:
    values = expanded_values(market, column)
    distribution.append(pd.Series(values).describe(percentiles=[.25, .5, .75, .95]).rename(column))
display(pd.DataFrame(distribution))
prices = expanded_values(market, 'unit_price_usd')
fig, ax = plt.subplots()
ax.hist(prices, bins=np.geomspace(prices.min(), prices.max(), 32), color=BLUE, edgecolor='white')
ax.set_xscale('log')
ax.set(title='Распределение цены проданного автомобиля', xlabel='Цена, USD (логарифмическая шкала)', ylabel='Автомобили')
save_figure('09_price_distribution')
print(f'Взвешенная медиана цены: {np.median(prices):,.0f} USD; 95-й процентиль: {np.percentile(prices,95):,.0f} USD.')'''),
('code', '''engines = expanded_values(market, 'engine_l')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].hist(engines, bins=np.arange(0, 15.5, .5), color=BLUE, edgecolor='white')
axes[0].set(title='Объём двигателя', xlabel='Объём ДВС, л; 0 — электромобиль', ylabel='Автомобили')
year_table = breakdown(positive, 'manufacture_year')
year_table = year_table.reindex(sorted(year_table.index, key=lambda s: (s == 'Неизвестно', s)))
axes[1].bar(year_table.index.astype(str), year_table.units, color=GOLD)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set(title='Год выпуска', xlabel='Год; неизвестные показаны отдельно', ylabel='Автомобили')
save_figure('10_engine_year')
print('Автомобилей с неизвестным объёмом:', positive.loc[positive.engine_l.isna(), 'quantity'].sum())
print('Автомобилей с неизвестным годом:', positive.loc[positive.manufacture_year.isna(), 'quantity'].sum())'''),
('md', '''## Взаимосвязи и неоднородность
Исследуем связь объёма двигателя с ценой отдельно от коммерческой техники и электромобилей. Корреляция — описательная характеристика, а не причинный эффект объёма на цену. Бренд, класс и комплектация могут объяснять часть различий.'''),
('code', '''passenger = positive.loc[positive.segment_2013.ne('Коммерческие автомобили') & positive.engine_l.gt(0)].copy()
sample = passenger[['engine_l', 'unit_price_usd', 'quantity']].dropna()
idx = np.repeat(np.arange(len(sample)), sample.quantity.to_numpy(dtype=int))
weighted = sample.iloc[idx]
corr = weighted[['engine_l', 'unit_price_usd']].corr().iloc[0, 1]
fig, ax = plt.subplots()
ax.scatter(sample.engine_l, sample.unit_price_usd, s=10, alpha=.18, color=BLUE, rasterized=True)
ax.set(title='Объём и цена: некоммерческие автомобили с ДВС', xlabel='Объём, л', ylabel='Цена, USD')
save_figure('11_engine_price')
print(f'Взвешенная по автомобилям корреляция Пирсона: {corr:.3f}; автомобилей: {len(weighted):,}.')
print('Точка — аналитическая строка, корреляция взвешена по количеству.')
segment_prices = positive.groupby('segment_2013', observed=True).agg(units=('quantity', 'sum'), revenue=('total_sales_usd', 'sum'))
segment_prices['avg_usd'] = segment_prices.revenue / segment_prices.units
display(segment_prices.sort_values('avg_usd', ascending=False))
bar(segment_prices.avg_usd.sort_values(ascending=False), 'Средняя стоимость автомобиля по сегментам', 'USD на автомобиль', '12_segment_price')'''),
('md', '''## Аномалии и чувствительность к удалению дубликатов
Удаление точных повторов задано условиями проекта, но без ID сделок не может считаться доказанным восстановлением фактического рынка. Сравнение ниже ограничивает интерпретацию абсолютных объёмов и долей. Также показываем долю продаж из крупных партий на уровне исходных записей, до агрегации.'''),
('code', '''raw = load_raw('data/autokz2019_raw.csv')
rows = []
for label, r in [('Без удаления повторов', raw), ('С удалением полных повторов', raw.drop_duplicates())]:
    r = r.loc[r['Регион'].str.strip().str.casefold().ne('экспорт') & r['Количество'].notna()].copy()
    company = r['Компания'].str.strip().replace(COMPANIES)
    mask = company.eq('Mercur Auto')
    rows.append({'scenario': label, 'market_units': r['Количество'].sum(),
                 'mercur_units': r.loc[mask, 'Количество'].sum(),
                 'mercur_share_pct': 100*r.loc[mask, 'Количество'].sum()/r['Количество'].sum()})
sensitivity = pd.DataFrame(rows)
display(sensitivity)
sensitivity.to_csv('tables/dedup_sensitivity.csv', index=False, encoding='utf-8-sig')
detail = pd.read_csv('data/clean_source_records.csv')
domestic_detail = detail.loc[~detail.is_export]
large = domestic_detail.loc[domestic_detail.quantity.ge(10)]
print(f'Партии от 10 автомобилей: {len(large)} записей, {large.quantity.sum():,.0f} автомобилей, {100*large.quantity.sum()/positive.quantity.sum():.1f}% положительных продаж.')'''),
('md', '''## Вывод этапа
Внутренний рынок в очищенном файле: **34 725 автомобилей нетто**, около **801.70 млн USD**. Экспорт составляет 440 автомобилей и анализируется отдельно. Лидирует Lada; существенную роль играют Hyundai и Toyota. Крупнейшие города — Алматы и Нур-Султан. Максимум месячного объёма приходится на июль, минимум — на январь. Эти результаты описывают только рассматриваемые девять месяцев.

Средняя стоимость положительно проданного автомобиля — около **23 087 USD**. Состав сегментов влияет на цену; поэтому сравнение компаний только по среднему чеку недостаточно. Неизвестные характеристики и чувствительность к дедупликации ограничивают точность общей картины. Крупные партии сохранены; возврат учитывается в нетто-показателях.

**Сверка с заданием:** распределения цены, двигателя и года; бренды, модели, регионы и области, топливо, коробки, привод и сегменты; временная динамика, взаимосвязь и аномалии рассмотрены. Все графики и таблицы основаны на очищенных данных.''')])

notebook('04_mercur_analysis.ipynb', [
('md', '''# 4. Позиция «Меркур Авто» и выводы для менеджмента

Сравниваем `Mercur Auto` (включая исходный вариант `Mercur Autos`) с внутренним рынком за январь–сентябрь 2019 года. Базовое сравнение — весь рынок; дополнительное — остальные компании и сопоставимые бренды. Доли по брендам/регионам/сегментам используют **объём соответствующей группы рынка** в знаменателе. Структура продаж компании использует общий объём самой компании.

Нет данных о прибыли, затратах или целевых планах: высокий средний чек не доказывает большую прибыльность или эффективность работы персонала.'''),
('code', ANALYSIS + '''
rest = market.loc[~market.company.eq('Mercur Auto')]
kpis = pd.DataFrame({'Рынок': metrics(market), 'Меркур Авто': metrics(mercur), 'Остальные компании': metrics(rest)}).T
display(kpis)
unit_share = 100 * mercur.quantity.sum() / market.quantity.sum()
value_share = 100 * mercur.total_sales_usd.sum() / market.total_sales_usd.sum()
print(f'Доля в автомобилях: {unit_share:.3f}%; доля в USD: {value_share:.3f}%.')
print(f'Средняя стоимость выше рынка на {(metrics(mercur).avg_vehicle_usd/metrics(market).avg_vehicle_usd-1)*100:.1f}%.')
kpis.to_csv('tables/mercur_kpi.csv', encoding='utf-8-sig')
'''),
('md', '## Бренды и модели\nДоля 100% означает отсутствие других продавцов бренда в предоставленном датасете; это не доказательство монополии на реальном рынке.'),
('code', '''brand_compare = compare(market, mercur, 'brand')
active_brands = brand_compare.loc[brand_compare.mercur_units.gt(0)]
display(active_brands)
brand_compare.to_csv('tables/mercur_brands.csv', encoding='utf-8-sig')
bar(active_brands.market_share_pct, 'Доля «Меркур Авто» внутри каждого бренда', '% автомобилей бренда в источнике', '13_mercur_brand_share', GOLD)
mercur_models = breakdown(mercur.assign(brand_model=mercur.brand + ' ' + mercur.model), 'brand_model')
display(mercur_models.head(10))
mercur_models.to_csv('tables/mercur_models.csv', encoding='utf-8-sig')'''),
('md', '## География и сегменты\nСопоставляем долю компании на локальном рынке и концентрацию её собственных продаж. Нулевое присутствие не означает готовую возможность открытия точки: нужны спрос, затраты и конкурентные условия.'),
('code', '''region_compare = compare(market, mercur, 'region')
segment_compare = compare(market, mercur, 'segment_2013')
class_compare = compare(market, mercur, 'class_2013')
for name, table in [('mercur_regions', region_compare), ('mercur_segments', segment_compare), ('mercur_classes', class_compare)]:
    table.to_csv(f'tables/{name}.csv', encoding='utf-8-sig')
display(region_compare)
bar(region_compare.loc[region_compare.mercur_units.gt(0), 'market_share_pct'],
    'Доля «Меркур Авто» на рынках присутствия', '% автомобилей региона', '14_mercur_region_share', GOLD)
display(segment_compare)
fig, ax = plt.subplots(figsize=(11, 5))
positions = np.arange(len(segment_compare))
ax.barh(positions-.18, segment_compare.market_mix_pct, height=.36, color=BLUE, label='Рынок')
ax.barh(positions+.18, segment_compare.mercur_mix_pct, height=.36, color=GOLD, label='Меркур Авто')
ax.set_yticks(positions, segment_compare.index)
ax.set(title='Структура продаж по сегментам', xlabel='% автомобилей внутри каждой группы', xlim=(0, 100))
ax.legend()
save_figure('15_mercur_segment_mix')'''),
('md', '## Сравнение среднего чека с учётом состава\nПоказываем цены по брендам и расчёт ожидаемой цены при структуре брендов «Меркур Авто» и среднерыночных ценах этих брендов. Это описательная стандартизация: модели и комплектации всё ещё могут различаться.'),
('code', '''def price_by_brand(z):
    p = z.loc[z.quantity.gt(0)]
    out = p.groupby('brand').agg(units=('quantity', 'sum'), revenue=('total_sales_usd', 'sum'))
    out['avg_usd'] = out.revenue/out.units
    return out
market_price = price_by_brand(market)
mercur_price = price_by_brand(mercur)
brand_prices = mercur_price[['units','avg_usd']].join(market_price[['avg_usd']].rename(columns={'avg_usd':'market_avg_usd'}))
rest_price = price_by_brand(rest)
brand_prices = brand_prices.join(rest_price[['avg_usd']].rename(columns={'avg_usd':'others_avg_usd'}))
display(brand_prices)
brand_prices.to_csv('tables/mercur_brand_prices.csv', encoding='utf-8-sig')
standardized = np.average(brand_prices.market_avg_usd, weights=brand_prices.units)
print(f'Средняя рыночная цена при структуре брендов Меркур: {standardized:,.0f} USD.')
print(f'Фактическая цена Меркур: {metrics(mercur).avg_vehicle_usd:,.0f} USD.')
print('Для Audi/Porsche независимого сравнения с другими продавцами в источнике нет.')
fuel_compare = compare(market, mercur, 'fuel_type')
gear_compare = compare(market, mercur, 'transmission')
display(fuel_compare)
display(gear_compare)'''),
('md', '''## Что произошло с июня
Сравниваем одинаковые месяцы, отдельно проверяем бренды. Снижение наблюдаемого объёма — факт источника; причины изменения без дополнительных данных не установлены.'''),
('code', '''monthly = market.groupby('sale_date').quantity.sum().to_frame('market_units')
monthly['mercur_units'] = mercur.groupby('sale_date').quantity.sum()
monthly['share_pct'] = 100*monthly.mercur_units/monthly.market_units
brand_monthly = mercur.pivot_table(index='sale_date', columns='brand', values='quantity', aggfunc='sum', fill_value=0)
display(monthly)
display(brand_monthly)
monthly.to_csv('tables/mercur_monthly.csv')
brand_monthly.to_csv('tables/mercur_monthly_brands.csv')
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
labels = monthly.index.strftime('%m.%Y')
for brand, color, marker in [('Volkswagen', BLUE, 'o'), ('Audi', GOLD, 's'), ('Porsche', '#666666', '^')]:
    axes[0].plot(labels, brand_monthly[brand], color=color, marker=marker, label=brand)
axes[0].set(title='Продажи «Меркур Авто» по брендам', ylabel='Автомобили, нетто', ylim=(0, None))
axes[0].legend()
axes[1].plot(labels, monthly.share_pct, color=GOLD, marker='o')
axes[1].set(ylabel='Доля в рынке, %', xlabel='Месяц', ylim=(0, None))
save_figure('16_mercur_monthly')
vw = market.loc[market.brand.eq('Volkswagen')].pivot_table(index='sale_date', columns='company', values='quantity', aggfunc='sum', fill_value=0)
display(vw)
vw.to_csv('tables/volkswagen_companies_monthly.csv', encoding='utf-8-sig')'''),
('md', '''С июня у «Меркур Авто» нет ни одной строки Volkswagen; это верно и до удаления дубликатов. Одновременно Volkswagen продолжается под именем Volkswagen Group Rus. **Гипотеза:** изменилась атрибуция/охват поставщика данных или канал продаж. Объединять компании без доказательств нельзя. Поэтому резкое падение нельзя уверенно назвать ухудшением эффективности автоцентра.'''),
('code', '''raw = load_raw('data/autokz2019_raw.csv')
raw_company = raw['Компания'].str.strip().replace(COMPANIES)
after_may = raw['Месяц'].map(MONTHS).ge(6)
assert not (raw_company.eq('Mercur Auto') & raw['Бренд'].eq('Volkswagen') & after_may).any()
first = mercur.loc[mercur.sale_date.dt.month.le(5), 'quantity'].sum()/5
last = mercur.loc[mercur.sale_date.dt.month.ge(6), 'quantity'].sum()/4
premium_first = mercur.loc[mercur.sale_date.dt.month.le(5) & mercur.brand.ne('Volkswagen'), 'quantity'].sum()/5
premium_last = mercur.loc[mercur.sale_date.dt.month.ge(6) & mercur.brand.ne('Volkswagen'), 'quantity'].sum()/4
print(f'Среднемесячно январь–май: {first:.1f}; июнь–сентябрь: {last:.2f}; изменение: {(last/first-1)*100:.1f}%.')
print(f'Audi+Porsche: {premium_first:.1f} против {premium_last:.2f} автомобиля в месяц.')
display(pd.read_csv('tables/dedup_sensitivity.csv'))'''),
('md', '''## Сильные стороны, ограничения и действия

**Наблюдаемые сильные стороны:** компания представлена тремя брендами, занимает 28.05% продаж Volkswagen в источнике и все наблюдаемые продажи Audi/Porsche; денежная доля выше доли в автомобилях. Средняя стоимость автомобиля выше рынка, что связано в том числе с портфелем брендов. Это характеристика позиционирования, не доказательство высокой маржи.

**Уязвимости:** концентрация на Volkswagen (294 из 394 автомобилей), на Алматы (251 из 394), небольшой общий объём и разрыв наблюдения Volkswagen после мая. Нельзя переносить долю первых месяцев на весь год или оценивать результат по единому среднему без разбивки по брендам.

**Рекомендуемые действия:**
1. Сверить продажи Volkswagen за июнь–сентябрь с выгрузкой дилерской системы и правилами атрибуции компаний. До проверки не делать кадровых или бюджетных выводов из падения.
2. Для Volkswagen сравнить наличие, цены, лиды и конверсию по моделям с сопоставимыми продавцами; проверить спрос на наиболее продаваемые модели внутри доступного бренда. Большой рынок Lada сам по себе не означает возможность продавать этот бренд.
3. В Алматы проверить удержание и работу с существующими клиентами; для Нур-Султана проверить спрос и экономику расширения. Присутствие в городе и большой общий рынок не доказывают окупаемости новой точки.
4. Для Audi/Porsche вести отдельные KPI: объём, маржа, оборачиваемость запасов, конверсия. Внешнее конкурентное сравнение потребует дополнительного источника.

**Сверка с заданием:** доли по брендам, регионам и сегментам; KPI и средний чек; структура и динамика продаж; сравнение с рынком; сильные и слабые стороны и рекомендации выполнены. Выводы ограничены качеством и охватом исходных данных.''')])

from enrich_notebooks import enrich
enrich()
print('Created four notebooks')
