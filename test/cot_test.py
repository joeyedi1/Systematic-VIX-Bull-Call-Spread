import sys; sys.path.insert(0, '.')
from data.cot_fetcher import COTFetcher

cot = COTFetcher()
cot_df = cot.fetch_all(start_year=2022)

print(cot_df.shape)
print(cot_df.columns.tolist())
print(cot_df.tail())