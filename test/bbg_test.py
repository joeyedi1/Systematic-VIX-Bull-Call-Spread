import sys; sys.path.insert(0, '.')
from data.bloomberg_fetcher import BloombergDataPipeline

pipeline = BloombergDataPipeline(use_cache=True)
df = pipeline.fetch_all(start_date="20220103", end_date="20260318")
pipeline.close()

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.tail())