# Synthetic Recommendation Dataset

This is generator of spen source synthetic recommendation dataset, which simulates pseudo shopping site.
The purpose of this dataset is mainly for testing recommendation algorithm without real data.

## Generating dataset

Run

```py
python synrecom.py
```

This generates two directories `tmp` and `output`.
The output files are in the `output`.

```
-rw-r--r-- 1 jovyan jovyan   1689536 Nov 21 15:42 item.parquet
-rw-r--r-- 1 jovyan jovyan 365243356 Nov 21 15:44 transaction.parquet
-rw-r--r-- 1 jovyan jovyan   8806794 Nov 21 15:42 user.parquet
```

## Data schema

user.parquet

|field name|type|description|
|---|---|--|
|user|string|User ID|
|gender|string|User gender. `alpha` or `beta`|
|age|float|User age between 0.0 and 10.0|
|location|string|User's location. `north`, `east`, `south`, `west`|

item.parquet

|field name|type|description|
|---|---|--|
|item|string|Item ID|
|color|string|Item color|

transaction.parquet

|field name|type|description|
|---|---|--|
|user|string|User ID|
|session|string|Session ID|
|timestamp|datetime|Timestamp of the transaction bewtween 2023-10-01 and 2023-10-31|
|item|string|Item ID|
