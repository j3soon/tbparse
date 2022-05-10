# Profiling

Hardware Stack:
- CPU: Intel i7-9700
- Disk: Seagate ST8000DM004 HDD

Software Stack:
- pandas 1.3.3
- tensorflow 2.6.0
- tbparse 0.0.6

## Generate Benchmarks

```py
rm -rf profiling/benchmarks
python profiling/generators/01.py
python profiling/generators/02.py
python profiling/generators/03.py
```

## Run Profile

```py
python profiling/timing/01.py
python profiling/timing/02.py
python profiling/timing/03.py
```

## Results

Note: The results of benchmarks 01 and 02 are pretty noisy.

Three runs:

| Benchmark | Pivot | Extra Columns | Seconds |
|-----------|-------|---------------|---------|
|        01 | False |        (None) | (10.88, 10.90, 10.47) |
|        01 |  True |        (None) | (23.67, 23.96, 23.07) |
|        02 | False |        (None) | (10.47, 10.59, 10.65) |
|        02 |  True |        (None) | (15.65, 15.60, 15.99) |
|        03 | False |        (None) | (102.98, 103.95, 105.84) |
|        03 |  True |        (None) | (184.05, 189.37, 189.57) |

## Profiling Details

```py
pip install line_profiler
# Add `@profile` on functions that should be profiled
kernprof -lv profiling/timing/01.py
```
