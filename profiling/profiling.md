# Profiling

Hardware Stack:
- CPU: Intel i7-9700
- Disk: Seagate ST8000DM004 HDD

Software Stack:
- pandas 1.3.3
- tensorflow 2.6.0
- tbparse (master)

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
|        01 | False |        (None) | (10.56) |
|        01 |  True |        (None) | (15.08) |
|        02 | False |        (None) | (10.80) |
|        02 |  True |        (None) | (13.95) |
|        03 | False |        (None) | (103.12, 103.05, 103.32) |
|        03 |  True |        (None) | (145.10, 145.49, 145.17) |

## Profiling Details

```py
pip install line_profiler
# Add `@profile` on functions that should be profiled
kernprof -lv profiling/timing/01.py
```
