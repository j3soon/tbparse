import os
import sys
import time
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{parent_dir}/../../")
from tbparse import SummaryReader

log_dir = f"{parent_dir}/../benchmarks/run_03"

def time_tbparse():
    for use_pivot in {False, True}:
        start = time.time()
        reader = SummaryReader(log_dir, pivot=use_pivot)
        df = reader.scalars
        end = time.time()
        print(f"pivot={use_pivot}: %.2f" % (end - start))
time_tbparse()