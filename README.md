# Frequent Itemset Mining (FIM) Analysis: Apriori vs. OFIM

## Overview
This repository contains the implementation and empirical benchmarking of Frequent Itemset Mining (FIM) algorithms on dense datasets. The project rigorously compares the classical **Apriori** algorithm against a state-of-the-art vertical data formatting approach, **OFIM** (Ordered Frequent Itemset Matrix). 

Additionally, it explores two distinct optimization strategies to address the inherent bottlenecks of the classical Apriori algorithm: CPU Multi-processing and Dynamic Transaction Reduction.

## Algorithms Implemented
* **Baseline Apriori:** The classical breadth-first search, generate-and-test paradigm.
* **OFIM (Ordered Frequent Itemset Matrix):** A modern vertical-format algorithm that computes support counts via hardware-level bitwise AND operations.
* **Optimized Apriori (MP):** An enhanced Apriori variant utilizing Python's `ProcessPoolExecutor` to distribute the computationally heavy support counting phase across multiple logical CPU cores.
* **Optimized Apriori (Tx):** An enhanced Apriori variant that dynamically prunes mathematically obsolete transactions from memory during execution to drastically reduce loop iterations and I/O overhead.

## Datasets
The algorithms are benchmarked using standard dense datasets from the FIMI (Frequent Itemset Mining Implementations) repository:
1. `chess.dat.gz` (3,196 transactions)
2. `connect.dat.gz` (67,557 transactions)
3. `accidents.dat.gz` (340,183 transactions)

*Note: Ensure the datasets are compressed in `.gz` format and placed in the root directory alongside the python script before running the benchmark.*

## Prerequisites
The codebase is designed to be lightweight and relies exclusively on standard Python libraries. No external pip packages (like Pandas or NumPy) are required.
* Python 3.10+
* Standard libraries utilized: `gzip`, `time`, `tracemalloc`, `itertools`, `os`, `concurrent.futures`

## Usage
To execute the benchmarking suite, run the following command in your terminal:

```bash
python3 benchmark_fim.py