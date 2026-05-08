import gzip
import time
import tracemalloc
from itertools import combinations
import os
import sys
import concurrent.futures

def load_data(filepath):
    transactions = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            transactions.append(set(line.strip().split()))
    return transactions

def get_f1_and_transactions(transactions, min_sup_count):
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    f1 = {frozenset([item]): count for item, count in item_counts.items() if count >= min_sup_count}
    return f1

def generate_candidates(fk_minus_1, k):
    candidates = set()
    items = list(fk_minus_1.keys())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            l1 = list(items[i])
            l2 = list(items[j])
            l1.sort()
            l2.sort()
            if l1[:k-2] == l2[:k-2]:
                c = items[i] | items[j]
                subsets = [frozenset(x) for x in combinations(c, k-1)]
                if all(sub in fk_minus_1 for sub in subsets):
                    candidates.add(c)
    return candidates

def apriori(transactions, min_sup_count):
    frequent_itemsets = {}
    candidates_count = 0
    
    f1 = get_f1_and_transactions(transactions, min_sup_count)
    frequent_itemsets.update(f1)
    
    k = 2
    current_f = f1
    
    while current_f:
        candidates = generate_candidates(current_f, k)
        candidates_count += len(candidates)
        
        current_f = {}
        for c in candidates:
            count = sum(1 for t in transactions if c.issubset(t))
            if count >= min_sup_count:
                current_f[c] = count
                
        frequent_itemsets.update(current_f)
        k += 1
        
    return frequent_itemsets, candidates_count

def count_support_chunk(chunk, candidates):
    counts = {c: 0 for c in candidates}
    for t in chunk:
        for c in candidates:
            if c.issubset(t):
                counts[c] += 1
    return counts

def apriori_multiprocess(transactions, min_sup_count):
    frequent_itemsets = {}
    candidates_count = 0
    
    f1 = get_f1_and_transactions(transactions, min_sup_count)
    frequent_itemsets.update(f1)
    
    k = 2
    current_f = f1
    
    num_workers = os.cpu_count() or 4
    chunk_size = len(transactions) // num_workers + 1
    chunks = [transactions[i:i + chunk_size] for i in range(0, len(transactions), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        while current_f:
            candidates = generate_candidates(current_f, k)
            candidates_count += len(candidates)
            if not candidates:
                break
                
            current_f = {}
            futures = [executor.submit(count_support_chunk, chunk, candidates) for chunk in chunks]
            
            candidate_counts = {c: 0 for c in candidates}
            for future in concurrent.futures.as_completed(futures):
                chunk_counts = future.result()
                for c, count in chunk_counts.items():
                    candidate_counts[c] += count
                    
            for c, count in candidate_counts.items():
                if count >= min_sup_count:
                    current_f[c] = count
                    
            frequent_itemsets.update(current_f)
            k += 1
            
    return frequent_itemsets, candidates_count

def apriori_tx_reduction(transactions, min_sup_count):
    frequent_itemsets = {}
    candidates_count = 0
    
    f1 = get_f1_and_transactions(transactions, min_sup_count)
    frequent_itemsets.update(f1)
    
    k = 2
    current_f = f1
    
    f1_items = set([list(itemset)[0] for itemset in f1.keys()])
    reduced_transactions = [set(item for item in t if item in f1_items) for t in transactions]
    reduced_transactions = [t for t in reduced_transactions if len(t) >= k]
    
    while current_f:
        candidates = generate_candidates(current_f, k)
        candidates_count += len(candidates)
        if not candidates:
            break
            
        current_f = {}
        for c in candidates:
            count = sum(1 for t in reduced_transactions if c.issubset(t))
            if count >= min_sup_count:
                current_f[c] = count
                
        frequent_itemsets.update(current_f)
        
        k += 1
        valid_items = set()
        for c in current_f.keys():
            valid_items.update(c)
            
        new_reduced = []
        for t in reduced_transactions:
            new_t = t.intersection(valid_items)
            if len(new_t) >= k:
                new_reduced.append(new_t)
        reduced_transactions = new_reduced
        
    return frequent_itemsets, candidates_count


def ofim(transactions, min_sup_count):
    """
    OFIM: Ordered Frequent Itemset Matrix Algorithm
    Reference: Al-Badani & Al-Khulaidi (2024)
    
    Constructs a compact 2D matrix (N x M) where each row contains a
    transaction's frequent items sorted by descending support frequency
    (Ordered Frequent Itemsets Lists / OFILs). Mines frequent k-itemsets
    via prefix-projection on the matrix rows.
    """
    from collections import defaultdict
    
    num_transactions = len(transactions)
    
    # Phase 1: Count item frequencies
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1
    
    # Filter frequent 1-itemsets
    freq_items = {item: count for item, count in item_counts.items() 
                  if count >= min_sup_count}
    
    if not freq_items:
        return {}
    
    # Sort items by descending frequency (ties broken by item name)
    sorted_items = sorted(freq_items.keys(), key=lambda x: (-freq_items[x], x))
    item_rank = {item: rank for rank, item in enumerate(sorted_items)}
    
    # Phase 2: Build the OFIM matrix (N x M)
    ofim_matrix = []
    for t in transactions:
        row = sorted([item for item in t if item in freq_items], 
                     key=lambda x: item_rank[x])
        if row:
            ofim_matrix.append(row)
    
    # Collect frequent itemsets
    frequent_itemsets = {}
    for item in freq_items:
        frequent_itemsets[frozenset([item])] = freq_items[item]
    
    # Phase 3: Mine k-itemsets using OFIL prefix-projection
    def mine_ofim_recursive(prefix, projected_rows, start_col_items):
        item_counts_local = defaultdict(int)
        item_rows = defaultdict(list)
        
        for row in projected_rows:
            seen = set()
            for item in row:
                if item in seen:
                    continue
                seen.add(item)
                item_counts_local[item] += 1
                idx = row.index(item)
                suffix = row[idx + 1:]
                if suffix:
                    item_rows[item].append(suffix)
        
        for item in start_col_items:
            if item not in item_counts_local:
                continue
            count = item_counts_local[item]
            if count >= min_sup_count:
                new_prefix = prefix | frozenset([item])
                frequent_itemsets[new_prefix] = count
                if item in item_rows and item_rows[item]:
                    next_items = [it for it in start_col_items 
                                  if item_rank.get(it, float('inf')) > item_rank.get(item, float('inf'))]
                    if next_items:
                        mine_ofim_recursive(new_prefix, item_rows[item], next_items)
    
    for i, item in enumerate(sorted_items):
        projected = []
        for row in ofim_matrix:
            if item in row:
                idx = row.index(item)
                suffix = row[idx + 1:]
                if suffix:
                    projected.append(suffix)
        if projected:
            remaining_items = sorted_items[i + 1:]
            if remaining_items:
                mine_ofim_recursive(frozenset([item]), projected, remaining_items)
    
    return frequent_itemsets

def run_experiment(dataset_name, filepath, supports):
    print(f"--- Running experiments for {dataset_name} ---")
    transactions = load_data(filepath)
    num_transactions = len(transactions)
    print(f"Total transactions: {num_transactions}")
    
    for sup in supports:
        min_sup_count = int(num_transactions * sup)
        print(f"\\nMin Support: {sup*100}% ({min_sup_count} transactions)")
        
        tracemalloc.start()
        start_time = time.time()
        apriori_freq, apriori_cands = apriori(transactions, min_sup_count)
        apriori_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        apriori_mem = peak / 1024 / 1024 # MB
        
        print(f"Apriori: Time={apriori_time:.4f}s, Mem={apriori_mem:.4f}MB, Freq={len(apriori_freq)}, Cands={apriori_cands}")
        
        
        tracemalloc.start()
        start_time = time.time()
        ofim_freq = ofim(transactions, min_sup_count)
        ofim_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ofim_mem = peak / 1024 / 1024 # MB
        
        print(f"OFIM: Time={ofim_time:.4f}s, Mem={ofim_mem:.4f}MB, Freq={len(ofim_freq)}")
        
        tracemalloc.start()
        start_time = time.time()
        api_mp_freq, api_mp_cands = apriori_multiprocess(transactions, min_sup_count)
        api_mp_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        api_mp_mem = peak / 1024 / 1024 
        
        print(f"Apriori MP: Time={api_mp_time:.4f}s, Mem={api_mp_mem:.4f}MB, Freq={len(api_mp_freq)}")
        
        tracemalloc.start()
        start_time = time.time()
        api_tx_freq, api_tx_cands = apriori_tx_reduction(transactions, min_sup_count)
        api_tx_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        api_tx_mem = peak / 1024 / 1024 
        
        print(f"Apriori Tx: Time={api_tx_time:.4f}s, Mem={api_tx_mem:.4f}MB, Freq={len(api_tx_freq)}")

        speedup = apriori_time / ofim_time if ofim_time > 0 else float('inf')
        speedup_mp = apriori_time / api_mp_time if api_mp_time > 0 else float('inf')
        speedup_tx = apriori_time / api_tx_time if api_tx_time > 0 else float('inf')
        print(f"Speedup (Apriori/OFIM): {speedup:.4f}")
        print(f"Speedup MP (Apriori/MP): {speedup_mp:.4f}")
        print(f"Speedup Tx (Apriori/Tx): {speedup_tx:.4f}")
        
        assert len(apriori_freq) == len(ofim_freq) == len(api_mp_freq) == len(api_tx_freq), "Mismatch in frequent itemsets!"
        
if __name__ == "__main__":
    supports = [0.90, 0.80, 0.70]
    datasets = [
        ("chess", "chess.dat.gz", [0.90, 0.85, 0.80]),
        ("connect", "connect.dat.gz", [0.98, 0.96, 0.94]),
        ("accidents", "accidents.dat.gz", [0.98, 0.96, 0.94])
    ]
    
    for name, filepath, sups in datasets:
        if os.path.exists(filepath):
            run_experiment(name, filepath, sups)
        else:
            print(f"File not found: {filepath}")
