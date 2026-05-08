import gzip
import time
import tracemalloc
from itertools import combinations
import os
import sys

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
                # Pruning step
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
        
        # Count support
        current_f = {}
        for c in candidates:
            count = sum(1 for t in transactions if c.issubset(t))
            if count >= min_sup_count:
                current_f[c] = count
                
        frequent_itemsets.update(current_f)
        k += 1
        
    return frequent_itemsets, candidates_count

def ofim(transactions, min_sup_count):
    # Convert to vertical bitvectors
    num_transactions = len(transactions)
    item_bitvectors = {}
    
    # First pass: find frequent 1-itemsets
    item_counts = {}
    for i, t in enumerate(transactions):
        for item in t:
            if item not in item_bitvectors:
                item_bitvectors[item] = 0
            item_bitvectors[item] |= (1 << i)
            item_counts[item] = item_counts.get(item, 0) + 1
            
    f1_items = [item for item, count in item_counts.items() if count >= min_sup_count]
    f1_items.sort() # Ensure consistent ordering
    
    frequent_itemsets = {}
    for item in f1_items:
        frequent_itemsets[frozenset([item])] = item_counts[item]
        
    def mine_vertical(prefix, prefix_bv, items_to_add):
        for i, item in enumerate(items_to_add):
            new_bv = prefix_bv & item_bitvectors[item]
            count = new_bv.bit_count()
            if count >= min_sup_count:
                new_itemset = prefix | frozenset([item])
                frequent_itemsets[new_itemset] = count
                mine_vertical(new_itemset, new_bv, items_to_add[i+1:])
                
    mine_vertical(frozenset(), (1 << num_transactions) - 1, f1_items)
    return frequent_itemsets

def run_experiment(dataset_name, filepath, supports):
    print(f"--- Running experiments for {dataset_name} ---")
    transactions = load_data(filepath)
    num_transactions = len(transactions)
    print(f"Total transactions: {num_transactions}")
    
    for sup in supports:
        min_sup_count = int(num_transactions * sup)
        print(f"\\nMin Support: {sup*100}% ({min_sup_count} transactions)")
        
        # Apriori
        tracemalloc.start()
        start_time = time.time()
        apriori_freq, apriori_cands = apriori(transactions, min_sup_count)
        apriori_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        apriori_mem = peak / 1024 / 1024 # MB
        
        print(f"Apriori: Time={apriori_time:.4f}s, Mem={apriori_mem:.4f}MB, Freq={len(apriori_freq)}, Cands={apriori_cands}")
        
        # OFIM
        tracemalloc.start()
        start_time = time.time()
        ofim_freq = ofim(transactions, min_sup_count)
        ofim_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ofim_mem = peak / 1024 / 1024 # MB
        
        print(f"OFIM: Time={ofim_time:.4f}s, Mem={ofim_mem:.4f}MB, Freq={len(ofim_freq)}")
        
        # Speedup
        speedup = apriori_time / ofim_time if ofim_time > 0 else float('inf')
        print(f"Speedup (Apriori/OFIM): {speedup:.4f}")
        
        assert len(apriori_freq) == len(ofim_freq), "Mismatch in frequent itemsets!"
        
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
