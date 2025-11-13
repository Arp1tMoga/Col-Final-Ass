import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create plots directory
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# Load data
df = pd.read_csv("results.csv")

print("=" * 80)
print("BENCHMARK RESULTS ANALYSIS")
print("=" * 80)
print("\nDataset Overview:")
print(df.head(10))
print(f"\nTotal benchmarks: {len(df)}")
print(f"Programs: {df['Program'].unique()}")
print(f"Matrix sizes: {df['Matrix'].unique()}")
print(f"Thread counts: {df['Threads'].unique()}")

# ============================================================================
# PLOT 1: Performance Comparison Across Matrix Sizes (Grouped by Threads)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Execution Time vs Matrix Size (Different Thread Counts)', fontsize=16, fontweight='bold')

thread_counts = sorted(df['Threads'].unique())
for idx, threads in enumerate(thread_counts):
    ax = axes[idx // 3, idx % 3]
    
    data = df[df['Threads'] == threads]
    
    for program in ['PYTHON', 'CPP-OLD', 'CPP-OPT']:
        prog_data = data[data['Program'] == program]
        ax.plot(prog_data['Matrix'], prog_data['Avg'], 
                marker='o', linewidth=2, markersize=8, label=program)
    
    ax.set_xlabel('Matrix Size (N)', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title(f'Threads = {threads}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig(plots_dir / '01_performance_by_threads.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_performance_by_threads.png")
plt.close()

# ============================================================================
# PLOT 2: Speedup Analysis (CPP-OPT vs PYTHON baseline)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

matrix_sizes = sorted(df['Matrix'].unique())
thread_counts = sorted(df['Threads'].unique())

x = np.arange(len(thread_counts))
width = 0.2

for i, matrix_size in enumerate(matrix_sizes):
    speedups = []
    for threads in thread_counts:
        python_time = df[(df['Program'] == 'PYTHON') & 
                        (df['Matrix'] == matrix_size) & 
                        (df['Threads'] == threads)]['Avg'].values[0]
        
        cpp_opt_time = df[(df['Program'] == 'CPP-OPT') & 
                         (df['Matrix'] == matrix_size) & 
                         (df['Threads'] == threads)]['Avg'].values[0]
        
        speedup = python_time / cpp_opt_time
        speedups.append(speedup)
    
    ax.bar(x + i * width, speedups, width, label=f'N={matrix_size}', alpha=0.8)

ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
ax.set_ylabel('Speedup (Python / CPP-OPT)', fontweight='bold', fontsize=12)
ax.set_title('CPP-OPT Speedup Over Python Baseline', fontweight='bold', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(thread_counts)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')

plt.tight_layout()
plt.savefig(plots_dir / '02_speedup_vs_python.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_speedup_vs_python.png")
plt.close()

# ============================================================================
# PLOT 3: Scaling Efficiency (Strong Scaling)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Strong Scaling Analysis (Speedup vs Thread Count)', fontsize=16, fontweight='bold')

programs = ['PYTHON', 'CPP-OLD', 'CPP-OPT']
colors = ['#E74C3C', '#F39C12', '#27AE60']

for idx, matrix_size in enumerate(matrix_sizes):
    ax = axes[idx // 2, idx % 2]
    
    for prog_idx, program in enumerate(programs):
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        base_time = data[data['Threads'] == 1]['Avg'].values[0]
        
        threads = []
        speedups = []
        for _, row in data.iterrows():
            threads.append(row['Threads'])
            speedups.append(base_time / row['Avg'])
        
        ax.plot(threads, speedups, marker='o', linewidth=2, markersize=8, 
                label=program, color=colors[prog_idx])
    
    # Ideal scaling line
    max_threads = max(thread_counts)
    ax.plot([1, max_threads], [1, max_threads], 'k--', linewidth=2, 
            alpha=0.5, label='Ideal Linear Scaling')
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Speedup vs 1 Thread', fontweight='bold')
    ax.set_title(f'Matrix Size N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_threads + 1])

plt.tight_layout()
plt.savefig(plots_dir / '03_strong_scaling.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_strong_scaling.png")
plt.close()

# ============================================================================
# PLOT 4: Performance Heatmap
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Performance Heatmap (Execution Time in Seconds)', fontsize=16, fontweight='bold')

for idx, program in enumerate(['PYTHON', 'CPP-OLD', 'CPP-OPT']):
    pivot = df[df['Program'] == program].pivot(index='Threads', columns='Matrix', values='Avg')
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Time (seconds)'}, ax=axes[idx], 
                linewidths=0.5, linecolor='gray')
    
    axes[idx].set_title(f'{program}', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Matrix Size (N)', fontweight='bold')
    axes[idx].set_ylabel('Number of Threads', fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / '04_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_performance_heatmap.png")
plt.close()

# ============================================================================
# PLOT 5: Standard Deviation Analysis (Variability)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Variability (Standard Deviation)', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    ax = axes[idx // 2, idx % 2]
    
    data = df[df['Matrix'] == matrix_size]
    
    x = np.arange(len(thread_counts))
    width = 0.25
    
    for prog_idx, program in enumerate(['PYTHON', 'CPP-OLD', 'CPP-OPT']):
        prog_data = data[data['Program'] == program]
        
        # Calculate coefficient of variation (CV) = StdDev / Mean * 100
        cv = (prog_data['StdDev'] / prog_data['Avg'] * 100).values
        
        ax.bar(x + prog_idx * width, cv, width, label=program, alpha=0.8)
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    ax.set_title(f'Matrix Size N = {matrix_size}', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(thread_counts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / '05_variability_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_variability_analysis.png")
plt.close()

# ============================================================================
# PLOT 6: CPP-OPT vs CPP-OLD Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(thread_counts))
width = 0.2

for i, matrix_size in enumerate(matrix_sizes):
    speedups = []
    for threads in thread_counts:
        cpp_old_time = df[(df['Program'] == 'CPP-OLD') & 
                         (df['Matrix'] == matrix_size) & 
                         (df['Threads'] == threads)]['Avg'].values[0]
        
        cpp_opt_time = df[(df['Program'] == 'CPP-OPT') & 
                         (df['Matrix'] == matrix_size) & 
                         (df['Threads'] == threads)]['Avg'].values[0]
        
        speedup = cpp_old_time / cpp_opt_time
        speedups.append(speedup)
    
    ax.bar(x + i * width, speedups, width, label=f'N={matrix_size}', alpha=0.8)

ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
ax.set_ylabel('Speedup (CPP-OLD / CPP-OPT)', fontweight='bold', fontsize=12)
ax.set_title('CPP-OPT Improvement Over CPP-OLD Baseline', fontweight='bold', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(thread_counts)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No improvement')

plt.tight_layout()
plt.savefig(plots_dir / '06_cpp_opt_vs_cpp_old.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_cpp_opt_vs_cpp_old.png")
plt.close()

# ============================================================================
# PLOT 7: Parallel Efficiency
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parallel Efficiency (Speedup / Number of Threads)', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    ax = axes[idx // 2, idx % 2]
    
    for prog_idx, program in enumerate(['PYTHON', 'CPP-OLD', 'CPP-OPT']):
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        base_time = data[data['Threads'] == 1]['Avg'].values[0]
        
        threads = []
        efficiency = []
        for _, row in data.iterrows():
            t = row['Threads']
            speedup = base_time / row['Avg']
            eff = (speedup / t) * 100  # Efficiency as percentage
            threads.append(t)
            efficiency.append(eff)
        
        ax.plot(threads, efficiency, marker='o', linewidth=2, markersize=8, 
                label=program, color=colors[prog_idx])
    
    ax.axhline(y=100, color='black', linestyle='--', linewidth=2, 
               alpha=0.5, label='100% Efficient')
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
    ax.set_title(f'Matrix Size N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 120])

plt.tight_layout()
plt.savefig(plots_dir / '07_parallel_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_parallel_efficiency.png")
plt.close()

# ============================================================================
# PLOT 8: Best Configuration Summary
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Best Thread Configuration for Each Matrix Size', fontsize=16, fontweight='bold')

for prog_idx, program in enumerate(['PYTHON', 'CPP-OLD', 'CPP-OPT']):
    ax = axes[prog_idx]
    
    best_times = []
    best_threads = []
    
    for matrix_size in matrix_sizes:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        best_row = data.loc[data['Avg'].idxmin()]
        best_times.append(best_row['Avg'])
        best_threads.append(best_row['Threads'])
    
    bars = ax.bar(range(len(matrix_sizes)), best_times, alpha=0.8, 
                   color=colors[prog_idx])
    
    # Annotate with thread count
    for i, (bar, threads) in enumerate(zip(bars, best_threads)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{threads}T\n{height:.3f}s',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Matrix Size (N)', fontweight='bold')
    ax.set_ylabel('Best Time (seconds)', fontweight='bold')
    ax.set_title(f'{program}', fontweight='bold')
    ax.set_xticks(range(len(matrix_sizes)))
    ax.set_xticklabels(matrix_sizes)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / '08_best_configurations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_best_configurations.png")
plt.close()

# ============================================================================
# Generate Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. BEST PERFORMANCE FOR EACH MATRIX SIZE (CPP-OPT):")
print("-" * 80)
for matrix_size in matrix_sizes:
    data = df[(df['Program'] == 'CPP-OPT') & (df['Matrix'] == matrix_size)]
    best = data.loc[data['Avg'].idxmin()]
    python_time = df[(df['Program'] == 'PYTHON') & 
                    (df['Matrix'] == matrix_size) & 
                    (df['Threads'] == best['Threads'])]['Avg'].values[0]
    speedup = python_time / best['Avg']
    
    print(f"  N={matrix_size:4d}: {best['Avg']:.4f}s with {int(best['Threads']):2d} threads "
          f"(Speedup vs Python: {speedup:.2f}x)")

print("\n2. MAXIMUM SPEEDUPS ACHIEVED:")
print("-" * 80)
max_speedup = 0
max_config = None
for _, row in df[df['Program'] == 'CPP-OPT'].iterrows():
    python_time = df[(df['Program'] == 'PYTHON') & 
                    (df['Matrix'] == row['Matrix']) & 
                    (df['Threads'] == row['Threads'])]['Avg'].values[0]
    speedup = python_time / row['Avg']
    if speedup > max_speedup:
        max_speedup = speedup
        max_config = row

print(f"  Maximum: {max_speedup:.2f}x at N={int(max_config['Matrix'])}, "
      f"Threads={int(max_config['Threads'])}")

print("\n3. CPP-OPT IMPROVEMENT OVER CPP-OLD:")
print("-" * 80)
for matrix_size in matrix_sizes:
    improvements = []
    for threads in thread_counts:
        cpp_old = df[(df['Program'] == 'CPP-OLD') & 
                    (df['Matrix'] == matrix_size) & 
                    (df['Threads'] == threads)]['Avg'].values[0]
        cpp_opt = df[(df['Program'] == 'CPP-OPT') & 
                    (df['Matrix'] == matrix_size) & 
                    (df['Threads'] == threads)]['Avg'].values[0]
        improvements.append(cpp_old / cpp_opt)
    
    avg_improvement = np.mean(improvements)
    max_improvement = np.max(improvements)
    print(f"  N={matrix_size:4d}: Avg={avg_improvement:.2f}x, Max={max_improvement:.2f}x")

print("\n4. SCALING EFFICIENCY (8 threads vs 1 thread, N=4096):")
print("-" * 80)
for program in ['PYTHON', 'CPP-OLD', 'CPP-OPT']:
    time_1t = df[(df['Program'] == program) & 
                (df['Matrix'] == 4096) & 
                (df['Threads'] == 1)]['Avg'].values[0]
    time_8t = df[(df['Program'] == program) & 
                (df['Matrix'] == 4096) & 
                (df['Threads'] == 8)]['Avg'].values[0]
    speedup = time_1t / time_8t
    efficiency = (speedup / 8) * 100
    print(f"  {program:10s}: {speedup:.2f}x speedup, {efficiency:.1f}% efficient")

print("\n5. MOST STABLE IMPLEMENTATION (Lowest Coefficient of Variation):")
print("-" * 80)
df['CV'] = (df['StdDev'] / df['Avg'] * 100)
for program in ['PYTHON', 'CPP-OLD', 'CPP-OPT']:
    prog_data = df[df['Program'] == program]
    avg_cv = prog_data['CV'].mean()
    min_cv = prog_data['CV'].min()
    max_cv = prog_data['CV'].max()
    print(f"  {program:10s}: Avg CV={avg_cv:.2f}%, Range=[{min_cv:.2f}%, {max_cv:.2f}%]")

print("\n" + "=" * 80)
print("All plots saved in 'plots/' directory")
print("=" * 80)
