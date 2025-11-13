#!/usr/bin/env python3
"""
Comprehensive Benchmark Visualization & Analysis
Generates 12+ different plots and detailed statistical analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Set working directory to script location
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

# Create output directory
plots_dir = script_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

# Load data
results_file = script_dir / 'results.csv' if (script_dir / 'results.csv').exists() else Path('results.csv')
df = pd.read_csv(results_file)

# Add computed columns
df['CV'] = (df['StdDev'] / df['Avg'] * 100).round(2)  # Coefficient of Variation
df['GFLOPS'] = (2 * df['Matrix']**3 / df['Avg']) / 1e9  # Gigaflops

matrix_sizes = sorted(df['Matrix'].unique())
thread_counts = sorted(df['Threads'].unique())

print("=" * 100)
print(" " * 35 + "STARTING VISUALIZATION GENERATION")
print("=" * 100)

# ============================================================================
# PLOT 1: Time vs Matrix Size (Fixed Threads) - Dynamic subplots
# ============================================================================
print("\n📊 Generating Plot 1: Time vs Matrix Size (Fixed Threads)...")
# Calculate optimal subplot layout
n_plots = len(thread_counts)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
fig.suptitle('Performance vs Matrix Size (Fixed Thread Count)', fontsize=16, fontweight='bold')

# Flatten axes array for easier indexing
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

for idx, threads in enumerate(thread_counts):
    ax = axes_flat[idx]
    
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Threads'] == threads)]
        ax.plot(data['Matrix'], data['Avg'], marker='o', linewidth=2, markersize=8, label=program)
        ax.fill_between(data['Matrix'], 
                        data['Avg'] - data['StdDev'], 
                        data['Avg'] + data['StdDev'], 
                        alpha=0.2)
    
    ax.set_xlabel('Matrix Size (N)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Threads = {threads}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

# Hide unused subplots
for idx in range(n_plots, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig(plots_dir / 'plot1_time_vs_matrixsize.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot1_time_vs_matrixsize.png")
plt.close()

# ============================================================================
# PLOT 2: Time vs Threads (Fixed Matrix Size) - 4 subplots
# ============================================================================
print("\n📊 Generating Plot 2: Time vs Threads (Fixed Matrix Size)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Scaling Behavior vs Thread Count', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        ax.plot(data['Threads'], data['Avg'], marker='s', linewidth=2, markersize=10, label=program)
        ax.errorbar(data['Threads'], data['Avg'], yerr=data['StdDev'], 
                   fmt='none', ecolor='gray', alpha=0.5, capsize=5)
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thread_counts)

plt.tight_layout()
plt.savefig(plots_dir / 'plot2_time_vs_threads.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot2_time_vs_threads.png")
plt.close()

# ============================================================================
# PLOT 3: Coefficient of Variation (Stability Analysis)
# ============================================================================
print("\n📊 Generating Plot 3: Coefficient of Variation (Stability)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Measurement Stability (Coefficient of Variation)', fontsize=16, fontweight='bold')

# By matrix size
for program in ['PYTHON', 'CPP']:
    data = df[df['Program'] == program].groupby('Matrix')['CV'].mean()
    ax1.plot(data.index, data.values, marker='o', linewidth=2, markersize=8, label=program)

ax1.set_xlabel('Matrix Size (N)')
ax1.set_ylabel('Average CV (%)')
ax1.set_title('Stability vs Matrix Size')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')

# Bythread count
for program in ['PYTHON', 'CPP']:
    data = df[df['Program'] == program].groupby('Threads')['CV'].mean()
    ax2.plot(data.index, data.values, marker='s', linewidth=2, markersize=8, label=program)

ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Average CV (%)')
ax2.set_title('Stability vs Thread Count')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(thread_counts)
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')

plt.tight_layout()
plt.savefig(plots_dir / 'plot3_coefficient_of_variation.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot3_coefficient_of_variation.png")
plt.close()

# ============================================================================
# PLOT 4: CPP Speedup Over Python (Grouped Bar Chart)
# ============================================================================
print("\n📊 Generating Plot 4: CPP Speedup Over Python...")
speedup_data = []
for matrix_size in matrix_sizes:
    for threads in thread_counts:
        py_time = df[(df['Program'] == 'PYTHON') & (df['Matrix'] == matrix_size) & (df['Threads'] == threads)]['Avg'].values[0]
        cpp_time = df[(df['Program'] == 'CPP') & (df['Matrix'] == matrix_size) & (df['Threads'] == threads)]['Avg'].values[0]
        speedup = py_time / cpp_time
        speedup_data.append({'Matrix': matrix_size, 'Threads': threads, 'Speedup': speedup})

speedup_df = pd.DataFrame(speedup_data)

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(matrix_sizes))
width = 0.15

# Define distinct colors for each thread count
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for idx, threads in enumerate(thread_counts):
    data = speedup_df[speedup_df['Threads'] == threads]
    offset = width * (idx - len(thread_counts)/2 + 0.5)
    bars = ax.bar(x + offset, data['Speedup'], width, label=f'{threads} threads', 
                  color=colors[idx % len(colors)], alpha=0.8)

ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Break-even (1x)')
ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup (Python Time / CPP Time)', fontsize=12, fontweight='bold')
ax.set_title('CPP Speedup Over Python\n(Above 1.0 = CPP Faster, Below 1.0 = Python Faster)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(matrix_sizes)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / 'plot4_cpp_speedup.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot4_cpp_speedup.png")
plt.close()

# ============================================================================
# PLOT 5: Strong Scaling Analysis
# ============================================================================
print("\n📊 Generating Plot 5: Strong Scaling Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Strong Scaling: Speedup vs Threads (baseline = 1 thread)', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        baseline = data[data['Threads'] == 1]['Avg'].values[0]
        speedups = baseline / data['Avg']
        
        ax.plot(data['Threads'], speedups, marker='o', linewidth=2, markersize=8, label=program)
    
    # Ideal linear scaling
    ax.plot(thread_counts, thread_counts, 'k--', linewidth=2, alpha=0.5, label='Ideal Linear')
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup vs 1 Thread')
    ax.set_title(f'N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thread_counts)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(plots_dir / 'plot5_strong_scaling.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot5_strong_scaling.png")
plt.close()

# ============================================================================
# PLOT 6: Performance Heatmaps (PYTHON vs CPP)
# ============================================================================
print("\n📊 Generating Plot 6: Performance Heatmaps...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Performance Heatmaps: Time (seconds)', fontsize=16, fontweight='bold')

for ax, program in zip([ax1, ax2], ['PYTHON', 'CPP']):
    pivot = df[df['Program'] == program].pivot(index='Threads', columns='Matrix', values='Avg')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Time (s)'})
    ax.set_title(f'{program} Performance', fontweight='bold')
    ax.set_xlabel('Matrix Size (N)')
    ax.set_ylabel('Number of Threads')

plt.tight_layout()
plt.savefig(plots_dir / 'plot6_heatmaps.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot6_heatmaps.png")
plt.close()

# ============================================================================
# PLOT 7: GFLOPS Comparison
# ============================================================================
print("\n📊 Generating Plot 7: GFLOPS Performance...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Computational Throughput (GFLOPS)', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        ax.plot(data['Threads'], data['GFLOPS'], marker='D', linewidth=2, markersize=8, label=program)
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thread_counts)

plt.tight_layout()
plt.savefig(plots_dir / 'plot7_gflops.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot7_gflops.png")
plt.close()

# ============================================================================
# PLOT 8: Parallel Efficiency
# ============================================================================
print("\n📊 Generating Plot 8: Parallel Efficiency...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parallel Efficiency (%)', fontsize=16, fontweight='bold')

for idx, matrix_size in enumerate(matrix_sizes):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        baseline = data[data['Threads'] == 1]['Avg'].values[0]
        speedups = baseline / data['Avg']
        efficiency = (speedups / data['Threads']) * 100
        
        ax.plot(data['Threads'], efficiency, marker='o', linewidth=2, markersize=8, label=program)
    
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% (ideal)')
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% (good)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (poor)')
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'N = {matrix_size}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thread_counts)
    ax.set_ylim([0, 110])

plt.tight_layout()
plt.savefig(plots_dir / 'plot8_parallel_efficiency.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot8_parallel_efficiency.png")
plt.close()

# ============================================================================
# PLOT 9: Standard Deviation Analysis (as percentage of mean)
# ============================================================================
print("\n📊 Generating Plot 9: Standard Deviation vs Matrix Size...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Measurement Variability (Relative StdDev %)', fontsize=16, fontweight='bold')

# By matrix size - using percentage (StdDev/Avg * 100)
for program in ['PYTHON', 'CPP']:
    data = df[df['Program'] == program].groupby('Matrix')[['StdDev', 'Avg']].apply(
        lambda x: (x['StdDev'] / x['Avg'] * 100).mean()
    )
    ax1.plot(data.index, data.values, marker='o', linewidth=2, markersize=8, label=program)

ax1.set_xlabel('Matrix Size (N)')
ax1.set_ylabel('Average Relative StdDev (%)')
ax1.set_title('Relative StdDev vs Matrix Size')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# By thread count - using percentage (StdDev/Avg * 100)
for program in ['PYTHON', 'CPP']:
    data = df[df['Program'] == program].groupby('Threads')[['StdDev', 'Avg']].apply(
        lambda x: (x['StdDev'] / x['Avg'] * 100).mean()
    )
    ax2.plot(data.index, data.values, marker='s', linewidth=2, markersize=8, label=program)

ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Average Relative StdDev (%)')
ax2.set_title('Relative StdDev vs Thread Count')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(thread_counts)

plt.tight_layout()
plt.savefig(plots_dir / 'plot9_stddev_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot9_stddev_analysis.png")
plt.close()

# ============================================================================
# PLOT 10: Crossover Analysis at N=4096
# ============================================================================
print("\n📊 Generating Plot 10: Critical Crossover Analysis at N=4096...")
fig, ax = plt.subplots(figsize=(12, 7))

n_4096_data = df[df['Matrix'] == 4096]
width = 0.35
x = np.arange(len(thread_counts))

py_times = n_4096_data[n_4096_data['Program'] == 'PYTHON']['Avg'].values
cpp_times = n_4096_data[n_4096_data['Program'] == 'CPP']['Avg'].values

bars1 = ax.bar(x - width/2, py_times, width, label='PYTHON', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, cpp_times, width, label='CPP', color='red', alpha=0.7)

ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('CRITICAL: Performance at N=4096\n(CPP becomes SLOWER than Python!)', fontsize=14, fontweight='bold', color='red')
ax.set_xticks(x)
ax.set_xticklabels(thread_counts)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add speedup annotations
for i, (py, cpp) in enumerate(zip(py_times, cpp_times)):
    speedup = py / cpp
    color = 'green' if speedup > 1 else 'red'
    ax.text(i, max(py, cpp) + 0.1, f'{speedup:.2f}x', ha='center', fontweight='bold', color=color, fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / 'plot10_crossover_n4096.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot10_crossover_n4096.png")
plt.close()

# ============================================================================
# PLOT 11: Best Configurations Summary
# ============================================================================
print("\n📊 Generating Plot 11: Best Configurations Summary...")
best_configs = []
for matrix_size in matrix_sizes:
    for program in ['PYTHON', 'CPP']:
        data = df[(df['Program'] == program) & (df['Matrix'] == matrix_size)]
        best = data.loc[data['Avg'].idxmin()]
        best_configs.append({
            'Program': program,
            'Matrix': matrix_size,
            'Best_Threads': int(best['Threads']),
            'Time': best['Avg'],
            'GFLOPS': best['GFLOPS']
        })

best_df = pd.DataFrame(best_configs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Best Configuration for Each Matrix Size', fontsize=16, fontweight='bold')

# Best time
for program in ['PYTHON', 'CPP']:
    data = best_df[best_df['Program'] == program]
    ax1.plot(data['Matrix'], data['Time'], marker='o', linewidth=2, markersize=10, label=program)
    for _, row in data.iterrows():
        ax1.annotate(f"T={row['Best_Threads']}", 
                    (row['Matrix'], row['Time']),
                    textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=8)

ax1.set_xlabel('Matrix Size (N)')
ax1.set_ylabel('Best Time (seconds)')
ax1.set_title('Best Time Achieved')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')

# Best GFLOPS
for program in ['PYTHON', 'CPP']:
    data = best_df[best_df['Program'] == program]
    ax2.plot(data['Matrix'], data['GFLOPS'], marker='D', linewidth=2, markersize=10, label=program)
    for _, row in data.iterrows():
        ax2.annotate(f"T={row['Best_Threads']}", 
                    (row['Matrix'], row['GFLOPS']),
                    textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=8)

ax2.set_xlabel('Matrix Size (N)')
ax2.set_ylabel('Peak GFLOPS')
ax2.set_title('Peak GFLOPS Achieved')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig(plots_dir / 'plot11_best_configs.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot11_best_configs.png")
plt.close()

# ============================================================================
# PLOT 12: Complete Speedup Map
# ============================================================================
print("\n📊 Generating Plot 12: Complete Speedup Map...")
speedup_pivot = speedup_df.pivot(index='Threads', columns='Matrix', values='Speedup')

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(speedup_pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0, 
            ax=ax, cbar_kws={'label': 'Speedup (Python/CPP)'}, 
            vmin=0, vmax=speedup_pivot.max().max())
ax.set_title('CPP Speedup Over Python\n(Green = CPP Faster, Red = Python Faster)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Matrix Size (N)', fontsize=12)
ax.set_ylabel('Number of Threads', fontsize=12)

plt.tight_layout()
plt.savefig(plots_dir / 'plot12_speedup_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: plots/plot12_speedup_heatmap.png")
plt.close()

# ============================================================================
# COMPREHENSIVE ANALYSIS & INSIGHTS
# ============================================================================
print("\n" + "=" * 100)
print(" " * 40 + "KEY INSIGHTS & ANALYSIS")
print("=" * 100)

print("\n🔍 1. PERFORMANCE CROSSOVER PHENOMENON (CRITICAL FINDING):")
print("-" * 100)
print("   At N=4096, CPP implementation becomes SLOWER than Python!")
print("   ")
for threads in thread_counts:
    py_time = df[(df['Program'] == 'PYTHON') & (df['Matrix'] == 4096) & (df['Threads'] == threads)]['Avg'].values[0]
    cpp_time = df[(df['Program'] == 'CPP') & (df['Matrix'] == 4096) & (df['Threads'] == threads)]['Avg'].values[0]
    speedup = py_time / cpp_time
    winner = "✅ CPP FASTER" if speedup > 1 else "❌ PYTHON FASTER"
    print(f"   Threads={threads}: Python={py_time:.3f}s, CPP={cpp_time:.3f}s, Speedup={speedup:.2f}x  {winner}")

print("\n   💡 INFERENCE:")
print("      • CPP is faster for N ≤ 2048 (up to 35x speedup)")
print("      • At N=4096 with T≤2, Python/NumPy DOMINATES")
print("      • CPP implementation has severe cache/memory issues at large N")
print("      • Block sizes likely too large for cache hierarchy at N=4096")

print("\n🚀 2. BEST PERFORMANCE ACHIEVED:")
print("-" * 100)
for matrix_size in matrix_sizes:
    py_data = df[(df['Program'] == 'PYTHON') & (df['Matrix'] == matrix_size)]
    cpp_data = df[(df['Program'] == 'CPP') & (df['Matrix'] == matrix_size)]
    
    best_py = py_data.loc[py_data['Avg'].idxmin()]
    best_cpp = cpp_data.loc[cpp_data['Avg'].idxmin()]
    
    py_gflops = (2 * matrix_size**3 / best_py['Avg']) / 1e9
    cpp_gflops = (2 * matrix_size**3 / best_cpp['Avg']) / 1e9
    
    print(f"   N={matrix_size}:")
    print(f"      Python: {best_py['Avg']:.4f}s @ {int(best_py['Threads'])}T → {py_gflops:.2f} GFLOPS")
    print(f"      CPP:    {best_cpp['Avg']:.4f}s @ {int(best_cpp['Threads'])}T → {cpp_gflops:.2f} GFLOPS")
    if best_cpp['Avg'] < best_py['Avg']:
        speedup = best_py['Avg'] / best_cpp['Avg']
        print(f"      ⚡ CPP is {speedup:.1f}x FASTER")
    else:
        speedup = best_cpp['Avg'] / best_py['Avg']
        print(f"      ⚠️  Python is {speedup:.1f}x FASTER (CPP optimization needed!)")

print("\n📊 3. SCALING EFFICIENCY ANALYSIS:")
print("-" * 100)
print("   Strong Scaling (8 threads vs 1 thread):")
print("   ")
for matrix_size in matrix_sizes:
    for program in ['PYTHON', 'CPP']:
        time_1t = df[(df['Program'] == program) & (df['Matrix'] == matrix_size) & (df['Threads'] == 1)]['Avg'].values[0]
        time_8t = df[(df['Program'] == program) & (df['Matrix'] == matrix_size) & (df['Threads'] == 8)]['Avg'].values[0]
        speedup = time_1t / time_8t
        efficiency = (speedup / 8) * 100
        
        if efficiency > 75:
            status = "🟢 EXCELLENT"
        elif efficiency > 50:
            status = "🟡 GOOD"
        elif efficiency > 25:
            status = "🟠 POOR"
        else:
            status = "🔴 VERY POOR"
        
        print(f"   {program:7s} @ N={matrix_size}: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency {status}")

print("\n   💡 INFERENCE:")
print("      • Python shows good scaling across all sizes")
print("      • CPP shows excellent scaling for small N but degrades at N=4096")
print("      • CPP has synchronization/memory bottlenecks at large matrices")

print("\n📈 4. COEFFICIENT OF VARIATION (STABILITY):")
print("-" * 100)
py_avg_cv = df[df['Program'] == 'PYTHON']['CV'].mean()
cpp_avg_cv = df[df['Program'] == 'CPP']['CV'].mean()
print(f"   Python Average CV: {py_avg_cv:.2f}%  {'🟢 STABLE' if py_avg_cv < 10 else '🟡 MODERATE'}")
print(f"   CPP Average CV:    {cpp_avg_cv:.2f}%  {'🟢 STABLE' if cpp_avg_cv < 10 else '🟡 MODERATE'}")

unstable = df[df['CV'] > 10].sort_values('CV', ascending=False)
if len(unstable) > 0:
    print("\n   Most unstable configurations (CV > 10%):")
    for _, row in unstable.head(5).iterrows():
        print(f"      {row['Program']:7s} N={int(row['Matrix']):4d} T={int(row['Threads'])} → CV={row['CV']:.2f}%")

print("\n⚡ 5. GFLOPS PERFORMANCE:")
print("-" * 100)
max_gflops_py = df[df['Program'] == 'PYTHON']['GFLOPS'].max()
max_gflops_cpp = df[df['Program'] == 'CPP']['GFLOPS'].max()
max_py_row = df[df['GFLOPS'] == max_gflops_py].iloc[0]
max_cpp_row = df[df['GFLOPS'] == max_gflops_cpp].iloc[0]

print(f"   Peak Python: {max_gflops_py:.2f} GFLOPS @ N={int(max_py_row['Matrix'])}, T={int(max_py_row['Threads'])}")
print(f"   Peak CPP:    {max_gflops_cpp:.2f} GFLOPS @ N={int(max_cpp_row['Matrix'])}, T={int(max_cpp_row['Threads'])}")

if max_gflops_py > max_gflops_cpp:
    print(f"   ⚠️  Python achieves {max_gflops_py/max_gflops_cpp:.1f}x higher GFLOPS than CPP!")
else:
    print(f"   ✅ CPP achieves {max_gflops_cpp/max_gflops_py:.1f}x higher GFLOPS than Python!")

print("\n❌ 6. WHY IS CPP SLOW AT N=4096? (ROOT CAUSE ANALYSIS)")
print("-" * 100)
print("   Likely causes:")
print("      1. 🧱 BLOCK SIZE: Current blocks may exceed cache capacity")
print("      2. 🔄 MEMORY PATTERN: Poor spatial/temporal locality at large N")
print("      3. 🧵 THREAD OVERHEAD: OpenMP overhead dominates for large N")
print("      4. ⚙️  VECTORIZATION: Insufficient SIMD utilization")
print("      5. 🚫 NO DATA PACKING: Raw matrices cause cache thrashing")
print("      6. 📊 PREFETCH: May not be effective at large strides")

print("\n🔧 7. RECOMMENDED OPTIMIZATIONS:")
print("-" * 100)
print("   Immediate fixes:")
print("      1. REDUCE BLOCK SIZES for better cache fit")
print("      2. IMPLEMENT PANEL PACKING (copy B matrix)")
print("      3. USE SMALLER MICRO-KERNEL (8x6 or 12x4)")
print("      4. ADAPTIVE PREFETCH DISTANCE based on N")
print("      5. DYNAMIC SCHEDULING for load balancing")
print("      6. Profile with perf/vtune to identify bottlenecks")

print("\n" + "=" * 100)
print("📁 All 12 plots saved in 'plots/' directory")
print("=" * 100)

print("\n✅ VISUALIZATION COMPLETE!")
print("🎯 Review the plots to understand performance characteristics")
print("⚠️  CRITICAL ACTION NEEDED: Optimize CPP implementation for N=4096")
