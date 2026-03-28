import pandas as pd
from plotnine import geom_rect, ggplot, aes, geom_line, geom_point, labs, scale_x_discrete, scale_x_log10, scale_y_log10, theme, theme_minimal, geom_vline, geom_abline, annotate, element_rect, theme
import numpy as np


df = pd.read_csv('benchmark_results.csv', header=None, 
                 names=['N_grid', 'W_type', 'lu_mem_mib', 't_apply'])

df['W_type'] = df['W_type'].astype(int).astype(str)
df['N_grid_cat'] = pd.Categorical(df['N_grid'].astype(int).astype(str), 
                                  categories=df['N_grid'].astype(int).astype(str).unique(), 
                                  ordered=True)
unique_n = df['N_grid'].unique()
boundary_val = 500 
split_idx = np.searchsorted(unique_n, boundary_val)+1

plot_time = (
    ggplot(df, aes(x='N_grid_cat', y='t_apply', color='W_type', group='W_type'))

    + geom_rect(aes(xmin=-np.inf, xmax=split_idx, ymin=0, ymax=np.inf), 
                fill="#f0f9ff", color=None, alpha=0.01, inherit_aes=False)
    + geom_rect(aes(xmin=split_idx, xmax=np.inf, ymin=0, ymax=np.inf), 
                fill="#fff7ed", color=None, alpha=0.01, inherit_aes=False)
    
    # --- Main Plot ---
    + geom_vline(xintercept=split_idx, linetype='dashed', color='grey')
    + geom_line(size=1)
    + geom_point(size=3)
    
    # --- Annotations ---
    + annotate("text", x=1.5, y=df['t_apply'].max()*0.9, label="Cache", color="#0369a1")
    + annotate("text", x=split_idx+1.5, y=df['t_apply'].max()*0.9, label="RAM", color="#9a3412")

    + labs(
        title='Preconditioner Time: solve Mx = y',
        subtitle='≈ x2 speedup once memory exceeds cache capacity',
        x='Grid Size',
        y='Time (ms)',
        color='iLU Precision (bits)'
    )
    
    # + scale_y_log10()
    + scale_x_discrete()
    + theme_minimal()
    + theme(
        panel_background=element_rect(fill='white', color='none'),
        plot_background=element_rect(fill='white', color='none')
    )
)

# 2. Plot: Memory usage of local LU factorizations (lu_mem_mib) vs Grid Size (N_grid)
plot_memory = (
    ggplot(df, aes(x='N_grid', y='lu_mem_mib', color='W_type', group='W_type'))
    + geom_line()
    + geom_point()
    + labs(
        title='Local LU Memory Consumption',
        subtitle='Impact of precision on factor storage',
        x='Grid Size (N)',
        y='Memory (MiB)',
        color='Precision (bits)'
    )
    + theme_minimal()
)

# To save:
plot_time.save("time_benchmark.png", dpi=450)
# plot_memory.save("memory_benchmark.png", dpi=300)
