import csv
import math
import os

def read_results(csv_path):
    """
    Reads the benchmark results CSV and parses it into a dictionary:
    {
        kernel_name: {
            N: {
                'time': float,
                'speedup': float,
                'error': float,
                'gflops': float,
                'bandwidth': float,
                'memory': float
            }
        }
    }
    """
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row['N'])
            kernel = row['Kernel']
            if kernel not in data:
                data[kernel] = {}
            data[kernel][N] = {
                'time': float(row['AvgTimeMS']),
                'speedup': float(row['Speedup']),
                'error': float(row['MaxError']),
                'gflops': float(row['GFLOPs']),
                'bandwidth': float(row['BandwidthGBs']),
                'memory': float(row['MemoryMB'])
            }
    return data

def generate_runtime_svg(data, output_path):
    """
    Generates a log-scale SVG of Runtime vs Sequence Length
    """
    kernels = ['CPU', 'Naive', 'Tiled', 'Fused']
    sequence_lengths = [256, 512, 1024, 2048, 4096]
    
    # Modern premium palette
    colors = {
        'CPU': '#7F8C8D',    # Cool slate gray
        'Naive': '#E74C3C',  # Vibrant Coral/Red
        'Tiled': '#3498DB',  # Sleek Electric Blue
        'Fused': '#2ECC71'   # Brilliant Mint Green
    }
    
    width = 800
    height = 500
    margin_left = 90
    margin_right = 180
    margin_top = 70
    margin_bottom = 70
    
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    # Calculate log10 min/max for y-axis
    # Let's cover from 0.01 ms to 100,000 ms
    log_min = -2.0  # 10^-2 = 0.01 ms
    log_max = 5.0   # 10^5 = 100,000 ms
    
    # Helper to map (index, value) to (x, y) pixels
    def map_coords(x_idx, time_ms):
        # x is spaced equally for categorical sequence lengths (log2 scaling)
        x = margin_left + x_idx * (plot_w / 4)
        
        # log10 scale for y
        log_val = math.log10(max(time_ms, 0.001))
        # clamp to log_min, log_max
        log_val = max(log_min, min(log_max, log_val))
        
        y = margin_top + plot_h - ((log_val - log_min) / (log_max - log_min)) * plot_h
        return x, y

    svg = []
    # Root SVG with styles
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
    svg.append('<defs>')
    # Glow filter for Fused FlashAttention line to make it pop!
    svg.append('''  <filter id="fused-glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>''')
    svg.append('</defs>')
    
    # Premium background styling
    svg.append(f'<rect width="{width}" height="{height}" fill="#111827" rx="12" />') # Dark mode theme
    
    # Grid lines and y-axis ticks (log10)
    for power in range(int(log_min), int(log_max) + 1):
        val = 10**power
        # Get y position
        _, y = map_coords(0, val)
        
        # Grid line
        svg.append(f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_w}" y2="{y}" stroke="#374151" stroke-dasharray="3,3" stroke-width="1" />')
        
        # Y Label
        label = f"{val} ms"
        if val >= 1000:
            label = f"{val // 1000} s"
        elif val == 0.01:
            label = "10 µs"
        elif val == 0.1:
            label = "100 µs"
            
        svg.append(f'  <text x="{margin_left - 12}" y="{y + 4}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" text-anchor="end">{label}</text>')

    # X-axis ticks
    for i, N in enumerate(sequence_lengths):
        x = margin_left + i * (plot_w / 4)
        # Vertical grid line
        svg.append(f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="0.5" />')
        # Label
        svg.append(f'  <text x="{x}" y="{margin_top + plot_h + 24}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle" font-weight="500">N={N}</text>')

    # Chart Title
    svg.append(f'  <text x="{margin_left}" y="38" fill="#F9FAFB" font-family="system-ui, sans-serif" font-size="18" font-weight="700">Attention Runtime Scaling (Log Scale)</text>')
    svg.append(f'  <text x="{margin_left}" y="54" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11">Sequence Length vs. Average Execution Time (d=64, lower is better)</text>')

    # Draw curves
    for k in kernels:
        if k not in data:
            continue
        
        points = []
        for i, N in enumerate(sequence_lengths):
            if N in data[k]:
                x, y = map_coords(i, data[k][N]['time'])
                points.append((x, y))
        
        # Write polyline
        if points:
            path_d = f"M {points[0][0]},{points[0][1]} " + " ".join(f"L {p[0]},{p[1]}" for p in points[1:])
            stroke_w = 4 if k == 'Fused' else 2.5
            filter_str = ' filter="url(#fused-glow)"' if k == 'Fused' else ''
            svg.append(f'  <path d="{path_d}" fill="none" stroke="{colors[k]}" stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round"{filter_str} />')
            
            # Draw point markers
            for p in points:
                svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="5" fill="#111827" stroke="{colors[k]}" stroke-width="2.5" />')
                if k == 'Fused':
                    svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="2" fill="#2ECC71" />')

    # Legend
    legend_x = width - margin_right + 25
    legend_y = margin_top + 10
    svg.append(f'  <rect x="{legend_x - 10}" y="{legend_y - 10}" width="150" height="150" fill="#1F2937" rx="8" stroke="#374151" stroke-width="1" />')
    
    for idx, k in enumerate(kernels):
        y_pos = legend_y + idx * 30
        svg.append(f'  <rect x="{legend_x}" y="{y_pos - 6}" width="14" height="14" fill="{colors[k]}" rx="3" />')
        svg.append(f'  <text x="{legend_x + 22}" y="{y_pos + 5}" fill="#F3F4F6" font-family="system-ui, sans-serif" font-size="12" font-weight="600">{k} Kernel</text>')

    # Axis Title
    svg.append(f'  <text x="{margin_left - 50}" y="{margin_top - 15}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" font-weight="600">Runtime (ms)</text>')
    
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(svg))
    print(f"Generated {output_path} successfully.")

def generate_speedup_svg(data, output_path):
    """
    Generates a linear SVG of Speedup vs CPU
    """
    # Exclude CPU since CPU speedup is always 1.0x
    gpu_kernels = ['Naive', 'Tiled', 'Fused']
    sequence_lengths = [256, 512, 1024, 2048, 4096]
    
    colors = {
        'Naive': '#E74C3C',
        'Tiled': '#3498DB',
        'Fused': '#2ECC71'
    }
    
    # Let's read all speedups to determine max value dynamically
    max_speedup = 1.0
    for k in gpu_kernels:
        if k in data:
            for N in sequence_lengths:
                if N in data[k]:
                    max_speedup = max(max_speedup, data[k][N]['speedup'])
    
    # Add a buffer margin to max speedup (round up to nearest 100 or 50)
    round_to = 200 if max_speedup > 500 else (50 if max_speedup > 100 else 10)
    y_max = math.ceil(max_speedup / round_to) * round_to
    # Ensure it's at least 100
    y_max = max(100, y_max)
    
    width = 800
    height = 500
    margin_left = 90
    margin_right = 180
    margin_top = 70
    margin_bottom = 70
    
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    def map_coords(x_idx, speedup):
        x = margin_left + x_idx * (plot_w / 4)
        y = margin_top + plot_h - (speedup / y_max) * plot_h
        return x, y

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
    svg.append('<defs>')
    svg.append('''  <filter id="speedup-glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>''')
    svg.append('</defs>')
    
    svg.append(f'<rect width="{width}" height="{height}" fill="#111827" rx="12" />')

    # Y grid lines
    num_ticks = 5
    for i in range(num_ticks + 1):
        val = int(i * (y_max / num_ticks))
        _, y = map_coords(0, val)
        
        svg.append(f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_w}" y2="{y}" stroke="#374151" stroke-dasharray="3,3" stroke-width="1" />')
        svg.append(f'  <text x="{margin_left - 12}" y="{y + 4}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" text-anchor="end">{val}x</text>')

    # X-axis ticks
    for i, N in enumerate(sequence_lengths):
        x = margin_left + i * (plot_w / 4)
        svg.append(f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="0.5" />')
        svg.append(f'  <text x="{x}" y="{margin_top + plot_h + 24}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle" font-weight="500">N={N}</text>')

    # Chart Title
    svg.append(f'  <text x="{margin_left}" y="38" fill="#F9FAFB" font-family="system-ui, sans-serif" font-size="18" font-weight="700">GPU Speedup vs. CPU Baseline</text>')
    svg.append(f'  <text x="{margin_left}" y="54" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11">Sequence Length vs. Speedup Factor (higher is better)</text>')

    # Draw curves
    for k in gpu_kernels:
        if k not in data:
            continue
        
        points = []
        for i, N in enumerate(sequence_lengths):
            if N in data[k]:
                x, y = map_coords(i, data[k][N]['speedup'])
                points.append((x, y))
        
        if points:
            path_d = f"M {points[0][0]},{points[0][1]} " + " ".join(f"L {p[0]},{p[1]}" for p in points[1:])
            stroke_w = 4 if k == 'Fused' else 2.5
            filter_str = ' filter="url(#speedup-glow)"' if k == 'Fused' else ''
            svg.append(f'  <path d="{path_d}" fill="none" stroke="{colors[k]}" stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round"{filter_str} />')
            
            # Point labels for the final point to show exactly the max speedup achieved!
            final_N = sequence_lengths[-1]
            if final_N in data[k]:
                final_x, final_y = map_coords(len(sequence_lengths)-1, data[k][final_N]['speedup'])
                lbl = f"{data[k][final_N]['speedup']:.1f}x"
                # Shift slightly above/below depending on kernel to avoid overlap
                y_offset = -12 if k == 'Fused' else (15 if k == 'Tiled' else 12)
                svg.append(f'  <text x="{final_x + 8}" y="{final_y + y_offset}" fill="{colors[k]}" font-family="system-ui, sans-serif" font-size="12" font-weight="700" text-anchor="start">{lbl}</text>')

            for p in points:
                svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="5" fill="#111827" stroke="{colors[k]}" stroke-width="2.5" />')
                if k == 'Fused':
                    svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="2" fill="#2ECC71" />')

    # Legend
    legend_x = width - margin_right + 25
    legend_y = margin_top + 10
    svg.append(f'  <rect x="{legend_x - 10}" y="{legend_y - 10}" width="150" height="120" fill="#1F2937" rx="8" stroke="#374151" stroke-width="1" />')
    
    for idx, k in enumerate(gpu_kernels):
        y_pos = legend_y + idx * 30
        svg.append(f'  <rect x="{legend_x}" y="{y_pos - 6}" width="14" height="14" fill="{colors[k]}" rx="3" />')
        svg.append(f'  <text x="{legend_x + 22}" y="{y_pos + 5}" fill="#F3F4F6" font-family="system-ui, sans-serif" font-size="12" font-weight="600">{k} GPU</text>')

    # Axis Title
    svg.append(f'  <text x="{margin_left - 50}" y="{margin_top - 15}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" font-weight="600">Speedup Factor</text>')
    
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(svg))
    print(f"Generated {output_path} successfully.")

def generate_memory_svg(data, output_path):
    """
    Generates a linear SVG of Memory Scaling vs Sequence Length
    """
    sequence_lengths = [256, 512, 1024, 2048, 4096]
    
    colors = {
        'Standard': '#E74C3C', # Red for standard memory explosion
        'Flash': '#2ECC71'     # Green for FlashAttention on-chip flat cost
    }
    
    # Calculate Standard vs Flash memory
    # Standard: 2 * N^2 * 4 bytes
    standard_mem = [2.0 * N * N * 4.0 / (1024.0 * 1024.0) for N in sequence_lengths] # MB
    # FlashAttention avoids materializing NxN, so it only requires memory to store Q, K, V, O which is 4 * N * d * 4 bytes
    # But even better: during execution, standard attention materializes 2 NxN matrices. FlashAttention keeps them in registers/SRAM,
    # meaning its *intermediate* materialized memory in HBM is literally 0.0 MB!
    flash_mem = [0.0 for _ in sequence_lengths]
    
    y_max = 128.0 # up to 128 MB for N=4096
    
    width = 800
    height = 500
    margin_left = 90
    margin_right = 180
    margin_top = 70
    margin_bottom = 70
    
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    def map_coords(x_idx, mem_mb):
        x = margin_left + x_idx * (plot_w / 4)
        y = margin_top + plot_h - (mem_mb / y_max) * plot_h
        return x, y

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
    svg.append('<defs>')
    svg.append('''  <filter id="flash-glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>''')
    svg.append('</defs>')
    
    svg.append(f'<rect width="{width}" height="{height}" fill="#111827" rx="12" />')

    # Y grid lines
    num_ticks = 4
    for i in range(num_ticks + 1):
        val = float(i * (y_max / num_ticks))
        _, y = map_coords(0, val)
        
        svg.append(f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_w}" y2="{y}" stroke="#374151" stroke-dasharray="3,3" stroke-width="1" />')
        svg.append(f'  <text x="{margin_left - 12}" y="{y + 4}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" text-anchor="end">{int(val)} MB</text>')

    # X-axis ticks
    for i, N in enumerate(sequence_lengths):
        x = margin_left + i * (plot_w / 4)
        svg.append(f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="0.5" />')
        svg.append(f'  <text x="{x}" y="{margin_top + plot_h + 24}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle" font-weight="500">N={N}</text>')

    # Chart Title
    svg.append(f'  <text x="{margin_left}" y="38" fill="#F9FAFB" font-family="system-ui, sans-serif" font-size="18" font-weight="700">Attention Intermediate Memory Footprint</text>')
    svg.append(f'  <text x="{margin_left}" y="54" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11">Sequence Length vs. Physical NxN Attention Matrix Materialization in HBM (lower is better)</text>')

    # Standard curve (O(N^2) explosion)
    points_std = []
    for i, val in enumerate(standard_mem):
        points_std.append(map_coords(i, val))
        
    path_std = f"M {points_std[0][0]},{points_std[0][1]} " + " ".join(f"L {p[0]},{p[1]}" for p in points_std[1:])
    svg.append(f'  <path d="{path_std}" fill="none" stroke="{colors["Standard"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />')
    
    # Label the standard peak
    svg.append(f'  <text x="{points_std[-1][0] - 12}" y="{points_std[-1][1] - 12}" fill="{colors["Standard"]}" font-family="system-ui, sans-serif" font-size="12" font-weight="700" text-anchor="end">128 MB (O(N²))</text>')

    for p in points_std:
        svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="5" fill="#111827" stroke="{colors["Standard"]}" stroke-width="2.5" />')

    # Flash curve (O(1) intermediate matrix footprint!)
    points_flash = []
    for i, val in enumerate(flash_mem):
        points_flash.append(map_coords(i, val))
        
    path_flash = f"M {points_flash[0][0]},{points_flash[0][1]} " + " ".join(f"L {p[0]},{p[1]}" for p in points_flash[1:])
    svg.append(f'  <path d="{path_flash}" fill="none" stroke="{colors["Flash"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#flash-glow)" />')
    svg.append(f'  <text x="{points_flash[-1][0] - 12}" y="{points_flash[-1][1] - 12}" fill="{colors["Flash"]}" font-family="system-ui, sans-serif" font-size="12" font-weight="700" text-anchor="end">0 MB (SRAM Fused)</text>')

    for p in points_flash:
        svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="5" fill="#111827" stroke="{colors["Flash"]}" stroke-width="2.5" />')
        svg.append(f'  <circle cx="{p[0]}" cy="{p[1]}" r="2" fill="#2ECC71" />')

    # Legend
    legend_x = width - margin_right + 25
    legend_y = margin_top + 10
    svg.append(f'  <rect x="{legend_x - 10}" y="{legend_y - 10}" width="150" height="90" fill="#1F2937" rx="8" stroke="#374151" stroke-width="1" />')
    
    # Standard legend
    svg.append(f'  <rect x="{legend_x}" y="{legend_y - 6}" width="14" height="14" fill="{colors["Standard"]}" rx="3" />')
    svg.append(f'  <text x="{legend_x + 22}" y="{legend_y + 5}" fill="#F3F4F6" font-family="system-ui, sans-serif" font-size="11" font-weight="600">Standard (S &amp; P)</text>')
    
    # Flash legend
    svg.append(f'  <rect x="{legend_x}" y="{legend_y + 24}" width="14" height="14" fill="{colors["Flash"]}" rx="3" />')
    svg.append(f'  <text x="{legend_x + 22}" y="{legend_y + 35}" fill="#F3F4F6" font-family="system-ui, sans-serif" font-size="11" font-weight="600">FlashAttention</text>')

    # Axis Title
    svg.append(f'  <text x="{margin_left - 50}" y="{margin_top - 15}" fill="#9CA3AF" font-family="system-ui, sans-serif" font-size="11" font-weight="600">Memory Consumed</text>')
    
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(svg))
    print(f"Generated {output_path} successfully.")

if __name__ == '__main__':
    csv_path = 'benchmarks/result/benchmark_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Run the attention_engine benchmark first!")
        exit(1)
        
    results = read_results(csv_path)
    
    # Create output directories if needed
    os.makedirs('benchmarks/result/plot', exist_ok=True)
    
    generate_runtime_svg(results, 'benchmarks/result/plot/runtime_scaling.svg')
    generate_speedup_svg(results, 'benchmarks/result/plot/speedup_comparison.svg')
    generate_memory_svg(results, 'benchmarks/result/plot/memory_scaling.svg')
    print("All benchmark visualizations successfully generated!")
