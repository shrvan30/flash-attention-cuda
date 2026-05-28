# GPU Comparison Graphs

This folder has the SVG graphs that we make after running our benchmark tests. They help us compare the different ways of running the attention code:

* **runtime_scaling.svg**: Shows how many milliseconds each version takes as the sequence length gets bigger.
* **speedup_comparison.svg**: Shows how many times faster the GPU versions are compared to the basic CPU code.
* **memory_scaling.svg**: Shows how much memory the basic attention uses compared to the smart FlashAttention version (which uses almost no memory).

These graphs are made automatically when you run the `plot_benchmarks.py` script.
