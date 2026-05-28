# Benchmarks Folder

This folder contains the scripts and the folders where we save our test results. We want to see how fast our GPU code is compared to our normal CPU code.

### What is inside here?
* **plot_benchmarks.py**: A Python script that takes the numbers from our GPU tests and turns them into nice graphs.
* **result/**: A folder that holds the CSV files with the raw test numbers and another folder inside called `plot` where the SVG graphs are saved.
* **python_cpu_plot/**: A folder that holds the graph for the CPU-only test.

### How to use this
1. Run `./build/attention_engine` to do the GPU tests. This will create the CSV file.
2. Run `python benchmarks/plot_benchmarks.py` to make the comparison graphs.
