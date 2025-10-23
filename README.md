# Run online in a sandbox at the following link:
https://codehs.com/sandbox/id/python-graphics-tkinter-YhtGaX/run
Once the page loads, click the green run button, and the script will run

Row Reduction (Gaussian Elimination) - Exact Fractions

Overview
This is a Python Tkinter GUI application that performs row reduction (Gaussian elimination) using exact arithmetic with Python's Fraction class. It records and animates each elementary row operation and allows stepping through the process. Original row numbers are displayed and remain associated with rows even after swaps.

Features
- Exact arithmetic: All numbers are Fractions. No floating point.
- Input: Enter rows as comma-separated values. Supports integers, a/b, and decimals (converted to exact Fraction).
- Step recording: Every swap, scaling, and elimination is recorded.
- Animation: Play/Pause stepping through operations with adjustable speed.
- Manual control: Prev/Next and Reset to navigate steps.
- Persistent row labels: Original row numbers (r1, r2, ...) stay with their rows when swapped.

Requirements
- Python 3.10+

Run
1. Open a terminal in this folder.
2. Run:
   
   ```bash
   python main.py
   ```

Usage
1. Enter each matrix row in the input field as comma-separated values, then click "Add Row" (or press Enter).
   - Examples: `1, -2, 3/5` or `2, 4, 6` or `1.25, -0.5`.
2. After adding all rows, click "Solve".
3. Use "Prev", "Next", or "▶ Play" to step/animate through the operations.
4. Adjust animation speed (milliseconds) in the box to the right of the controls.
5. "Reset Steps" returns to the initial matrix view.

Notes
- The algorithm computes RREF and records steps at each elementary operation.
- Decimals like 0.1 are parsed as exact Fractions (1/10), not binary floats.
- Column headers are labeled c1, c2, ... and rows show their original labels r1, r2, ... throughout.


