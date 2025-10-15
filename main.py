import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction
from typing import List, Tuple, Dict, Any


def parse_fraction(token: str) -> Fraction:
	"""Parse a string token into a Fraction, allowing forms like -2, 3/5, 1.25."""
	s = token.strip()
	if not s:
		raise ValueError("Empty number token")
	try:
		# Fraction can parse integers, 'a/b', and decimals from strings exactly
		return Fraction(s)
	except Exception as exc:
		raise ValueError(f"Invalid number: {token}") from exc


def fraction_to_str(value: Fraction) -> str:
	"""Render Fraction in simplified exact form (no floats)."""
	if value.denominator == 1:
		return str(value.numerator)
	return f"{value.numerator}/{value.denominator}"


def clone_matrix(matrix: List[List[Fraction]]) -> List[List[Fraction]]:
	return [row.copy() for row in matrix]


def format_linear_expr(constant: Fraction, terms: List[Tuple[Fraction, str]]) -> str:
	"""Format an expression like constant + sum(coef*name) with clean signs.
	Examples: 0 and [(1,'t1')] -> "t1"; 3 and [(-2,'t1')] -> "3 - 2*t1"."""
	parts: List[str] = []
	if constant != 0 or not terms:
		parts.append(fraction_to_str(constant))
	first_term = (constant == 0)
	for coef, name in terms:
		if coef == 0:
			continue
		is_positive = coef > 0
		abs_coef = coef if coef >= 0 else -coef
		coef_str = fraction_to_str(abs_coef)
		term_core = name if coef_str == '1' else f"{coef_str}*{name}"
		if first_term:
			# no leading plus; show minus if negative
			parts.append(term_core if is_positive else f"-{term_core}")
			first_term = False
		else:
			parts.append(("+ " if is_positive else "- ") + term_core)
	return " ".join(parts) if parts else "0"


class GaussianEliminationStepper:
	"""Compute RREF with exact Fractions and record each elementary row operation as a step."""

	def __init__(self, matrix: List[List[Fraction]]):
		if not matrix or not matrix[0]:
			raise ValueError("Matrix must have at least one row and one column")
		self.initial_matrix: List[List[Fraction]] = clone_matrix(matrix)
		self.row_labels: List[int] = list(range(1, len(matrix) + 1))  # persistent original row ids
		self.steps: List[Dict[str, Any]] = []
		self._compute_steps()

	def _record_step(self, matrix: List[List[Fraction]], labels: List[int], description: str) -> None:
		self.steps.append({
			"matrix": clone_matrix(matrix),
			"row_labels": labels.copy(),
			"description": description,
		})

	def _compute_steps(self) -> None:
		matrix = clone_matrix(self.initial_matrix)
		labels = self.row_labels.copy()
		m = len(matrix)
		n = len(matrix[0])
		self._record_step(matrix, labels, "Initial matrix")

		pivot_row = 0
		for col in range(n):
			if pivot_row >= m:
				break

			# Find pivot in or below pivot_row
			pivot = None
			for r in range(pivot_row, m):
				if matrix[r][col] != 0:
					pivot = r
					break

			if pivot is None:
				continue  # no pivot in this column

			# Swap to bring pivot to pivot_row if needed
			if pivot != pivot_row:
				matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
				labels[pivot_row], labels[pivot] = labels[pivot], labels[pivot_row]
				self._record_step(
					matrix,
					labels,
					f"Swap rows r{labels[pivot]} ↆ r{labels[pivot_row]}"
				)

			# Scale pivot row to make leading 1
			lead = matrix[pivot_row][col]
			if lead != 1:
				scale = Fraction(1, 1) / lead
				matrix[pivot_row] = [v * scale for v in matrix[pivot_row]]
				self._record_step(
					matrix,
					labels,
					f"Scale row r{labels[pivot_row]} by {fraction_to_str(scale)}"
				)

			# Eliminate all other rows in this column (to achieve RREF)
			for r in range(m):
				if r == pivot_row:
					continue
				factor = matrix[r][col]
				if factor == 0:
					continue
				matrix[r] = [vr - factor * vp for vr, vp in zip(matrix[r], matrix[pivot_row])]
				desc_factor = fraction_to_str(factor)
				self._record_step(
					matrix,
					labels,
					f"R r{labels[r]} = R r{labels[r]} - ({desc_factor}) * R r{labels[pivot_row]}"
				)

			pivot_row += 1
			if pivot_row >= m:
				break


class RowReductionApp(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("Row Reduction")
		self.geometry("980x640")
		self.minsize(860, 540)

		self._build_styles()

		# Model state (must be initialized before building layout)
		self.input_rows: List[List[Fraction]] = []
		self.num_cols: int = 0
		self.solver: GaussianEliminationStepper | None = None
		self.step_index: int = 0
		self.play_job: str | None = None
		self.play_interval_ms: int = 500

		self._build_layout()
		self._render_input_matrix()

	def _hide_summary(self) -> None:
		try:
			self.summary_frame.pack_forget()
		except Exception:
			pass

	def on_copy_summary(self) -> None:
		# Build a plain text block with rank, solutions, dimension, and description
		text = []
		text.append(f"Rank: {self.sum_rank_val.cget('text')}")
		text.append(f"Solutions: {self.sum_solutions_val.cget('text')}")
		text.append(f"Dimension: {self.sum_dim_val.cget('text')}")
		try:
			desc = self.sum_desc_text.get("1.0", tk.END).strip()
		except Exception:
			desc = ""
		if desc:
			text.append("Description:")
			text.append(desc)
		full = "\n".join(text)
		self.clipboard_clear()
		self.clipboard_append(full)
		self.update()

	def _update_solution_summary(self) -> None:
		if not self.solver:
			return
		# Use the final step's matrix (RREF) for analysis
		final_step = self.solver.steps[-1]
		A: List[List[Fraction]] = final_step["matrix"]
		m = len(A)
		n = len(A[0])
		# Consider last column as constants if augmented
		is_augmented = True if n >= 2 else False
		num_vars = n - 1 if is_augmented else n

		# Compute rank and inconsistency
		rank = 0
		inconsistent = False
		pivot_cols: List[int] = []
		for r in range(m):
			row = A[r]
			# Find first nonzero among variables columns
			lead_col = None
			for c in range(num_vars):
				if row[c] != 0:
					lead_col = c
					break
			# Check inconsistency: all zero in variables but constant != 0
			if all(row[c] == 0 for c in range(num_vars)):
				if is_augmented and row[-1] != 0:
					inconsistent = True
				continue
			if lead_col is not None:
				rank += 1
				pivot_cols.append(lead_col)

		free_cols = [c for c in range(num_vars) if c not in pivot_cols]
		dim_solution = 0
		solutions_label = "-"
		if inconsistent:
			solutions_label = "0"
			dim_solution = 0
		elif is_augmented:
			if rank == num_vars:
				solutions_label = "1"
				dim_solution = 0
			else:
				solutions_label = "infinite"
				dim_solution = len(free_cols)
		else:
			# Homogeneous or non-augmented matrix: solution space dimension = num_vars - rank
			solutions_label = "infinite" if num_vars - rank > 0 else "1"
			dim_solution = max(0, num_vars - rank)

		# Build variable names: special-case 4 vars -> w,x,y,z; otherwise start from x,y,z then a..w
		if num_vars == 4:
			var_names = ['w', 'x', 'y', 'z']
		else:
			name_pool = [chr(c) for c in range(ord('x'), ord('z') + 1)] + [chr(c) for c in range(ord('a'), ord('w') + 1)]
			var_names = name_pool[:num_vars]

		# Map each pivot column to its row (first nonzero in that row)
		pivot_col_to_row: Dict[int, int] = {}
		for r in range(m):
			row = A[r]
			lead_col = None
			for c in range(num_vars):
				if row[c] != 0:
					lead_col = c
					break
			if lead_col is not None:
				pivot_col_to_row[lead_col] = r

		desc = []
		if inconsistent or num_vars == 0:
			desc_str = "No solution" if inconsistent else ""
		else:
			# Unique solution (no free variables)
			if rank == num_vars and is_augmented:
				values = [Fraction(0) for _ in range(num_vars)]
				for pc, r in pivot_col_to_row.items():
					values[pc] = A[r][-1]
				desc_parts = [f"{var_names[i]} = {fraction_to_str(values[i])}" for i in range(num_vars)]
				desc_str = "\n".join(desc_parts)
			else:
				# Parametric solution: use actual variable names for free variables
				param_names = [var_names[free] for free in free_cols]
				param_map = {free: param_names[i] for i, free in enumerate(free_cols)}
				assignments: List[str] = []
				for j in range(num_vars):
					if j in free_cols:
						assignments.append(f"{var_names[j]} = {param_map[j]}")
						continue
					row_idx = pivot_col_to_row.get(j)
					if row_idx is None:
						assignments.append(f"{var_names[j]} = 0")
						continue
					row = A[row_idx]
					rhs = row[-1] if is_augmented else Fraction(0)
					terms: List[Tuple[Fraction, str]] = []
					for k in free_cols:
						coeff = row[k]
						if coeff != 0:
							terms.append((-coeff, param_map[k]))
					expr = format_linear_expr(rhs, terms)
					assignments.append(f"{var_names[j]} = {expr}")
				desc_str = "\n".join(assignments) if assignments else ""

		# Reveal and populate summary
		try:
			self.summary_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 6))
		except Exception:
			pass
		self.sum_rank_val.configure(text=str(rank))
		self.sum_solutions_val.configure(text=solutions_label)
		self.sum_dim_val.configure(text=str(dim_solution))
		self.sum_desc_text.configure(state=tk.NORMAL)
		self.sum_desc_text.delete("1.0", tk.END)
		self.sum_desc_text.insert("1.0", desc_str)
		self.sum_desc_text.configure(state=tk.DISABLED)

	def _build_styles(self) -> None:
		style = ttk.Style(self)
		try:
			style.theme_use("clam")
		except Exception:
			pass
		style.configure("TButton", padding=6)
		style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
		style.configure("Op.TLabel", font=("Segoe UI", 11))
		style.configure("MatrixCell.TLabel", padding=(6, 2))
		style.configure("RowLabel.TLabel", padding=(8, 2), foreground="#555")

	def _build_layout(self) -> None:
		# Top: Input controls
		input_frame = ttk.Frame(self, padding=10)
		input_frame.pack(side=tk.TOP, fill=tk.X)

		row_entry_lbl = ttk.Label(input_frame, text="Enter row (comma-separated):")
		row_entry_lbl.pack(side=tk.LEFT)

		self.row_entry = ttk.Entry(input_frame, width=50)
		self.row_entry.pack(side=tk.LEFT, padx=(8, 8))
		self.row_entry.bind("<Return>", lambda e: self.on_add_row())

		add_btn = ttk.Button(input_frame, text="Add Row", command=self.on_add_row)
		add_btn.pack(side=tk.LEFT)

		clear_btn = ttk.Button(input_frame, text="Clear Input", command=self.on_clear_input)
		clear_btn.pack(side=tk.LEFT, padx=(8, 0))

		self.solve_btn = ttk.Button(input_frame, text="Solve", command=self.on_solve)
		self.solve_btn.pack(side=tk.RIGHT)

		# Middle: matrix display area
		content = ttk.Frame(self, padding=(10, 0, 10, 10))
		content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

		left = ttk.Frame(content)
		left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		left_header = ttk.Label(left, text="Matrix", style="Header.TLabel")
		left_header.pack(side=tk.TOP, anchor=tk.W, pady=(6, 6))

		self.matrix_canvas = tk.Canvas(left, highlightthickness=0)
		self.matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.matrix_scroll_y = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.matrix_canvas.yview)
		self.matrix_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
		self.matrix_canvas.configure(yscrollcommand=self.matrix_scroll_y.set)

		self.matrix_container = ttk.Frame(self.matrix_canvas)
		self.matrix_canvas.create_window((0, 0), window=self.matrix_container, anchor="nw")
		self.matrix_container.bind("<Configure>", lambda e: self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all")))

		# Right: operations and controls
		right = ttk.Frame(content, width=280)
		right.pack(side=tk.RIGHT, fill=tk.Y)
		right.pack_propagate(False)

		right_header = ttk.Label(right, text="Controls", style="Header.TLabel")
		right_header.pack(side=tk.TOP, anchor=tk.W, pady=(6, 6))

		controls = ttk.Frame(right)
		controls.pack(side=tk.TOP, fill=tk.X)

		self.prev_btn = ttk.Button(controls, text="◀ Prev", command=self.on_prev, state=tk.DISABLED)
		self.prev_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

		self.play_btn = ttk.Button(controls, text="▶ Play", command=self.on_play, state=tk.DISABLED)
		self.play_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

		self.next_btn = ttk.Button(controls, text="Next ▶", command=self.on_next, state=tk.DISABLED)
		self.next_btn.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

		for c in range(3):
			controls.columnconfigure(c, weight=1)

		spd_frame = ttk.Frame(right)
		spd_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 6))
		spd_lbl = ttk.Label(spd_frame, text="Animation speed (ms):")
		spd_lbl.pack(side=tk.LEFT)
		self.spd_entry = ttk.Entry(spd_frame, width=8)
		self.spd_entry.insert(0, str(self.play_interval_ms))
		self.spd_entry.pack(side=tk.LEFT, padx=(6, 0))

		self.reset_btn = ttk.Button(right, text="Reset Steps", command=self.on_reset_steps, state=tk.DISABLED)
		self.reset_btn.pack(side=tk.TOP, anchor=tk.W, pady=(6, 12))

		self.step_label = ttk.Label(right, text="Step: -/-", style="Header.TLabel")
		self.step_label.pack(side=tk.TOP, anchor=tk.W, pady=(6, 6))

		self.op_label = ttk.Label(right, text="Operation: (awaiting input)", style="Op.TLabel", wraplength=240, justify=tk.LEFT)
		self.op_label.pack(side=tk.TOP, anchor=tk.W)

		# Bottom-right: solution summary (hidden until solving)
		self.summary_frame = ttk.LabelFrame(right, text="Solution Summary")
		self.summary_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 6))
		self.summary_frame.pack_forget()

		sf = ttk.Frame(self.summary_frame)
		sf.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

		lbl_rank = ttk.Label(sf, text="Rank:")
		lbl_rank.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
		self.sum_rank_val = ttk.Label(sf, text="-")
		self.sum_rank_val.grid(row=0, column=1, sticky="w", pady=2)

		lbl_solutions = ttk.Label(sf, text="Solutions:")
		lbl_solutions.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
		self.sum_solutions_val = ttk.Label(sf, text="-")
		self.sum_solutions_val.grid(row=1, column=1, sticky="w", pady=2)

		lbl_dim = ttk.Label(sf, text="Dimension:")
		lbl_dim.grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
		self.sum_dim_val = ttk.Label(sf, text="-")
		self.sum_dim_val.grid(row=2, column=1, sticky="w", pady=2)

		lbl_desc = ttk.Label(sf, text="Description:")
		lbl_desc.grid(row=3, column=0, sticky="nw", padx=(0, 6), pady=2)
		# Use a readonly Text so users can select and copy
		self.sum_desc_text = tk.Text(sf, width=26, height=6, wrap=tk.WORD)
		self.sum_desc_text.insert("1.0", "-")
		self.sum_desc_text.configure(state=tk.DISABLED)
		self.sum_desc_text.grid(row=3, column=1, sticky="we", pady=2)

		copy_btn = ttk.Button(sf, text="Copy", command=self.on_copy_summary)
		copy_btn.grid(row=4, column=1, sticky="w", pady=(4, 2))

	def on_add_row(self) -> None:
		text = self.row_entry.get()
		if not text.strip():
			return
		try:
			parts = [p.strip() for p in text.split(',')]
			row = [parse_fraction(p) for p in parts]
		except ValueError as e:
			messagebox.showerror("Invalid Row", str(e))
			return

		if self.num_cols == 0:
			self.num_cols = len(row)
		elif len(row) != self.num_cols:
			messagebox.showerror("Mismatched Columns", f"Expected {self.num_cols} values, got {len(row)}.")
			return

		self.input_rows.append(row)
		self.row_entry.delete(0, tk.END)
		self._render_input_matrix()

	def on_clear_input(self) -> None:
		if self.play_job is not None:
			self.after_cancel(self.play_job)
			self.play_job = None
		self.input_rows.clear()
		self.num_cols = 0
		self.solver = None
		self.step_index = 0
		self._update_controls_enabled(False)
		self.op_label.configure(text="Operation: (awaiting input)")
		self.step_label.configure(text="Step: -/-")
		self._render_input_matrix()
		self._hide_summary()

	def on_solve(self) -> None:
		if not self.input_rows:
			messagebox.showinfo("No rows", "Please add at least one row before solving.")
			return
		# Stop any ongoing playback before solving anew
		if self.play_job is not None:
			self.after_cancel(self.play_job)
			self.play_job = None
			self.play_btn.configure(text="▶ Play")
		try:
			self.solver = GaussianEliminationStepper(self.input_rows)
		except Exception as e:
			messagebox.showerror("Solve Error", str(e))
			return
		self.step_index = 0
		self._update_controls_enabled(True)
		self._render_step()
		# Compute and show solution summary (based on final RREF)
		self._update_solution_summary()
		# Auto-play after solving
		self.on_play()

	def on_prev(self) -> None:
		if not self.solver:
			return
		self.step_index = max(0, self.step_index - 1)
		self._render_step()

	def on_next(self) -> None:
		if not self.solver:
			return
		self.step_index = min(len(self.solver.steps) - 1, self.step_index + 1)
		self._render_step()

	def on_play(self) -> None:
		if not self.solver:
			return
		if self.play_job is None:
			# start playing
			self.play_btn.configure(text="❚❚ Pause")
			# update speed from entry
			try:
				val = int(self.spd_entry.get().strip())
				if val > 0:
					self.play_interval_ms = val
			except Exception:
				pass
			self._schedule_next()
		else:
			# pause
			self.after_cancel(self.play_job)
			self.play_job = None
			self.play_btn.configure(text="▶ Play")

	def _schedule_next(self) -> None:
		if not self.solver:
			return
		if self.step_index < len(self.solver.steps) - 1:
			self.step_index += 1
			self._render_step()
			self.play_job = self.after(self.play_interval_ms, self._schedule_next)
		else:
			# reached the end
			self.play_btn.configure(text="▶ Play")
			self.play_job = None

	def on_reset_steps(self) -> None:
		if not self.solver:
			return
		if self.play_job is not None:
			self.after_cancel(self.play_job)
			self.play_job = None
			self.play_btn.configure(text="▶ Play")
		self.step_index = 0
		self._render_step()

	def _update_controls_enabled(self, enabled: bool) -> None:
		state = tk.NORMAL if enabled else tk.DISABLED
		self.prev_btn.configure(state=state)
		self.play_btn.configure(state=state)
		self.next_btn.configure(state=state)
		self.reset_btn.configure(state=state)

	def _clear_matrix_view(self) -> None:
		for child in self.matrix_container.winfo_children():
			child.destroy()

	def _render_input_matrix(self) -> None:
		self._clear_matrix_view()
		# show instructions if empty
		if not self.input_rows:
			msg = ttk.Label(self.matrix_container, text="Add rows using the input above. Example: 1, -2, 3/5")
			msg.grid(row=0, column=0, sticky="w", padx=6, pady=6)
			return

		# Row labels header
		lbl_hdr = ttk.Label(self.matrix_container, text="orig", style="RowLabel.TLabel")
		lbl_hdr.grid(row=0, column=0, padx=(4, 10), pady=(0, 6))

		for c in range(self.num_cols):
			hdr = ttk.Label(self.matrix_container, text=f"c{c+1}", style="RowLabel.TLabel")
			hdr.grid(row=0, column=c + 1, padx=4, pady=(0, 6))

		for r, row in enumerate(self.input_rows, start=1):
			lbl = ttk.Label(self.matrix_container, text=f"r{r}", style="RowLabel.TLabel")
			lbl.grid(row=r, column=0, padx=(4, 10), pady=2, sticky="e")
			for c, val in enumerate(row):
				cell = ttk.Label(self.matrix_container, text=fraction_to_str(val), style="MatrixCell.TLabel", relief=tk.GROOVE)
				cell.grid(row=r, column=c + 1, padx=2, pady=2, sticky="nsew")

	def _render_step(self) -> None:
		if not self.solver:
			return
		self._clear_matrix_view()
		step = self.solver.steps[self.step_index]
		matrix: List[List[Fraction]] = step["matrix"]
		labels: List[int] = step["row_labels"]

		# Header
		lbl_hdr = ttk.Label(self.matrix_container, text="orig", style="RowLabel.TLabel")
		lbl_hdr.grid(row=0, column=0, padx=(4, 10), pady=(0, 6))
		for c in range(len(matrix[0])):
			hdr = ttk.Label(self.matrix_container, text=f"c{c+1}", style="RowLabel.TLabel")
			hdr.grid(row=0, column=c + 1, padx=4, pady=(0, 6))

		for r, (row, rid) in enumerate(zip(matrix, labels), start=1):
			lbl = ttk.Label(self.matrix_container, text=f"r{rid}", style="RowLabel.TLabel")
			lbl.grid(row=r, column=0, padx=(4, 10), pady=2, sticky="e")
			for c, val in enumerate(row):
				cell = ttk.Label(self.matrix_container, text=fraction_to_str(val), style="MatrixCell.TLabel", relief=tk.GROOVE)
				cell.grid(row=r, column=c + 1, padx=2, pady=2, sticky="nsew")

		self.step_label.configure(text=f"Step: {self.step_index + 1}/{len(self.solver.steps)}")
		self.op_label.configure(text=f"Operation: {step['description']}")

		# enable/disable prev/next depending on position
		self.prev_btn.configure(state=(tk.NORMAL if self.step_index > 0 else tk.DISABLED))
		self.next_btn.configure(state=(tk.NORMAL if self.step_index < len(self.solver.steps) - 1 else tk.DISABLED))

		# If at final step, ensure summary is visible
		if self.solver and self.step_index == len(self.solver.steps) - 1:
			self._update_solution_summary()


def main() -> None:
	app = RowReductionApp()
	app.mainloop()


if __name__ == "__main__":
	main()


