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
		self.title("Row Reduction (Gaussian Elimination) - Exact Fractions")
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


def main() -> None:
	app = RowReductionApp()
	app.mainloop()


if __name__ == "__main__":
	main()


