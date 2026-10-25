"""Selectable error details for the recording window; no device access."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SelectableErrorPanel(ttk.Frame):
    """Read-only details with normal Text selection and Ctrl+C bindings."""

    def __init__(self, parent, *, max_rows=8, **kwargs):
        super().__init__(parent, **kwargs)
        self.max_rows = max(3, int(max_rows))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = tk.Text(self, wrap="word", height=3, width=72,
                            state="disabled", undo=False, takefocus=True,
                            font="TkDefaultFont")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.copy_button = ttk.Button(self, text="エラーをコピー", command=self.copy_error)
        self.copy_button.grid(row=1, column=0, columnspan=2, sticky="e", pady=(4, 0))
        self._message = None
        self.set_error("")

    def set_error(self, message):
        """Update only changed details so periodic UI refresh keeps selections."""
        message = "" if message is None else str(message)
        if message == self._message:
            return
        self._message = message
        displayed = message or "エラーはありません。"
        # Long errors remain scrollable without consuming the entire window.
        rows = sum(max(1, (len(line) + 71) // 72) for line in displayed.split("\n"))
        self.text.configure(state="normal", height=min(self.max_rows, max(3, rows)))
        try:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", displayed)
        finally:
            self.text.configure(state="disabled")
        self.copy_button.configure(state="normal" if message else "disabled")

    def copy_error(self):
        """Copy the exact full error, even if only part is currently visible."""
        if not self._message:
            return False
        try:
            self.clipboard_clear()
            self.clipboard_append(self._message)
        except Exception:
            # A temporarily unavailable clipboard must not replace the error.
            return False
        return True

