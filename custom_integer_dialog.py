import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

class CustomIntegerDialog(tk.Toplevel):
    def __init__(self, parent, title=None, prompt="", theme=None, initialvalue=0):
        super().__init__(parent)
        self.transient(parent)
        if title:
            self.title(title)
        self.theme = theme
        if theme:
            self.style = ttk.Style()
            self.style.theme_use(theme)
        self.configure(background='#333')  # Usa un color de fondo que coincida con tu tema oscuro
        self.parent = parent
        self.result = initialvalue
        self.prompt = prompt
        self.initialvalue = initialvalue
        self.create_widgets()
        self.grab_set()  # Make modal
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)  # Handle window close button
        self.wait_window(self)  # Wait for the dialog to be closed

    def create_widgets(self):
        # Prompt label
        ttk.Label(self, text=self.prompt).pack(padx=20, pady=(20, 10))
        
        # Integer entry field with initial value set
        self.entry_var = tk.StringVar(value=str(self.initialvalue))
        self.entry = ttk.Entry(self, textvariable=self.entry_var)
        self.entry.pack(padx=20, pady=10)
        self.entry.bind("<Return>", self.on_ok)  # Bind Return key to accept
        self.entry.bind("<Escape>", self.on_cancel)  # Bind Escape key to cancel
        
        # Frame for buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(padx=10, pady=(0, 20), fill=tk.X)
        
        # OK and Cancel buttons
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Set focus to the entry field
        self.entry.focus_set()

    def on_ok(self, event=None):
        # Validate and return the integer
        try:
            self.result = int(self.entry_var.get())
            self.destroy()
        except ValueError:
            # Handle invalid integer entry
            ttk.Label(self, text="Please enter a valid integer.", foreground="red").pack(pady=(0, 20))

    def on_cancel(self, event=None):
        # Set result to None and close dialog
        self.result = None
        self.destroy()