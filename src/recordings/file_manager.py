"""
file_manager.py
-----------------

Handles system files interactions and clean extraction of the data
"""

import os 
import tkinter as tk
from tkinter import filedialog
import numpy as np

class FileManager:
    @staticmethod
    def select_recording_path(initial_dir: str = "data/raw") -> str | None:
        """ """
        root = tk.Tk()
        root.withdraw()

        target_dir = os.path.abspath(initial_dir)
        if not os.path.exists(target_dir):
            target_dir = os.getcwd()

        file_path = filedialog.askopenfilename(
            initialdir=target_dir, 
            title="Select Gesture Record (.npz)", 
            filetypes=[("NumPy files compressed", "*.npz")]
        )

        return file_path if file_path else None

    
    @staticmethod
    def load_recording_data(file_path: str) -> dict[str, dict[str, np.ndarray]]:
        """ """
        raw_data = np.load(file_path)
        keys = raw_data.files
        tracker_names = sorted(list(set([k.rsplit('_', 1)[0] for k in keys])))

        structured_data = {}
        for name in tracker_names:
            structured_data[name] = {
                "pos": raw_data[f"{name}_position"],
                "rot": raw_data[f"{name}_rotation"],
                "time": raw_data[f"{name}_timestamps"],
            }

        return structured_data