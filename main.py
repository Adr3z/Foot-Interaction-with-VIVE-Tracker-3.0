"""
main.py
-------
Entry point for the VR Tracker Visualizer.
"""

import sys

from src.visualization.pygame_viewer import PygameViewer
from src.visualization.recording_viewer import RecordingViewer
from ml import ( run_ml_pipeline, run_augmented_pipeline, train_knn, train_svm_validation, train_svm_final, )


# ── sub-menus ──────────────────────────────────────────────────────────────────

def _menu_dataset() -> str:
    """Sub-menu for dataset building."""
    while True:
        print("\n  -- Build ML Dataset --")
        print("  1. Normal dataset       (from raw recordings)")
        print("  2. Augmented dataset    (4-way flips from base dataset)")
        print("  0. Back")
        choice = input("  Select [0-2]: ").strip()
        if choice == "1":
            return "dataset_normal"
        elif choice == "2":
            return "dataset_augmented"
        elif choice == "0":
            return "back"
        else:
            print("  [!] Invalid choice.")


def _menu_train() -> str:
    """Sub-menu for model training."""
    while True:
        print("\n  -- Train Model --")
        print("  1. Validation model   (train/test split + metrics)")
        print("  2. Final model        (full augmented dataset, no metrics)")
        print("  0. Back")
        choice = input("  Select [0-2]: ").strip()
        if choice == "1":
            return _menu_train_validation()
        elif choice == "2":
            return "train_svm_final"
        elif choice == "0":
            return "back"
        else:
            print("  [!] Invalid choice.")


def _menu_train_validation() -> str:
    """Sub-menu for validation model classifier selection."""
    while True:
        print("\n    -- Validation Model --")
        print("    1. k-NN")
        print("    2. SVM (RBF)")
        print("    0. Back")
        choice = input("    Select [0-2]: ").strip()
        if choice == "1":
            return "train_knn"
        elif choice == "2":
            return "train_svm_validation"
        elif choice == "0":
            return _menu_train()
        else:
            print("    [!] Invalid choice.")


# ── main menu ──────────────────────────────────────────────────────────────────

def show_terminal_menu() -> str:
    """Displays an interactive terminal menu to select the operation mode."""
    while True:
        print("\n" + "=" * 45)
        print("          VR TRACKER VISUALIZER             ")
        print("=" * 45)
        print(" 1. OpenVR           (Real Hardware / SteamVR)")
        print(" 2. Mock             (Simulated Trackers)")
        print(" 3. Recording Viewer (Analyze NPZ data)")
        print(" 4. Build ML Dataset")
        print(" 5. Train Model")
        print(" 6. Exit")
        print("-" * 45)

        choice = input("Select a mode [1-6]: ").strip()

        if choice == "1":
            return "openvr"
        elif choice == "2":
            return "mock"
        elif choice == "3":
            return "recording"
        elif choice == "4":
            result = _menu_dataset()
            if result != "back":
                return result
        elif choice == "5":
            result = _menu_train()
            if result != "back":
                return result
        elif choice == "6":
            print("\nExiting visualizer. Goodbye!")
            sys.exit(0)
        else:
            print("\n[!] Invalid choice. Please enter a number between 1 and 6.")


# ── entry point ───────────────────────────────────────────────────────────────

def _run_mode(mode: str) -> None:
    if mode == "openvr":
        from src.tracker.openvr_tracker import OpenVRSession

        with OpenVRSession() as session:
            trackers = session.get_trackers()

            if not trackers:
                print("No trackers found. Start SteamVR and try again.")
                return

            viewer = PygameViewer(tracker_sources=trackers, window_size=(2100, 620))
            viewer.run()

    elif mode == "mock":
        from src.tracker.mock_tracker import MockUltimateTracker, MockTracker30

        trackers = [
            MockUltimateTracker(),
            MockTracker30(),
        ]
        viewer = PygameViewer(tracker_sources=trackers, window_size=(1280, 800))
        viewer.run()

    elif mode == "recording":
        viewer = RecordingViewer(window_size=(1970, 630), loop=False)
        viewer.run()

    elif mode == "dataset_normal":
        run_ml_pipeline()

    elif mode == "dataset_augmented":
        run_augmented_pipeline()

    elif mode == "train_knn":
        train_knn()

    elif mode == "train_svm_validation":
        train_svm_validation()

    elif mode == "train_svm_final":
        train_svm_final()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [openvr|mock|recording|dataset_normal|dataset_augmented|train_knn|train_svm_validation|train_svm_final]")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        _run_mode(sys.argv[1].lower())
        return

    while True:
        mode = show_terminal_menu()
        _run_mode(mode)
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
