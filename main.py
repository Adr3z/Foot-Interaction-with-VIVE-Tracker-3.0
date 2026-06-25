"""
main.py
-------
Entry point for the VR Tracker Visualizer.
"""

import sys

from src.visualization.pygame_viewer import PygameViewer
from src.visualization.recording_viewer import RecordingViewer
from ml import run_ml_pipeline, train_knn, train_svm


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
        print(" 5. Train k-NN Classifier")
        print(" 6. Train SVM Classifier (RBF)")
        print(" 7. Exit")
        print("-" * 45)

        choice = input("Select a mode [1-7]: ").strip()

        if choice == "1":
            return "openvr"
        elif choice == "2":
            return "mock"
        elif choice == "3":
            return "recording"
        elif choice == "4":
            return "ml_dataset"
        elif choice == "5":
            return "train_knn"
        elif choice == "6":
            return "train_svm"
        elif choice == "7":
            print("\nExiting visualizer. Goodbye!")
            sys.exit(0)
        else:
            print("\n[!] Invalid choice. Please enter a number between 1 and 7.")


def main() -> None:
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = show_terminal_menu()

    # ---- MODE EXECUTION ----

    if mode == "openvr":
        from src.tracker.openvr_tracker import OpenVRSession

        with OpenVRSession() as session:
            trackers = session.get_trackers()

            if not trackers:
                print("No se encontraron trackers. Inicia SteamVR y vuelve a intentarlo.")
                sys.exit(1)

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

    elif mode == "ml_dataset":
        run_ml_pipeline()

    elif mode == "train_knn":
        train_knn()

    elif mode == "train_svm":
        train_svm()

    else:
        print(f"Modo desconocido: {mode}")
        print("Uso: python main.py [openvr|mock|recording|ml_dataset|train_knn|train_svm]")
        sys.exit(1)


if __name__ == "__main__":
    main()
