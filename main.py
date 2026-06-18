"""
main.py
-------
Entry point for the VR Tracker Visualizer.
"""

import sys

from src.visualization.pygame_viewer import PygameViewer
from src.visualization.recording_viewer import RecordingViewer

DEFAULT_MODE = "recording"


def main() -> None:
    mode = DEFAULT_MODE
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

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
        viewer = RecordingViewer(window_size=(1800, 550), loop=False)
        viewer.run()

    else:
        print(f"Modo desconocido: {mode}")
        print("Uso: python main.py [openvr|mock|recording]")
        sys.exit(1)


if __name__ == "__main__":
    main()
