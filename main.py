"""
main.py
-------
Entry point for the VR Tracker Visualizer.
"""

import sys

MODE = "openvr"

from src.visualization.pygame_viewer import PygameViewer

def main() -> None:

    if MODE == "openvr":
        from src.tracker.openvr_tracker import OpenVRSession

        with OpenVRSession() as session:
            trackers = session.get_trackers()

            if not trackers:
                print("No se encontraron trackers. Inicia SteamVR y vuelve a intentarlo.")
                sys.exit(1)

            viewer = PygameViewer(tracker_sources=trackers, window_size=(1280, 720))
            viewer.run()

    else:  # mock
        from tracker.mock_tracker import MockUltimateTracker, MockTracker30

        trackers = [
            MockUltimateTracker(),
            MockTracker30(),
        ]
        viewer = PygameViewer(tracker_sources=trackers, window_size=(1280, 720))
        viewer.run()


if __name__ == "__main__":
    main()
