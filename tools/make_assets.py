"""Pre-render the dialogue lines with espeak-ng into assets/sounds/.

Run once (needs the espeak-ng program):  python tools/make_assets.py
The WAV files are committed, so users of the lab do not need espeak-ng.
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds"

LINES = {
    # cinema dialogue scene
    "maya_1": ("en-us+f3", 150, "Did you hear that? Something is moving out there, past the trees."),
    "leo_1": ("en-gb+m3", 145, "It's only the wind. Stay where you are, I'll go and look."),
    "maya_2": ("en-us+f3", 150, "Leo, wait. Don't go alone."),
    "leo_2": ("en-gb+m3", 140, "I'm right here. Can you hear me now? I am walking towards you."),
    # car scene
    "radio": ("en-us+m2", 160, "Good evening, and welcome back. Heavy rain is expected across the region tonight, so drive carefully."),
    "nav": ("en-us+f4", 165, "In four hundred metres, turn left onto the motorway."),
    "caller": ("en-gb+f2", 155, "Hi, it's me. I'm running a little late, I'll see you at the station."),
    # calibration voice
    "test_voice": ("en-us+m3", 150, "Testing, one, two, three. This voice should come from the centre of the screen."),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (voice, speed, text) in LINES.items():
        f = OUT / f"{name}.wav"
        subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(f), text], check=True)
        print("wrote", f)


if __name__ == "__main__":
    main()
