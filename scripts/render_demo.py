#!/usr/bin/env python3
"""Render the recorded terminal stream to MP4; no log messages are synthesized."""

import argparse
import json
import math
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1600, 1000, 12
BG = "#09111f"
PANEL = "#101c2e"
TEXT = "#e8f1ff"
MUTED = "#8398b4"
CYAN = "#55dfd2"
BLUE = "#8eafff"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STAGE = re.compile(r"\[STAGE (\d+)/10\] ([^\r\n]+)")


def font(size, mono=False):
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
        if mono
        else ["/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install DejaVu Sans and DejaVu Sans Mono to render the demo")


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class Terminal:
    def __init__(self):
        self.lines = [""]
        self.stage = 0
        self.title = "Starting the recorded experiment"

    def feed(self, text):
        match = STAGE.search(text)
        if match:
            self.stage, self.title = int(match[1]), match[2]
            self.lines = [""]
        text = ANSI.sub("", text).replace("\r\n", "\n")
        for char in text:
            if char == "\n":
                self.lines.append("")
            elif char == "\r":
                self.lines[-1] = ""
            elif char == "\b":
                self.lines[-1] = self.lines[-1][:-1]
            elif char == "\t":
                self.lines[-1] += "    "
            elif char.isprintable():
                self.lines[-1] += char
        self.lines = self.lines[-200:]

    def visible(self):
        result = []
        for line in self.lines:
            result.extend(
                textwrap.wrap(line, width=111, replace_whitespace=False, drop_whitespace=False)
                or [""]
            )
        return result[-29:]


def frame(terminal, elapsed, total, metadata, fonts):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    # Editorial chrome surrounds the recorded output; the terminal itself is verbatim.
    draw.rounded_rectangle((40, 34, 93, 87), 14, fill=CYAN)
    draw.text((54, 38), "S", font=fonts["logo"], fill=BG)
    draw.text((111, 35), "STaR", font=fonts["heading"], fill=TEXT)
    draw.text((111, 91), "SELF-TAUGHT REASONING  /  APPLE SILICON", font=fonts["label"], fill=MUTED)
    draw.rounded_rectangle((1202, 45, 1560, 94), 24, fill="#16312f")
    draw.ellipse((1220, 65, 1228, 73), fill=CYAN)
    device = metadata["hardware"]["device"].upper()
    memory = metadata["hardware"].get("memory_gib")
    memory_label = f"{memory:g} GB" if memory else "LOCAL"
    draw.text(
        (1242, 58), f"RECORDED  /  {device}  /  {memory_label}", font=fonts["label"], fill=CYAN
    )
    draw.text(
        (42, 133),
        "A complete experiment, from baseline to a verified checkpoint.",
        font=fonts["subheading"],
        fill=TEXT,
    )

    draw.rounded_rectangle((40, 188, 1228, 910), 18, fill=PANEL, outline="#22354c", width=1)
    draw.rounded_rectangle((40, 188, 1228, 237), 18, fill="#19273d")
    draw.rectangle((41, 215, 1227, 237), fill="#19273d")
    for x, color in [(61, "#ee6e75"), (83, "#e8bb60"), (105, "#6cbd92")]:
        draw.ellipse((x, 207, x + 11, 218), fill=color)
    draw.text((149, 199), "STaR  —  recorded terminal output", font=fonts["small"], fill=MUTED)
    for row, line in enumerate(terminal.visible()):
        color = TEXT
        if line.startswith("$") or line.startswith("[STAGE"):
            color = BLUE
        elif line.startswith("PASS") or line.startswith("RESULT:") or line.startswith("[exit 0"):
            color = CYAN
        elif "FAILED" in line or "Error:" in line:
            color = "#ff9195"
        elif line.strip().startswith(('"', "{", "}")):
            color = "#b9c9de"
        draw.text((62, 252 + row * 22), line, font=fonts["terminal"], fill=color)

    x = 1260
    draw.text((x, 194), "EXPERIMENT", font=fonts["label"], fill=MUTED)
    model = metadata["config"]["model"].split("/")[-1]
    model = "GPT-2 · 124M" if model == "gpt2" else model
    facts = [
        ("MODEL", model),
        ("DATASET", "CommonsenseQA"),
        (
            "SAMPLE",
            f"{metadata['config']['train_samples']} train / {metadata['config']['eval_samples']} dev",
        ),
        ("TRAINING", f"{metadata['config']['iterations']} rounds · full epochs"),
    ]
    if terminal.stage == 3:
        facts = [
            ("MODEL", "Tiny GPT-J (fixture)"),
            ("DATASET", "Synthetic control"),
            ("SAMPLE", "2 controlled questions"),
            ("TRAINING", "2 branch-check rounds"),
        ]
    y = 232
    for key, value in facts:
        draw.text((x, y), key, font=fonts["small"], fill=MUTED)
        draw.text((x, y + 24), value, font=fonts["body"], fill=TEXT)
        y += 76
    draw.line((x, 548, 1557, 548), fill="#2a3c52", width=1)
    draw.text((x, 570), f"STAGE {terminal.stage:02d} / 10", font=fonts["label"], fill=CYAN)
    display_title = (
        "Pretrained GPT-2 before STaR fine-tuning" if terminal.stage == 4 else terminal.title
    )
    for index, line in enumerate(textwrap.wrap(display_title, 24)):
        draw.text((x, 607 + index * 28), line, font=fonts["body"], fill=TEXT)
    if terminal.stage == 10:
        draw.rounded_rectangle((x, 722, 1558, 782), 12, fill="#15352f")
        draw.text(
            (x + 20, 738), f"PASS · {metadata['tests_passed']} tests", font=fonts["body"], fill=CYAN
        )
    else:
        draw.text((x, 743), "Actual execution", font=fonts["body"], fill=TEXT)
        draw.text((x, 777), "No simulated log output", font=fonts["small"], fill=MUTED)
    draw.text((x, 835), "RECORDED ELAPSED TIME", font=fonts["small"], fill=MUTED)
    draw.text((x, 865), clock(elapsed), font=fonts["timer"], fill=TEXT)
    draw.rounded_rectangle((40, 936, 1560, 941), 2, fill="#25384c")
    if elapsed > 0:
        draw.rounded_rectangle(
            (40, 936, 40 + int(1520 * min(1, elapsed / total)), 941), 2, fill=CYAN
        )
    draw.text(
        (40, 961),
        "REAL TERMINAL CAPTURE  ·  TIME COMPRESSED; LONG IDLE GAPS SHORTENED",
        font=fonts["small"],
        fill=MUTED,
    )
    draw.text((1272, 961), "ENGINEERING VALIDATION", font=fonts["small"], fill=MUTED)
    return image


def render(folder, output, speed=5.0, idle_limit=2.0):
    metadata = json.loads((folder / "result.json").read_text())
    if metadata["status"] != "passed":
        raise ValueError("Cannot render a passing demo from a failed experiment")
    raw = [json.loads(line) for line in (folder / "terminal.cast").read_text().splitlines()]
    events = [event for event in raw[1:] if event[1] == "o"]
    if not events:
        raise ValueError("The recording is empty")
    fonts = {
        "logo": font(40),
        "heading": font(49),
        "subheading": font(24),
        "label": font(16),
        "small": font(14),
        "body": font(23),
        "timer": font(37, True),
        "terminal": font(17, True),
    }
    schedule, displayed, previous = [], 2.0, 0.0
    stage_started = None
    for event in events:
        displayed += min(max(0, event[0] - previous) / speed, idle_limit)
        if STAGE.search(event[2]):
            if stage_started is not None:
                displayed = max(displayed, stage_started + 3.0)
            stage_started = displayed
        schedule.append((displayed, event))
        previous = event[0]
    duration = displayed + 9.0
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    terminal, index, elapsed, last_frame = Terminal(), 0, 0.0, None
    try:
        for number in range(math.ceil(duration * FPS)):
            current = number / FPS
            while index < len(schedule) and schedule[index][0] <= current:
                _, event = schedule[index]
                terminal.feed(event[2])
                elapsed = event[0]
                index += 1
            clock_time = elapsed
            if 0 < index < len(schedule):
                previous_display = schedule[index - 1][0]
                next_display, next_event = schedule[index]
                if next_display > previous_display:
                    fraction = (current - previous_display) / (next_display - previous_display)
                    clock_time += max(0, min(1, fraction)) * (next_event[0] - elapsed)
            last_frame = frame(terminal, clock_time, events[-1][0], metadata, fonts)
            process.stdin.write(last_frame.tobytes())
            if number == int(duration * FPS / 2):
                last_frame.save(output.with_name(output.stem + "-midpoint.png"))
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    last_frame.save(output.with_name(output.stem + "-poster.png"))
    info = {
        "source": str(folder / "terminal.cast"),
        "source_seconds": events[-1][0],
        "video_seconds": math.ceil(duration * FPS) / FPS,
        "speed": speed,
        "maximum_idle_gap_seconds": idle_limit,
        "minimum_stage_seconds": 3.0,
        "events": len(events),
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "audio": False,
        "note": "All recorded output events retained in order. Timing compressed; long idle gaps shortened; each stage held for at least 3 seconds for readability. Project path prefix removed during capture.",
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/media/star-e2e-demo.mp4"))
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--idle-limit", type=float, default=2.0)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    if args.speed <= 0 or args.idle_limit <= 0:
        parser.error("speed and idle-limit must be positive")
    render(args.report_dir, args.output, args.speed, args.idle_limit)
