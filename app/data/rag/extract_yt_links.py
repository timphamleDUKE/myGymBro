import random
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

# Extraction behavior (tune these for future runs)
REQUEST_DELAY_SECONDS = 15
JITTER_MIN_SECONDS = 0.3
JITTER_MAX_SECONDS = 1.2
MAX_RETRIES = 3
SKIP_EXISTING_FILES = True
STOP_AFTER_CONSECUTIVE_BLOCKS = 1

def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/") or None
    return None

base_dir = Path(__file__).resolve().parent
links_file = base_dir / "links" / "yt-links.txt"
out_dir = base_dir / "raw" / "videos"
out_dir.mkdir(parents=True, exist_ok=True)

video_ids = []

with open(links_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vid = extract_video_id(line)
        if vid:
            video_ids.append(vid)

api = YouTubeTranscriptApi()
consecutive_blocks = 0

for video_id in video_ids:
    out_file = out_dir / f"{video_id}.txt"
    if SKIP_EXISTING_FILES and out_file.exists():
        print(f"[SKIP] {video_id} already exists")
        continue

    text = ""
    blocked_this_video = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            transcript = api.fetch(video_id)
            text = " ".join([entry.text for entry in transcript])
            consecutive_blocks = 0
            break
        except Exception as exc:
            error_message = str(exc)
            lowered = error_message.lower()
            blocked_this_video = (
                "youtube is blocking requests from your ip" in lowered
                or "ip has been blocked" in lowered
                or "requestblocked" in lowered
                or "ipblocked" in lowered
            )
            if attempt == MAX_RETRIES:
                text = f"[ERROR] {exc}"
            else:
                backoff = REQUEST_DELAY_SECONDS * attempt
                sleep_seconds = backoff + random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS)
                print(f"[RETRY] {video_id} attempt {attempt}/{MAX_RETRIES} in {sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)

    if blocked_this_video:
        consecutive_blocks += 1
        print(f"[BLOCKED] {video_id} (consecutive blocks: {consecutive_blocks})")
        if consecutive_blocks >= STOP_AFTER_CONSECUTIVE_BLOCKS:
            print(
                f"[STOP] Reached {STOP_AFTER_CONSECUTIVE_BLOCKS} consecutive IP blocks. "
                "Switch networks or wait before retrying."
            )
            break

    sleep_seconds = REQUEST_DELAY_SECONDS + random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS)
    print(f"[WAIT] sleeping {sleep_seconds:.1f}s before next video")
    time.sleep(sleep_seconds)