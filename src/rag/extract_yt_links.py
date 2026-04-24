import random
import time
import json
import re
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

REQUEST_DELAY_SECONDS = 15
JITTER_MIN_SECONDS = 0.3
JITTER_MAX_SECONDS = 1.2
MAX_RETRIES = 3
SKIP_EXISTING_FILES = False
STOP_AFTER_CONSECUTIVE_BLOCKS = 1


def fetch_video_metadata(video_id: str) -> tuple[str | None, str | None]:
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    # Try YouTube's oEmbed endpoint first since it returns compact JSON without an API key.
    oembed_url = f"https://www.youtube.com/oembed?url={watch_url}&format=json"
    try:
        request = Request(oembed_url, headers=request_headers)
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        title = payload.get("title", "").strip()
        author = payload.get("author_name", "").strip()
        if title or author:
            return title or None, author or None
    except Exception:
        pass

    # Fall back to scraping the watch page title if oEmbed is unavailable.
    try:
        request = Request(watch_url, headers=request_headers)
        with urlopen(request, timeout=15) as response:
            page = response.read().decode("utf-8", errors="ignore")
        match = re.search(r"<title>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None, None
        title = unescape(match.group(1)).strip()
        title = re.sub(r"\s*-\s*YouTube\s*$", "", title).strip()

        author_match = re.search(r'"ownerChannelName":"(.*?)"', page)
        author = unescape(author_match.group(1)).strip() if author_match else None
        return title or None, author or None
    except Exception:
        return None, None

def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/") or None
    return None

base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent.parent
links_file = project_root / "knowledge_base" / "links" / "yt-links.txt"
out_dir = project_root / "data" / "raw" / "videos"
out_dir.mkdir(parents=True, exist_ok=True)

video_entries: list[tuple[str, str]] = []

with open(links_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vid = extract_video_id(line)
        if vid:
            video_entries.append((vid, line))

api = YouTubeTranscriptApi()
consecutive_blocks = 0

for video_id, source_url in video_entries:
    out_file = out_dir / f"{video_id}.txt"
    if SKIP_EXISTING_FILES and out_file.exists():
        print(f"[SKIP] {video_id} already exists")
        continue

    title, author = fetch_video_metadata(video_id)
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

    if text and not text.startswith("[ERROR]"):
        header_lines = [
            f"TITLE: {title or ''}",
            f"AUTHOR: {author or ''}",
            "WEBSITE: youtube.com",
            f"URL: {source_url}",
            "TYPE: Video",
            "",
            text,
        ]
        text = "\n".join(header_lines)

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
