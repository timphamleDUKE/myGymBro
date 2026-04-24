import hashlib
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from newspaper import Article

REQUEST_DELAY_SECONDS = 6.0
JITTER_MIN_SECONDS = 0.3
JITTER_MAX_SECONDS = 1.0
MAX_RETRIES = 3
SKIP_EXISTING_FILES = True
STOP_AFTER_CONSECUTIVE_BLOCKS = 3
SHUFFLE_LINKS = True


def read_links(path: Path) -> list[str]:
    links: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            links.append(line)
    return links


def file_stem_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug).strip("-")
    if not slug:
        slug = "article"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def is_block_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("403", "429", "forbidden", "too many requests", "access denied", "blocked", "captcha")
    )


def get_site_name(url: str) -> str | None:
    netloc = urlparse(url).netloc.lower()
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent.parent
links_file = project_root / "knowledge_base" / "links" / "article-links.txt"
out_dir = project_root / "data" / "raw" / "articles"
out_dir.mkdir(parents=True, exist_ok=True)

urls = read_links(links_file)
if SHUFFLE_LINKS:
    random.shuffle(urls)
    print(f"[INFO] Shuffled {len(urls)} article links before extraction")

consecutive_blocks = 0

for url in urls:
    out_file = out_dir / f"{file_stem_from_url(url)}.txt"
    if SKIP_EXISTING_FILES and out_file.exists():
        print(f"[SKIP] {out_file.name} already exists")
        continue

    text = ""
    blocked_this_url = False
    metadata_lines: list[str] = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()

            title = (article.title or "").strip()
            authors = [author.strip() for author in (article.authors or []) if author.strip()]
            site_name = get_site_name(url)

            metadata_lines = [
                f"TITLE: {title}",
                f"AUTHOR: {', '.join(authors)}",
                f"WEBSITE: {site_name or ''}",
                f"URL: {url}",
                "TYPE: Article",
            ]

            consecutive_blocks = 0
            break
        except Exception as exc:
            blocked_this_url = is_block_error(str(exc))
            if attempt == MAX_RETRIES:
                text = f"[ERROR] {exc}\nURL: {url}"
            else:
                backoff = REQUEST_DELAY_SECONDS * attempt
                sleep_seconds = backoff + random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS)
                print(f"[RETRY] {out_file.name} attempt {attempt}/{MAX_RETRIES} in {sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)

    if text and not text.startswith("[ERROR]") and metadata_lines:
        text = "\n".join(metadata_lines) + f"\n\n{text}"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)

    if blocked_this_url:
        consecutive_blocks += 1
        print(f"[BLOCKED] {out_file.name} (consecutive blocks: {consecutive_blocks})")
        if consecutive_blocks >= STOP_AFTER_CONSECUTIVE_BLOCKS:
            print(f"[STOP] Hit {STOP_AFTER_CONSECUTIVE_BLOCKS} consecutive blocked responses.")
            break

    sleep_seconds = REQUEST_DELAY_SECONDS + random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS)
    print(f"[WAIT] sleeping {sleep_seconds:.1f}s before next article")
    time.sleep(sleep_seconds)
