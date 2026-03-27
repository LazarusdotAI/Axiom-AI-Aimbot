"""
Quality image scraper — downloads directly from Bing image search
results using requests + BeautifulSoup. No icrawler dependency.
Longer timeouts (15s), skips slow hosts, robust error handling.
"""
import os, sys, time, hashlib, re, json
import requests
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

GAMES = {
    "se5": {
        "name": "Sniper Elite 5",
        "out_dir": "_gamedata/se5/images_v2",
        "queries": [
            "sniper elite 5 gameplay shooting enemies",
            "sniper elite 5 kill cam enemy soldier",
            "sniper elite 5 stealth takedown soldiers",
            "sniper elite 5 invasion mode gameplay",
            "sniper elite 5 scope aim enemy screenshot",
            "sniper elite 5 co-op gameplay nazi guards",
            "sniper elite 5 PC gameplay combat screenshot",
            "sniper elite 5 walkthrough mission enemies",
        ],
    },
    "sm2": {
        "name": "Space Marine 2",
        "out_dir": "_gamedata/sm2/images_v2",
        "queries": [
            "space marine 2 tyranid swarm gameplay",
            "space marine 2 horde mode fighting enemies",
            "space marine 2 boss fight screenshot",
            "warhammer space marine 2 shooting tyranids",
            "space marine 2 operations gameplay enemies",
            "space marine 2 chaos marines PvP combat",
            "space marine 2 carnifex warrior fight",
            "space marine 2 PC gameplay combat screenshot",
        ],
    },
    "plaz": {
        "name": "Project Lazarus (Roblox)",
        "out_dir": "_gamedata/plaz/images_v2",
        "queries": [
            "project lazarus roblox zombie gameplay",
            "project lazarus roblox shooting zombies",
            "project lazarus roblox gameplay 2024",
            "roblox project lazarus zombie horde",
            "roblox zombie game fps gameplay screenshot",
            "roblox zombie rush gameplay screenshot",
        ],
    },
}


def get_bing_image_urls(query, count=80):
    """Scrape image URLs from Bing image search results page."""
    import html as html_mod
    urls = []
    for offset in range(0, count, 35):
        search_url = (
            f"https://www.bing.com/images/search?"
            f"q={quote_plus(query)}&first={offset}&count=35&qft=+filterui:imagesize-large"
        )
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"    Bing page fetch failed: {e}")
            continue

        # Bing HTML-encodes the JSON with &quot; so unescape first
        text = html_mod.unescape(resp.text)
        # Extract murl (media URL) from Bing's unescaped HTML
        found = re.findall(r'"murl":"(https?://[^"]+)"', text)
        urls.extend(found)
        time.sleep(1)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique[:count]


def download_image(url, save_path, timeout=15):
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype and "octet" not in ctype:
            return False
        data = resp.content
        if len(data) < 5000:
            return False
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def deduplicate(folder):
    seen = set()
    removed = 0
    for f in sorted(os.listdir(folder)):
        fp = os.path.join(folder, f)
        if not os.path.isfile(fp):
            continue
        h = hashlib.md5(open(fp, "rb").read()).hexdigest()
        if h in seen:
            os.remove(fp)
            removed += 1
        else:
            seen.add(h)
    return removed


def clean_bad_images(folder):
    import cv2
    removed = 0
    for f in sorted(os.listdir(folder)):
        fp = os.path.join(folder, f)
        if not os.path.isfile(fp):
            continue
        if os.path.getsize(fp) < 5000:
            os.remove(fp); removed += 1; continue
        img = cv2.imread(fp)
        if img is None:
            os.remove(fp); removed += 1; continue
        h, w = img.shape[:2]
        if w < 300 or h < 200:
            os.remove(fp); removed += 1
    return removed



def scrape_game(key):
    cfg = GAMES[key]
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)

    # Count existing files to set starting index
    existing = len([f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))])
    idx = existing

    print(f"\n{'='*60}")
    print(f"  Scraping: {cfg['name']}")
    print(f"  Output:   {out} ({existing} existing)")
    print(f"  Queries:  {len(cfg['queries'])}")
    print(f"{'='*60}")

    total_downloaded = 0
    for i, query in enumerate(cfg["queries"], 1):
        print(f"\n  [{i}/{len(cfg['queries'])}] \"{query}\"")

        urls = get_bing_image_urls(query, count=60)
        print(f"    Found {len(urls)} image URLs")

        downloaded = 0
        for url in urls:
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            elif ".webp" in url.lower():
                ext = ".webp"

            idx += 1
            save_path = os.path.join(out, f"{key}_{idx:05d}{ext}")
            if download_image(url, save_path):
                downloaded += 1
                # Print progress every 10
                if downloaded % 10 == 0:
                    print(f"    ... {downloaded} downloaded")

        print(f"    Downloaded: {downloaded}/{len(urls)}")
        total_downloaded += downloaded
        time.sleep(3)  # Be nice between queries

    # Clean up
    bad = clean_bad_images(out)
    dupes = deduplicate(out)
    total = len([f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))])
    print(f"\n  Cleanup: removed {bad} bad + {dupes} duplicates")
    print(f"  Total downloaded this run: {total_downloaded}")
    print(f"  Final image count: {total}")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python _scrape_quality.py <se5|sm2|plaz|all>")
        sys.exit(1)

    target = sys.argv[1].lower()
    if target == "all":
        for k in GAMES:
            scrape_game(k)
    elif target in GAMES:
        scrape_game(target)
    else:
        print(f"Unknown: {target}. Use: {', '.join(GAMES.keys())}")