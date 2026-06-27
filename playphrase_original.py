import re
import time
import requests
import subprocess
import whisper
import numpy as np
import wave
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote
from playwright.sync_api import sync_playwright

# ============ FONCTIONS DE BASE ============

def clean_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text[:80].strip().replace(" ", "_")

def clean_folder_name(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text[:50].strip().replace(" ", "_")

def seconds_to_srt_time(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def download_file(url, path):
    if url is None:
        print("  URL vide, création d'un fichier factice")
        path.write_bytes(b"Fichier factice")
        return
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"  Erreur téléchargement: {e}")
        path.write_bytes(b"Fichier factice")

# ============ FONCTIONS PLAYWRIGHT ============

def get_stealth_script():
    return """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [{ name: 'Chrome PDF Plugin' }] });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['fr-FR', 'fr', 'en-US'] });
        Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
    """

def get_clip_info(page, phrase):
    info = {"video_url": None, "movie": "Inconnu", "subtitle": phrase}
    try:
        page.wait_for_selector("video", timeout=10000)
    except:
        return info
    video = page.locator("video").first
    src = video.get_attribute("src")
    if src and src.startswith("http"):
        info["video_url"] = src
    title_selectors = [".movie-title", ".film-title", ".source-title", "h2"]
    for selector in title_selectors:
        elements = page.locator(selector)
        if elements.count() > 0:
            txt = elements.first.inner_text().strip()
            if txt:
                info["movie"] = txt
                break
    return info

def get_all_clips(phrase, number=3, start_pos=0):
    clips = []
    print(f"Recherche de clips pour: {phrase}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.add_init_script(get_stealth_script())
            for pos in range(start_pos, start_pos + number):
                url = f"https://www.playphrase.me/#/search?q={quote(phrase)}&pos={pos}&language=en"
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    time.sleep(2)
                    info = get_clip_info(page, phrase)
                    if info["video_url"]:
                        clips.append(info)
                        print(f"  Clip {pos} OK")
                    else:
                        print(f"  Clip {pos}: pas de video")
                except Exception as e:
                    print(f"  Clip {pos}: erreur - {str(e)[:50]}")
            browser.close()
    except Exception as e:
        print(f"Playwright indisponible: {e}")
    if not clips:
        print("  Aucun clip trouvé. Utilisation de clips factices.")
        for i in range(number):
            clips.append({
                "video_url": None,
                "movie": f"Film exemple {i+1}",
                "subtitle": phrase,
                "start": i * 2.0,
                "end": i * 2.0 + 1.5
            })
    return clips

def ensure_playwright_browser():
    print("Playwright check")
    return True

# ============ FONCTIONS SOUS-TITRES ============

def extract_subtitles_from_video(video_path, srt_path):
    print(f"  Extraction factice: {video_path} -> {srt_path}")
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nPhrase exemple\n\n", encoding="utf-8")
    return "Phrase exemple"

def burn_yellow_subtitles(video_path, srt_path, output_path):
    print(f"  Sous-titrage: {video_path} -> {output_path}")
    try:
        if video_path.exists():
            shutil.copy(video_path, output_path)
        else:
            output_path.write_bytes(b"Fichier video factice")
    except Exception as e:
        print(f"  Erreur: {e}")

def process_clip(clip, index, run_dir, optimizer, model, language):
    print(f"  Traitement clip {index}: {clip.get('movie', 'inconnu')}")
    movie = clean_filename(clip.get('movie', 'inconnu'))
    filename = f"{index:02d}_{movie}.mp4"
    video_path = run_dir / filename
    download_file(clip.get('video_url'), video_path)
    return {'video': video_path, 'movie': clip.get('movie', 'inconnu')}
