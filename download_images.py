#!/usr/bin/env python3

"""
Image Downloader for Weed ID Training
====================================
Downloads up to 500 high-quality images per species, using:

1. GBIF (from multimedia.txt and occurrence.txt):
   - Filters for open CC licenses
   - Only downloads or converts to .jpg
   - Uses gbif_taxon_cache.json to map gbifID to species
   - Deduplicates using downloaded/rejected log files

2. iNaturalist fallback:
   - API-based, with retry/backoff for 429s
   
Input:
    [1] An Darwin Core Archive GBIF download of all occurences found in the species list. To submit a download request, use create_gbif_media_request.py
    [2] A mapping of species to GBIF Taxon Keys (gbif_taxon_mappings.json). The same create_gbif_media_request.py will output this.
    
    Place the required files (gbif_taxon_mappings.json, occurence.txt and multimedia.txt) in the root folder of this script.
    
    Last Generated GBIF Media Download
    https://www.gbif.org/occurrence/download/0000583-250711103210423
    
Output:
- Organizes into `training_data/Genus_species/`
- Names images as `0001.jpg`, `0002.jpg`, ...
- Logs all downloaded and rejected image URLs

Usage:
    python download_images.py
"""

# Imports
import os
import sys
import csv
import json
import time
import logging
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED

import signal
import threading

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn, track
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.console import Group

# Global shutdown flag
shutdown_flag = threading.Event()

# Handle an exit event
def handle_exit(signum, frame):
    logger.warning("🛑 Received interrupt. Shutting down gracefully...")
    shutdown_flag.set()

# Listen to exit signals
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# === Config ===
THREADS = 32
TARGET_IMAGES = 500
SCRIPT_DIR = Path(__file__).parent
MULTIMEDIA_FILE = SCRIPT_DIR / "multimedia.txt"
OCCURRENCE_FILE = SCRIPT_DIR / "occurrence.txt"
TAXON_CACHE_FILE = SCRIPT_DIR / "gbif_taxon_mappings.json"
SPECIES_FILE = SCRIPT_DIR / "species.json"
OUTPUT_DIR = Path("/mnt/d/training_data/")
USER_AGENT = "WeedIDBot/1.0"

OCCURRENCE_MAP_CACHE = SCRIPT_DIR / "occurrence_gbifid_species_map.json"
MULTIMEDIA_MAP_CACHE = SCRIPT_DIR / "multimedia_gbif_species_urls.json"
OCCURRENCE_LINE_COUNT = 12869179
MULTIMEDIA_LINE_COUNT = 19080086

# Fix CSV Field Size
csv.field_size_limit(sys.maxsize)

# Ignore decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None   # disables the warning

# === Logging Setup ===
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(console=console, show_path=False)])
logger = logging.getLogger("downloader")
failed_log = SCRIPT_DIR / "rejected_urls.log"
success_log = SCRIPT_DIR / "downloaded_urls.log"

# === Load species ===
with open(SPECIES_FILE) as f:
    accepted_species = set(json.load(f))

# === Cleanup unlisted species folders ===
logger.info("🧹 Cleaning up species folders not in species list...")
for folder in OUTPUT_DIR.iterdir():
    if folder.is_dir():
        folder_name = folder.name.replace("_", " ")
        if folder_name not in accepted_species:
            logger.warning(f"🗑️  Deleting unlisted folder: {folder.name}")
            for file in folder.glob("*"):
                file.unlink()
            folder.rmdir()
            
# Log configuration and summary
logger.info(f"📷 Downloading {TARGET_IMAGES} images per species across {len(accepted_species)} species using {THREADS} download threads")
logger.info("🚧 Using GBIF for primary data source and falling back to iNaturalist")

# === Load taxon mapping cache ===
with open(TAXON_CACHE_FILE) as f:
    taxon_cache = json.load(f)
    taxon_to_species = {v: k for k, v in taxon_cache.items() if k in accepted_species}

# === Build GBIF ID → Species Map ===
gbifid_to_species = {}
# If a cached mapping exists, load it
if OCCURRENCE_MAP_CACHE.exists():
    logger.info("🔁 Loading cached gbifID → species mapping...")
    with open(OCCURRENCE_MAP_CACHE, "r", encoding="utf-8") as f:
        gbifid_to_species = json.load(f)
# Otherwise build a new mapping
else:
    logger.info("📇 Building GBIF ID → Species mapping from occurrence.txt...")
    # Open the occurence.txt file
    with open(OCCURRENCE_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # For each row in the occurence.txt, save the GBID ID if it matches a species in our accepted_species.json list
        for row in track(reader, description="Indexing occurrence.txt...", total=OCCURRENCE_LINE_COUNT):
            gbif_id = row.get("gbifID")
            taxon_id = int(row.get("taxonKey")) if row.get("taxonKey") and row.get("taxonKey").isdigit() else None
            species_name = taxon_to_species.get(taxon_id)
            
            basis = row.get("basisOfRecord", "").strip().upper()

            # Only accept if it's a field observation, not a preserved specimen
            if basis not in {"HUMAN_OBSERVATION", "OBSERVATION", "MACHINE_OBSERVATION"}:
                continue
            
            if gbif_id and species_name:
                gbifid_to_species[gbif_id] = species_name
                
    # Save the mapping so we don't have to recreate it
    with open(OCCURRENCE_MAP_CACHE, "w", encoding="utf-8") as f:
        json.dump(gbifid_to_species, f, indent=2)
    logger.info(f"✅ Cached mapping to {OCCURRENCE_MAP_CACHE}")

# === Index multimedia.txt and group image URLs ===
# GBIF Image list to be populated and keyed by species name
gbif_images = defaultdict(list)
# Load an existing multimedia map if it exists
if MULTIMEDIA_MAP_CACHE.exists():
    logger.info("🔁 Loading cached multimedia image URLs...")
    with open(MULTIMEDIA_MAP_CACHE, "r", encoding="utf-8") as f:
        gbif_images = json.load(f)
# Otherwise recreate the mapping
else:
    logger.info("📷 Indexing multimedia.txt by species...")
    with open(MULTIMEDIA_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # For each row in the multimedia.txt, save the URL in the dictionary if it meets the requirements
        for row in track(reader, description="Indexing GBIF images...", total=MULTIMEDIA_LINE_COUNT):
            gbif_id = row.get("gbifID")
            species = gbifid_to_species.get(gbif_id)
            
            # Stop saving images if we reach twice the target images, this should be plenty
            if not species or len(gbif_images[species]) >= TARGET_IMAGES * 2:
                continue
            
            # Extract info about the media
            url = row.get("identifier", "")
            license_name = (row.get("license") or "").lower()
            license_ok = ("creativecommons" in license_name) or ("cc" in license_name)
            jpeg = "jpeg" in (row.get("format") or "").lower()
            still = "stillimage" in (row.get("type") or "").lower()
            
            # If the image meets the requirements, add it to the dictionary
            if url.lower().endswith(".jpg") and license_ok and jpeg and still:
                gbif_images[species].append(url)
    # Save the mapping so it doesn't need to be generated again
    with open(MULTIMEDIA_MAP_CACHE, "w", encoding="utf-8") as f:
        json.dump(gbif_images, f, indent=2)
    logger.info(f"✅ Cached mapping to {MULTIMEDIA_MAP_CACHE}")

# === iNat Fetcher ===
def fetch_inat(species_name, max_needed):
    urls = []
    page = 1
    
    # Iteratively fetch pages of results until we reach the max_needed
    while len(urls) < max_needed:
        try:
            # Attempt a request to the inaturalist API
            r = requests.get("https://api.inaturalist.org/v1/observations", params={
                "taxon_name": species_name,
                "quality_grade": "research",
                "photo_license": "CC0,CC-BY,CC-BY-NC",
                "per_page": 200,
                "page": page
            }, headers={"User-Agent": USER_AGENT}, timeout=20)
            # If we get a 429 code, we've been throttled so wait 10 seconds
            if r.status_code == 429:
                time.sleep(10)
                continue
            
            # If we get something else thats not 200, something went really wrong, or we ran out of results. Give up for now.
            if r.status_code != 200:
                break
            
            # If we run out of results, break and return the urls
            if len(r.json().get("results", [])) == 0:
                break
            
            # Loop through each photo in the results and append them to the url list
            for obs in r.json().get("results", []):
                for photo in obs.get("photos", []):
                    # Use original full size URL
                    url = photo["url"].replace("square", "original") \
                    .replace("small", "original") \
                    .replace("medium", "original") \
                    .replace("large", "original") \
                    
                    if url.lower().endswith(".jpg"):
                        urls.append(url)
                        if len(urls) >= max_needed:
                            break
            # Increment the page number
            page += 1
        # For all other exceptions, break out of the loop and return the urls so far
        except:
            break
    return urls

# Download Images for a species attempting GBIF First, then falling back to iNaturalist
def download_for_species(species, total_progress, total_task, species_progress, species_task_id):
    # Create the output directory if it doesn't already exist
    folder_name = species.replace(" ", "_")
    out_dir = OUTPUT_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a shutdown is triggered, remove the task
    if shutdown_flag.is_set():
        logger.info(f"⏹️  Skipping {species} due to shutdown signal")
        species_progress.remove_task(species_task_id)
        return species
    
    # Load previously downloaded URLs to ensure we don't download duplicates
    existing_urls = set()
    if success_log.exists():
        with open(success_log) as f:
            for line in f:
                if line.strip().endswith(f"{species}"):
                    existing_urls.add(line.split()[0])

    # Check existing image files and update the count
    existing_imgs = sorted([p for p in out_dir.glob("*.jpg") if p.name[:4].isdigit()])
    idx = len(existing_imgs) + 1
    count = len(existing_imgs)
    species_progress.update(species_task_id, completed=count)

    # If we have reached the target number of images, remove the task and return the populated species list 
    if count >= TARGET_IMAGES:
        species_progress.remove_task(species_task_id)
        total_progress.update(total_task, advance=1)
        return species
    
    # Attempt to download and save a series of image URLs
    def download_and_save_images(urls):
        nonlocal count, idx
        for url in urls:
            # Stop iterating if we reach the target
            if count >= TARGET_IMAGES:
                break
            # Skip the URL if its already been used
            if url in existing_urls:
                continue
            # Break out of the loop if shutdown triggered
            if shutdown_flag.is_set():
                break
            try:
                # Attempt to request the extracted image URL
                resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
                
                # If we don't get a 200 the URL is likely broken, skip this image URL
                if resp.status_code != 200:
                    continue
                
                # Decode, save and then release the image from memory
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                
                # Reject small images
                if img.width < 500 or img.height < 500:
                    with open(failed_log, "a") as f:
                        f.write(f"{url} {species} [too_small: {img.width}x{img.height}]\n")
                    continue
                    
                img.save(out_dir / f"{idx:04d}.jpg")
                img.close()
                
                # Increment counters
                idx += 1
                count += 1
                
                # Update the progress
                species_progress.update(species_task_id, completed=count)
                
                # Update the success log
                with open(success_log, "a") as f:
                    f.write(f"{url} {species}\n")
            except:
                # Something went wrong with this image, update the failure log
                with open(failed_log, "a") as f:
                    f.write(f"{url} {species}\n")

    # 1. Attempt GBIF First, as it has higher image quality
    # Attempt to extract images from the local GBIF download first
    urls = gbif_images.get(species) or []
    if urls:
        download_and_save_images(urls)
    else:
        logger.warning(f"No GBIF images found for {species}")
    
    # If a shutdown is triggered between data tasks, remove the task
    if shutdown_flag.is_set():
        logger.info(f"⏹️  Skipping {species} due to shutdown signal")
        species_progress.remove_task(species_task_id)
        return species
        
    logger.warning(f"GBIF images exhausted for {species} ({len(urls)})")
    # 2. Attempt iNaturalist next
    # If we still havent reached the target amount of images, move onto iNaturalist
    if count < TARGET_IMAGES:
        # Update the source
        species_progress.update(species_task_id, source="iNaturalist")
        # Fetch the number of iNaturalist URLs we require
        inat_urls = fetch_inat(species, TARGET_IMAGES - count)
        # Loop through the fetch iNaturalist URLs
        download_and_save_images(inat_urls)

    # Remove the current species task and update the total progress
    species_progress.remove_task(species_task_id)
    total_progress.update(total_task, advance=1)
    return species

# Setup progress bars
total_progress = Progress(
    TextColumn("[blue]Species Progress:"),
    BarColumn(),
    "{task.completed}/{task.total}",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
)

species_progress = Progress(
    TextColumn("[blue]{task.fields[species]}", justify="right"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TextColumn("[blue] ({task.fields[source]})", justify="left"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
    refresh_per_second=1,
)

progress_group = Group(
    Panel(total_progress, title="Overall"),
    Panel(species_progress, title="Species in Progress")
)

# === Set up threading tasks ===
pending = deque()
completed_count = 0
species_to_task_id = {}

logger.info("🔍 Scanning existing species image folders...")

# Loop through folders and determine how many species still have not met the target number of images
for species in accepted_species:
    folder_name = species.replace(" ", "_")
    out_dir = OUTPUT_DIR / folder_name
    existing_imgs = [p for p in out_dir.glob("*.jpg") if p.name[:4].isdigit()]
    
    # If there are enough images, increment the completed count
    if len(existing_imgs) >= TARGET_IMAGES:
        completed_count += 1
    # Otherwise add it as a pending task
    else:
        pending.append(species)

# Log the results
logger.info(f"📸 {completed_count} species already have enough images")
logger.info(f"⏳ {len(pending)} species need more images")

# === Setup layout with two named regions ===
layout = Layout()
layout.split_column(
    Layout(name="overall", size=3),
    Layout(name="species", ratio=1)
)

# Bind progress bars to layout regions
layout["overall"].update(Panel(total_progress, title="Overall Progress"))
layout["species"].update(Panel(species_progress, title="Species in Progress"))

# Live update loop for the progress bars
with Live(layout, console=console, refresh_per_second=2):
    # Setup total progress
    total_task = total_progress.add_task("All species", total=len(accepted_species))
    total_progress.update(total_task, advance=completed_count)

    # Add species progress tasks
    for species in pending:
        task_id = species_progress.add_task(
            "Queued",
            species=species,
            source="GBIF",
            total=TARGET_IMAGES,
            completed=0,
        )
        species_progress.start_task(task_id)
        species_to_task_id[species] = task_id

    # Future to species and task id mappings
    futures_to_species = {}
    futures_to_task_id = {}

    # Execute tasks in a threadpool
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        try:
            # Initial batch
            for _ in range(min(THREADS, len(pending))):
                species = pending.popleft()
                urls = gbif_images.get(species, [])
                task_id = species_to_task_id[species]
                species_progress.update(task_id, description="Downloading")

                future = executor.submit(
                    download_for_species,
                    species,
                    total_progress, total_task,
                    species_progress, task_id
                )
                futures_to_species[future] = species
                futures_to_task_id[future] = task_id

            # While there are current tasks running or pending
            while futures_to_species or pending:
                # If shutdown was triggered, bail out
                if shutdown_flag.is_set():
                    logger.warning("⏹️  Shutdown signal received. Cancelling remaining tasks.")
                    break

                # Wait for any future to complete (timeout avoids hang if none are ready)
                if futures_to_species:
                    done, _ = wait(futures_to_species.keys(), timeout=1, return_when=FIRST_COMPLETED)
                else:
                    done = []

                # Loop through the completed tasks
                for future in done:
                    species = futures_to_species.pop(future)
                    task_id = futures_to_task_id.pop(future)

                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"[{species}] Error: {e}")

                    # Schedule the next task to complete
                    if pending:
                        next_species = pending.popleft()
                        urls = gbif_images.get(next_species, [])
                        next_task_id = species_to_task_id[next_species]
                        species_progress.update(next_task_id, description="Downloading")

                        next_future = executor.submit(
                            download_for_species,
                            next_species,
                            total_progress, total_task,
                            species_progress, next_task_id
                        )
                        futures_to_species[next_future] = next_species
                        futures_to_task_id[next_future] = next_task_id

        # Catch any unexpected errors and log them
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        # Last, shut down the thread pool
        finally:
            logger.info("⚙️ Shutting down thread pool...")
            executor.shutdown(wait=False, cancel_futures=True)

# Yay, all images completed!
logger.info("✅ All species processed.")
