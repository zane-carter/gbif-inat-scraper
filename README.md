# GBIF & iNaturalist Image Scraper

Downloads publicly licensed images of plant species from GBIF and iNaturalist. Useful for training machine learning models for plant classification.

## Features

- ✅ Queries GBIF and iNaturalist using species names
- 🎯 Filters images by license (Creative Commons), type (StillImage), and format (.jpg)
- 📁 Saves structured folders by species
- 🧾 Logs image counts and missing species for gap-filling

## Requirements

- Python 3.8+
- `requests`
- `pandas`
- `tqdm`

Install dependencies:

```bash
pip install -r requirements.txt
