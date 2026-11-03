"""
Download the project's datasets from their official sources into this data/ folder.

Usage (from the project's main folder):
    python data/download_data.py                 # PlantVillage + PlantDoc
    python data/download_data.py --plantvillage  # only PlantVillage (~1 GB)
    python data/download_data.py --plantdoc      # only PlantDoc

Requires git to be installed.
"""
import argparse
import os
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PLANTVILLAGE_URL = 'https://github.com/spMohanty/PlantVillage-Dataset'
PLANTDOC_URL = 'https://github.com/pratikkayal/PlantDoc-Dataset'


def count_images(folder):
    exts = ('.jpg', '.jpeg', '.png')
    return sum(1 for _, _, files in os.walk(folder) for f in files if f.lower().endswith(exts))


def download_plantvillage():
    repo = os.path.join(DATA_DIR, 'PlantVillage-Dataset')
    color = os.path.join(repo, 'raw', 'color')
    if os.path.isdir(color) and len(os.listdir(color)) == 38:
        print('PlantVillage already downloaded.')
    else:
        print('Downloading PlantVillage (raw/color only, ~1 GB)...')
        # sparse clone: fetch only the colour images, not the grayscale/segmented copies
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                        PLANTVILLAGE_URL, repo], check=True)
        subprocess.run(['git', '-C', repo, 'sparse-checkout', 'set', 'raw/color'], check=True)
    print(f'PlantVillage: {len(os.listdir(color))} classes, {count_images(color)} images (expected 38, 54305)')


def download_plantdoc():
    repo = os.path.join(DATA_DIR, 'PlantDoc-Dataset')
    if os.path.isdir(os.path.join(repo, 'test')):
        print('PlantDoc already downloaded.')
    else:
        print('Downloading PlantDoc...')
        subprocess.run(['git', 'clone', '--depth', '1', PLANTDOC_URL, repo], check=True)
    print(f'PlantDoc: {count_images(repo)} images')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plantvillage', action='store_true', help='download only PlantVillage')
    parser.add_argument('--plantdoc', action='store_true', help='download only PlantDoc')
    args = parser.parse_args()
    both = not (args.plantvillage or args.plantdoc)
    try:
        if both or args.plantvillage:
            download_plantvillage()
        if both or args.plantdoc:
            download_plantdoc()
    except FileNotFoundError:
        sys.exit('git is not installed. Install it from https://git-scm.com and try again.')
    except subprocess.CalledProcessError as e:
        sys.exit(f'Download failed ({e}). Check your internet connection and try again.')


if __name__ == '__main__':
    main()
