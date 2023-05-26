from bing_image_downloader import downloader

QUERIES = ["Male Mallard Duck", "Male Mandarin duck", "Male Whistling duck", "Male Nothern shoveler", "Male allier white duck", "Male tufted duck", "Male gadwall", "Female goosander"]
OUTPUT_DIR = 'dataset_images'
NB_IMAGES = 50

for query in QUERIES:
    downloader.download(
        query, NB_IMAGES, OUTPUT_DIR,
        adult_filter_off=False, force_replace=False, timeout=15
    )
