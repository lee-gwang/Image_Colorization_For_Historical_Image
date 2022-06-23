import os

keywords = ["military%20base", "small%20buildings", "war", "sports",
    "riots", "crowd", "city", "park", "korea%20military", "korea%20mountains"]
# keywords = ["park", "korea%20military", "korea%20mountains"]
# keywords = ["war", "sports", "riots", "crowd", "city"]

for keyword in keywords:
    os.system(f"python scrape.py --keyword={keyword}")