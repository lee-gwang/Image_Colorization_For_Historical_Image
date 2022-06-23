from bs4 import BeautifulSoup
import urllib
import requests
import os
from PIL import Image
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='scrape getty images')
parser.add_argument('--keyword', type=str, default='', help='keyword list')
parser.add_argument('--output_dir', type=str, default='/home/data/gettyimages', help='output_dir')
args = parser.parse_args()

def open_image(idx, url_):
    try:
        img = Image.open(requests.get(url_, stream = True).raw)
        img.save(f'{args.output_dir}/{args.keyword}/img{idx}.jpg')
    except:
        pass

def scrape(keyword):
    os.makedirs(f"{args.output_dir}/{keyword}", exist_ok=True)

    url_list = []

    # 전체 page crawling
    for page in tqdm(range(1,101)):
        try:
            url = f'https://www.gettyimages.com/search/2/image?family=creative&phrase={keyword}&page={page}'
            html = urllib.request.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')
            get_urls = soup.select("source")

            # url_list : 전체 page의 image url
            for i in range(len(get_urls)):
                url_list.append(str(get_urls[i]).split('"')[1].replace("s=612x612",""))
        except:
            break

    # keyword에 대한 image 저장 (multithreading)
    Parallel(n_jobs=-1, backend='threading')(delayed(open_image)(idx, url_)\
        for idx, url_ in tqdm(enumerate(url_list), total=len(url_list)))
    
if __name__ == '__main__':
    scrape(args.keyword)