import re
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup


def get_wikipedia_page(page_input: str, data_dir: str):
    page = format_page(page_input)
    try:
        print("Getting: ", page)
        data_dir = os.path.join(data_dir, "docs")
        text = get_page_from_url(page)
        write_file(os.path.join(data_dir, page_input), text)
    except Exception as e:
        page_input = "failed"
        print("Failed to retrieve: ",  page)
        s = str(e)
        print(s)
        pass
    return page_input


def get_page_from_url(page):
    source = urlopen('https://en.wikipedia.org/wiki/' + page).read()
    soup = BeautifulSoup(source, 'lxml')
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.text
    text = re.sub(r'\[.*?\]+', '', text)
    text = text.replace('\n', '')
    return text


def write_file(data_dir, text):
    try:
        text.encode('utf-8', errors='ignore')
        with open(data_dir, "w", encoding="utf-8") as f:
            f.write(text)
        f.close()
    except Exception as e:
        print("Failed to write page")
        s = str(e)
        print(s)
        pass


def format_page(page: str):
    ret = page.replace('-LRB-', '(')
    ret = ret.replace('-RRB-', ')')
    ret = ret.replace('-COLON-', ':')
    return ret
