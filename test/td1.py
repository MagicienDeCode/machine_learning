import requests
from bs4 import BeautifulSoup
import os

url = "https://www.nature.com/articles/s41467-025-61671-8"
urls_list = [url,url]

save_dir = "./review_files/"
# create dir if not exists
os.makedirs(save_dir, exist_ok=True)

def download_peer_review_files(url):
    print(f"Downloading peer review files from {url}")
    article_id = url.split("/articles/")[-1]
    pdf_path = os.path.join(save_dir, f"{article_id}_peer_review.pdf")
    if os.path.exists(pdf_path):
        print(f"Peer review file already exists at {pdf_path}. Skipping download.")
        return
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        link = soup.find('a', class_ = "print-link", string="Transparent Peer Review file")
     
        if link and link.has_attr("href"):
            pdf_response = requests.get(link['href'])
            # This is typically used for non-text files like PDFs, images, etc. for writing in binary mode ('wb'). 
            with open(pdf_path, 'wb') as f: f.write(pdf_response.content)
            print(f"downloaded peer review file to {pdf_path}")
        else:
            print(f"No peer review file found. please check link for {url}")
    except Exception as e:
        print(f"An error occurred during download peer review for {url} : {e}")

def download_peer_review_files_from_list(urls):
    for url in urls: 
        download_peer_review_files(url)

if __name__ == "__main__":
    download_peer_review_files_from_list(urls_list)

