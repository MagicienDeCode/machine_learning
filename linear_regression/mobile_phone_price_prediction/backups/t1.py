import requests
from bs4 import BeautifulSoup
import os

url = "https://www.nature.com/articles/s41467-025-61671-8"

save_path = "./review_files/"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

def download_review_file(url):
    article_id = url.split('/articles/')[-1]
    pdf_path = os.path.join(save_path, f"{article_id}_peer_review.pdf")
    if os.path.exists(pdf_path):
        print(f"File already exists: {pdf_path}")
        return
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    link = soup.find("a", class_="print-link", string="Transparent Peer Review file")
    if link and link.has_attr("href"):
        pdf_response = requests.get(link["href"])
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_response.content)
        print(f"Downloaded: {pdf_path}")
    else:
        print(f"No Transparent Peer Review file found for {article_id}")

def download_review_file_with_list_of_urls(urls):
    for url in urls:
        download_review_file(url)

if __name__ == "__main__":
    download_review_file(url)
