import requests
from bs4 import BeautifulSoup

# Predefined URL to fetch data from
PREDEFINED_URL = 'https://www.livemint.com/market/stock-market-news'

def scrape_data(ticker):
    url = PREDEFINED_URL + ticker
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return extract_data(soup)

def extract_data(soup):
    news_data = []
    news_items = soup.find_all('div', class_='news-item')
    for item in news_items:
        title_tag = item.find('h2', class_='news-title')
        link_tag = title_tag.find('a', href=True) if title_tag else None
        if title_tag and link_tag:
            title = title_tag.text.strip()
            url = link_tag['href']
            news_data.append({
                'title': title,
                'url': url
            })
    return news_data