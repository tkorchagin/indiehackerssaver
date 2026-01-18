#!/usr/bin/env python3
"""Fetch Indie Hackers stories and export to Excel.

Usage:
    python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python scripts/fetch_stories.py
"""
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook
from pathlib import Path
from urllib.parse import urljoin
import os

BASE_URL = "https://www.indiehackers.com"
STORIES_URL = f"{BASE_URL}/stories"

def fetch_page(url: str):
    # Add fake user agent to avoid bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

def parse_stories(html_content, base_url):
    """
    Parses the HTML content to find stories and extract their data.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    stories = []
    
    # Based on the user's debug.html, stories are <a> tags with class 'slick-story database__story'
    story_elements = soup.select('a.database__story')
    
    for story in story_elements:
        try:
            # 1. Title
            title_elem = story.select_one('.slick-story__title')
            title = title_elem.get_text(strip=True) if title_elem else "No Title"

            # 2. Author
            author_elem = story.select_one('.slick-story__author')
            author = author_elem.get_text(strip=True) if author_elem else ""

            # 3. MRR
            mrr_elem = story.select_one('.database__story-mrr')
            mrr = mrr_elem.get_text(strip=True).replace('MRR', '').strip() if mrr_elem else ""

            # 4. Product Name
            # The product name is typically in the footer, as the last span
            # Structure: <footer> <span class="mrr">...</span> <span>ProductName</span> </footer>
            footer_elem = story.select_one('.slick-story__footer')
            product = ""
            if footer_elem:
                spans = footer_elem.find_all('span')
                if spans and len(spans) > 1:
                    # Usually the last span contains the product name
                    product = spans[-1].get_text(strip=True)
                elif spans and not mrr_elem:
                     # Fallback if only one span exists and it's not MRR selector
                    product = spans[0].get_text(strip=True)

            # 5. URL
            href = story.get('href')
            if href:
                full_url = urljoin(base_url, href)
            else:
                full_url = ""

            # 6. ID
            story_id = story.get('id', "")

            if title and full_url:
                stories.append({
                    'ID': story_id,
                    'Title': title,
                    'Author': author,
                    'Product': product,
                    'MRR': mrr,
                    'URL': full_url
                })
        except Exception as e:
            print(f"Error parsing story: {e}")
            continue
            
    return stories

def save_to_excel(stories, filename):
    """
    Saves the list of stories to an Excel file.
    """
    # Simply create a DataFrame-like structure using openpyxl
    wb = Workbook()
    ws = wb.active
    ws.title = "Stories"
    
    # Header
    headers = ["ID", "Title", "Author", "Product", "MRR", "URL"]
    ws.append(headers)
    
    for story in stories:
        ws.append([
            story.get('ID', ''),
            story.get('Title', ''),
            story.get('Author', ''),
            story.get('Product', ''),
            story.get('MRR', ''),
            story.get('URL', '')
        ])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    wb.save(filename)
    print(f"Saved {len(stories)} stories to {filename}")

def main():
    html = fetch_page(STORIES_URL)
    stories = parse_stories(html, BASE_URL)
    output_dir = Path("output")
    output_filename = output_dir / "stories.xlsx"
    save_to_excel(stories, str(output_filename))

if __name__ == "__main__":
    main()
