import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import urllib
import sys
import feedparser
import os
from dotenv import load_dotenv
from requests_html import HTMLSession
import re
from urllib.parse import urlparse, urljoin


# Load .env file
load_dotenv()
# --------------------------
# CONFIGURATION
# --------------------------
GROUPME_BOT_ID = os.getenv("GROUPME_BOT_ID")

# Best Buy AMD search (scraping example)
BESTBUY_API_KEY = os.getenv("BESTBUY_API_KEY")

MODEL="facebook/bart-large-cnn"
# MODEL="microsoft/phi-3-mini-4k-instruct"
GOOGLE_SEARCH_TERM='"AMD processors" "AMD news"'

# microsoft/phi-3-mini-4k-instruct #? Maybe can provide better summarization

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
# model = AutoModelForCausalLM.from_pretrained(MODEL)

CATEGORY_IDS = {
    "laptops": "abcat0502000",
    "gaming_laptops": "pcmcat287600050003",
    "desktops": "abcat0501000",
    # "processors": "abcat0507010",
    # "graphics_cards": "abcat0507002",
}

USER_AGENT = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# --------------------------
# HELPER FUNCTIONS
# --------------------------

def shorten_url(url):
    api_url = "https://tinyurl.com/api-create.php"
    response = requests.get(api_url, params={"url": url})
    if response.status_code == 200:
        return response.text
    return url  # fallback to original if something fails

def fetch_amd_news(limit=5, scrape_articles=True):
    print("Fetching News...")
    """
    Fetch latest AMD news from Google News RSS and return list of dicts with shortened links.
    """
    feed_url = "https://news.google.com/rss/search?q=AMD&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    news_list = []
    for entry in feed.entries[:limit]:
        article_content = scrape_article(entry.link) if scrape_articles else ""
        news_list.append({
            "title": entry.title,
            "link": entry.link,
            "content": article_content
        })

    return news_list


def fetch_bestbuy_amd_all(keyword="AMD", limit_per_category=10):
    """
    Fetch AMD products for all categories, available in-store and online.
    Returns a dictionary with categories as keys and sorted product lists.
    """
    print("Fetching Best Buy Deals...")
    all_products = {cat: [] for cat in CATEGORY_IDS.keys()}

    for category, category_id in CATEGORY_IDS.items():
        url = (
            f"https://api.bestbuy.com/v1/products("
            f"categoryPath.id={category_id}&search={keyword}&onSale=true&"
            f"inStoreAvailability=true&onlineAvailability=true&inStorePickup=true"
            f")?format=json&show=sku,name,regularPrice,salePrice,percentSavings,dollarSavings,"
            f"inStoreAvailability,onlineAvailability,shortDescription,longDescription,"
            f"manufacturer,customerReviewAverage,largeFrontImage,url&pageSize=100&page=1&apiKey={BESTBUY_API_KEY}"
        )
        try:
            response = requests.get(url)
            data = response.json()
            products = data.get("products", [])
            # print(products)
            products = [p for p in products if "refurbished" not in p.get("name").lower() ]
            # Filter out items with 0% discount
            products = [p for p in products if float(p.get("percentSavings", 0)) > 0]
            
            # Sort by percentSavings descending
            products.sort(key=lambda x: float(x.get("percentSavings", 0)), reverse=True)


            # Limit results per category
            products = products[:limit_per_category]

            all_products[category] = products

        except Exception as e:
            print(f"Error fetching Best Buy AMD {category}: {e}")
            all_products[category] = []

    return all_products



def scrape_google_news_search(query, max_items=10, debug=False):
    url = f"https://www.google.com/search?q={requests.utils.requote_uri(query)}&tbm=nws&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers=USER_AGENT, timeout=10)
    # if debug:
        # print("Status:", r.status_code)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()
    
    # Google news results often contain <a> tags inside result cards, maybe with href
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # restrict to plausible news result links
        if href.startswith("/url?") or "news.google.com" in href or "articles/" in href:
            full = urljoin("https://www.google.com", href)
            if full in seen:
                continue
            seen.add(full)
            
            title = a.get_text(strip=True)
            # sometimes Google wraps link in /url?q=<real_url>&...
            m = re.search(r"/url\?q=([^&]+)", href)
            real = None
            if m:
                real = requests.utils.unquote(m.group(1))
            else:
                real = full

            results.append({"title": title, "url": real} if title != "Maps" else None)
            if len(results) >= max_items:
                break
    return results


def fetch_amd_news_with_content(limit=5):
    items = scrape_google_news_search(GOOGLE_SEARCH_TERM, max_items=10, debug=False)
    # print(items)
    results = []
    for item in items[0:6]:
        if item is None:
            continue
        title = item.get("title", "")
        url = item.get("url", "")
        article_text = ""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}  # avoid being blocked
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
                
            # Remove all script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
                # Remove any remaining unwanted tags
            # for tag in soup(["nav", "footer", "header", "aside"]):
            #     tag.decompose()
                
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            results.append({
                "title": title, 
                "link": url,
                "content": article_text
            })
            
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return results
        
def fetch_amd_news_with_content_rss(limit=5):
    """Fetch AMD news with actual publisher URLs and scraped content."""
    feed_url = "https://news.google.com/rss/search?q=AMD&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    
    news_list = []

    for entry in feed.entries[:limit]:
        title = entry.title
        
        # Extract the real URL from the entry ID or link
        real_url = entry.link  # default to the link
        print(get_final_url(real_url))

        
        # Try scraping the article content
        article_text = ""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}  # avoid being blocked
            response = requests.get(real_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove all script and style tags
            for script in soup(["script", "style"]):
                script.decompose()

            # Remove any remaining unwanted tags
            for tag in soup(["nav", "footer", "header", "aside"]):
                tag.decompose()
            # print(soup)

            # Common selectors for news content
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            
            # optional: truncate if too long
            if len(article_text) > 3000:
                article_text = article_text[:3000] + "..."
        except Exception as e:
            print(f"Failed to scrape {real_url}: {e}")
        
        news_list.append({
            "title": title,
            "link": real_url,
            "content": article_text
        })
    
    return news_list
USER_AGENT = {"User-Agent": "Mozilla/5.0"}


def get_final_url(url, timeout=10, debug=False):
    source_domain = urlparse(url).netloc
    try:
        # 1) HEAD first (fast) and follow redirects
        try:
            head = requests.head(url, headers=USER_AGENT, allow_redirects=True, timeout=timeout)
            final = head.url
            if debug:
                print("HEAD ->", head.status_code, head.url)
                if getattr(head, "history", None):
                    print("HEAD history:", [(h.status_code, h.url) for h in head.history])
        except Exception as e_head:
            if debug:
                print("HEAD failed:", e_head)
            head = None
            final = None

        # If head gave an external final URL, return it
        if final and urlparse(final).netloc and urlparse(final).netloc != source_domain:
            return final

        # 2) GET and follow redirects (some servers block HEAD or respond differently)
        resp = requests.get(url, headers=USER_AGENT, allow_redirects=True, timeout=timeout)
        if debug:
            print("GET ->", resp.status_code, resp.url)
            if resp.history:
                print("GET history:", [(h.status_code, h.url) for h in resp.history])

        # If GET final location is external, return
        if resp.url and urlparse(resp.url).netloc and urlparse(resp.url).netloc != source_domain:
            return resp.url

        # 3) If we're still on the source domain, inspect HTML for meta refresh / og:url / canonical / anchors
        html = resp.text or ""
        extracted = extract_from_html(html, source_domain)
        if extracted:
            if debug:
                print("Extracted from HTML:", extracted)
            return extracted

        # 4) Last resort: sometimes the real article is in a <script> as encoded url param — try a query param like 'url='
        m = re.search(r"[?&](?:url|u)=((?:https?%3A%2F%2F)[^&]+)", url)
        if m:
            import urllib.parse
            decoded = urllib.parse.unquote(m.group(1))
            if debug:
                print("Found encoded param url ->", decoded)
            if _looks_external(decoded, source_domain):
                return decoded

        # 5) Nothing found — return the GET final url (even if still news.google.com) as fallback
        return resp.url or url

    except Exception as e:
        if debug:
            print("Error resolving URL", url, ":", e)
        return url

def summarize_text(text, title):
    if "This website is using a security service to protect itself" in text:
        return ""
    """Summarize text using local Flan-T5 model."""

    # print("Summarizing...", title)
    input_text = "Summarize: \n" + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # outputs = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    outputs = model.generate(inputs, max_length=512, min_length=50, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def post_to_groupme(message):
    """Send message to GroupMe group."""
    payload = {"bot_id": GROUPME_BOT_ID, "text": message}
    print("Posting to GroupMe...")
    try:
        response = requests.post("https://api.groupme.com/v3/bots/post", json=payload)
        if response.status_code == 202:
            print("Message sent successfully!")
        else:
            print("Failed to send message:", response.text)
    except Exception as e:
        print("Error posting to GroupMe:", e)

def gather_news_articles(news_results):
    news_message=""
    news_message+= "AMD NEWS\n"
    news_message+= f"="*8 + "\n"
    print(f"Summarizing {len(news_results)} articles...")
    for article in news_results:
        if not article['content'] or "Please enable" in article['content']:
            continue
        title = article["title"]
        # print(f"Title: {article['title']}")
        # print(f"Link: {article['link']}")
        # print(f"Content: {article['content'][:200]}...")  # print first 200 chars
        # print(summarize_text(article['content']))
        summarized= summarize_text(article['content'], title)
        # print("-" * 40)
        # print("\n")
        # news_headline=title
        news_message += (
            f"Title: {title}\n"
            # f"Source: {article['link']}\n"
            # f"Content: {article.get('content')}\n"
            f"Summary: {summarized}\n"
            f"Link: {article.get('link')}\n\n"
        )
    return news_message

def gather_bby_deals(amd_products):
    str_multiplier=8
    message = ""  
    message+= "BEST BUY AMD DEALS"
    
    for category, products in amd_products.items():
        # print(f"\n=== {category.upper()} ===")
        message += f"\n{"="*str_multiplier} {category.upper()} {"="*str_multiplier}\n"
        for p in products[:5]:
            message += (
                f"{p.get('name')}\n"
                f"Sale Price: ${p.get('salePrice')}\n"
                f"Customer Rating: {p.get('customerReviewAverage', 'N/A')} / 5\n"
                f"Discount: {p.get('percentSavings')}% off\n"
                f"SKU: {p.get('sku')}\n"
                # f"Link: {shorten_url(p.get('url'))}\n\n"
                f"Link: {p.get('url')}\n\n"
            )
    return message

# --------------------------
# MAIN LOGIC
# --------------------------
def main():
    # news_list = fetch_amd_news()
    # u_input=input("what to search? >: ")
    
    # amd_products = fetch_bestbuy_amd_all(keyword="AMD", limit_per_category=10)
    # amd_bby_deals = gather_bby_deals(amd_products)
    # post_to_groupme(amd_bby_deals)
   
    news_results = fetch_amd_news_with_content()
    news_message= gather_news_articles(news_results)
    print(news_message)
    
    post_to_groupme(news_message)




  


    


# Example usage:
if __name__ == "__main__":
    main()
