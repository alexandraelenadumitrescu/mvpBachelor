import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_employees(url: str) -> list[dict]:
    """
    Scrape {name, email, photo_url} from a page.
    Supports two card formats:
      - .employee-card  (mock page)
      - article.speaker-card  (nexus.html / conference pages)
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    employees = []

    # Format 1: .employee-card (mock page)
    for card in soup.select(".employee-card"):
        img   = card.find("img")
        name  = card.find("h3")
        email = card.find("p")
        if not (img and name and email):
            continue
        photo_url = img.get("src", "")
        if photo_url and not photo_url.startswith(("http://", "https://")):
            photo_url = urljoin(url, photo_url)
        employees.append({
            "name":      name.get_text(strip=True),
            "email":     email.get_text(strip=True),
            "photo_url": photo_url,
        })

    if employees:
        return employees

    # Format 2: article.speaker-card (nexus.html)
    for card in soup.select("article.speaker-card"):
        img        = card.find("img", class_="speaker-photo")
        name_tag   = card.find(class_="speaker-name")
        email_tag  = card.find(class_="speaker-email")

        if not (img and name_tag and email_tag):
            continue

        email = email_tag.get("data-email") or email_tag.get_text(strip=True)
        photo_url = img.get("src", "")
        if photo_url and not photo_url.startswith(("http://", "https://")):
            photo_url = urljoin(url, photo_url)

        employees.append({
            "name":      name_tag.get_text(strip=True),
            "email":     email,
            "photo_url": photo_url,
        })

    return employees
