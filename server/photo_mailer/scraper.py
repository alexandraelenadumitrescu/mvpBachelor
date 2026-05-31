import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_employees(url: str) -> list[dict]:
    """
    Scrape {name, email, photo_url} from a page with .employee-card divs.
    Expected card structure:
      <div class="employee-card">
        <img src="PHOTO_URL">
        <h3>Name</h3>
        <p>email@domain.com</p>
      </div>
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    employees = []
    for card in soup.select(".employee-card"):
        img   = card.find("img")
        name  = card.find("h3")
        email = card.find("p")

        if not (img and name and email):
            continue

        photo_url = img.get("src", "")
        if photo_url.startswith("/"):
            photo_url = urljoin(url, photo_url)

        employees.append({
            "name":      name.get_text(strip=True),
            "email":     email.get_text(strip=True),
            "photo_url": photo_url,
        })

    return employees
