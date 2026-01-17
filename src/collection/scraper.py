from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import csv
from pathlib import Path


def run_scraper(seed_url, max_steps=50):
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_FILE = OUTPUT_DIR / "recommendation_walk.csv"

    options = Options()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    wait = WebDriverWait(driver, 30)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "title", "url"])

    driver.get(seed_url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
    time.sleep(4)

    current_url = seed_url

    for step in range(max_steps):
        for _ in range(2):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1)

        links = driver.find_elements(By.XPATH, "//a[@href]")
        candidates = []

        for link in links:
            try:
                href = link.get_attribute("href")
                title = link.get_attribute("title") or link.get_attribute("aria-label")
            except:
                continue

            if not href or "watch?v=" not in href or href == current_url:
                continue
            if not title or len(title.strip()) < 15:
                continue

            candidates.append((title.strip(), href))

        if not candidates:
            break

        title, next_url = candidates[0]

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, title, next_url])

        current_url = next_url
        driver.get(next_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "ytd-app")))
        time.sleep(3)

    driver.quit()
    return OUTPUT_FILE
