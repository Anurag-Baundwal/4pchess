# move_fetcher.py
# v12.1 (Removes initial state file write)

import time
import random
import os
import argparse
import sys
import pickle
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
GAME_STATE_FILE = "game_state.json"
POLL_INTERVAL_SECONDS = 0.1
COOKIE_FILE = "chess_cookies.pkl"

CASTLING_KING_MOVES = {
    ('R', 'O-O'):   'h1-j1',
    ('R', 'O-O-O'): 'h1-f1',
    ('B', 'O-O'):   'a7-a5',
    ('B', 'O-O-O'): 'a7-a9',
    ('Y', 'O-O'):   'g14-e14',
    ('Y', 'O-O-O'): 'g14-i14',
    ('G', 'O-O'):   'n8-n10',
    ('G', 'O-O-O'): 'n8-n6',
}

# Unique RGB values for each player's clock from chess.com
CLOCK_COLOR_MAP = {
    'R': "rgb(191, 59, 68)",
    'B': "rgb(65, 132, 191)",
    'Y': "rgb(191, 148, 38)",
    'G': "rgb(78, 145, 97)",
}
# -----------------------------------------------

def load_cookies(driver, cookie_file):
    """Loads cookies from a file into the WebDriver session."""
    print(f"Attempting to load cookies from '{cookie_file}'...")
    try:
        with open(cookie_file, 'rb') as file:
            cookies = pickle.load(file)
        driver.get("https://www.chess.com")
        for cookie in cookies:
            if 'expiry' in cookie:
                del cookie['expiry']
            driver.add_cookie(cookie)
        print("✅ Cookies loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"⚠️ Cookie file not found. Please run 'save_cookies.py' first.")
        return False
    except Exception as e:
        print(f"❌ An error occurred while loading cookies: {e}")
        return False

def _parse_clock_str(time_str: str) -> float:
    """Converts a 'M:SS' or 'S.s' string to total seconds."""
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return float(minutes * 60 + seconds)
        else:
            return float(time_str)
    except (ValueError, IndexError):
        print(f"Warning: Could not parse clock string '{time_str}'. Defaulting to 0.")
        return 0.0

def _fetch_clock_times(driver) -> dict:
    """
    Scrapes the page for all four player clocks using a robust content-first approach.
    It finds candidate spans and validates their text to find the time.
    """
    clocks_found = {}
    time_format_regex = re.compile(r"^\d{1,2}:\d{2}(\.\d)?$|^\d+\.\d$")

    try:
        clock_elements = driver.find_elements(By.CSS_SELECTOR, ".clock-component.playerbox-clock")

        for clock_element in clock_elements:
            style = clock_element.get_attribute('style')
            time_text = ""

            all_spans_in_clock = clock_element.find_elements(By.TAG_NAME, "span")
            for span in all_spans_in_clock:
                candidate_text = span.get_attribute('textContent').strip()
                if time_format_regex.match(candidate_text):
                    time_text = candidate_text
                    break

            if time_text:
                for color_char, rgb_val in CLOCK_COLOR_MAP.items():
                    if rgb_val in style:
                        clocks_found[color_char] = _parse_clock_str(time_text)
                        break
            else:
                pass

    except StaleElementReferenceException:
        print("Clock element became stale, will retry on next poll.")
        return {}
    except Exception as e:
        print(f"An error occurred in _fetch_clock_times: {e}")

    return clocks_found


def _standardize_move_notation(move_notation: str, turn_index: int) -> str:
    """
    Converts various PGN-style notations from chess.com into the simple
    'from-to' algebraic format the engine expects.
    """
    turn_order = ['R', 'B', 'Y', 'G']
    player_char = turn_order[turn_index % 4]
    if move_notation in ('O-O', 'O-O-O'):
        lookup_key = (player_char, move_notation)
        if lookup_key in CASTLING_KING_MOVES:
            return CASTLING_KING_MOVES[lookup_key]
        else:
            print(f"!!! ERROR: Could not translate castling move for {lookup_key}")
            return move_notation

    coords = re.findall(r'[a-n]\d{1,2}', move_notation)
    if len(coords) == 2:
        return f"{coords[0]}-{coords[1]}"

    if '-' in move_notation:
        return move_notation.split('=')[0].replace('+', '').replace('#', '')

    print(f"!!! WARNING: Unhandled move notation format: '{move_notation}'. Passing it as is.")
    return move_notation

def fetch_and_sync_moves(driver):
    """
    Finds moves on the page, formats them, and writes them to a JSON file.
    """
    last_sent_moves = []
    moves_list_selector = ".moves-moves-list"

    print("\n--- Move Fetcher is running ---")
    print(f"Polling for moves every {POLL_INTERVAL_SECONDS} seconds and writing to '{GAME_STATE_FILE}'.")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            if "chess.com/variants" not in driver.current_url:
                print("No longer on a chess.com game page. Stopping.")
                break

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, moves_list_selector))
            )

            move_elements = driver.find_elements(By.CSS_SELECTOR, ".moves-pointer")

            current_moves = []
            for i, move_element in enumerate(move_elements):
                title = move_element.get_attribute('title')
                if title:
                    move_notation = title.split('•')[0].strip()

                    if not move_notation or move_notation == '#' or len(move_notation) <= 1:
                        continue
                    else:
                        standardized_move = _standardize_move_notation(move_notation, i)
                        current_moves.append(standardized_move)

            if current_moves != last_sent_moves:
                detection_time = time.time()
                current_clocks = _fetch_clock_times(driver)
                print(f"Detected new state with {len(current_moves)} moves. Clocks: {current_clocks}. Writing to file...")

                payload = {
                    'url': driver.current_url,
                    'moves': current_moves,
                    'clocks': current_clocks,
                    'detection_timestamp': detection_time
                }
                
                temp_file = GAME_STATE_FILE + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(payload, f)
                os.replace(temp_file, GAME_STATE_FILE)
                
                last_sent_moves = current_moves

            base_interval = POLL_INTERVAL_SECONDS
            jitter = random.uniform(-POLL_INTERVAL_SECONDS/10, POLL_INTERVAL_SECONDS/10)
            time.sleep(base_interval + jitter)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Will retry in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches Chrome, loads login cookies, navigates to a game URL, and fetches moves.")
    parser.add_argument('--url', required=True, help="The full URL of the chess.com game.")
    args = parser.parse_args()

    options = Options()
    options.add_argument("--headless")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    print("\nInitializing Selenium WebDriver...")
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("Done initializing WebDriver.")
    except Exception as e:
        print(f"\n--- ❌ CRITICAL ERROR INITIALIZING WEBDRIVER ❌ ---\nError: {e}")
        sys.exit(1)

    if not load_cookies(driver, COOKIE_FILE):
        driver.quit()
        sys.exit(1)

    try:
        print(f"\nNavigating to game URL: {args.url}")
        driver.get(args.url)
        time.sleep(3)
    except Exception as e:
        print(f"Could not navigate to the URL. Error: {e}")
        driver.quit()
        sys.exit(1)

    fetch_and_sync_moves(driver)

    print("Script finished. Closing browser.")
    driver.quit()