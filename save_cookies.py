# save_cookies.py

import pickle
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
LOGIN_URL = "https://www.chess.com/login_and_go"
COOKIE_FILE = "chess_cookies.pkl"
# ---------------------

print("--- Chess.com Cookie Exporter ---")

# Initialize a standard WebDriver
try:
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    exit()

# Go to the login page
driver.get(LOGIN_URL)

print("\n--- ACTION REQUIRED ---")
print(f"A Chrome window has been opened. Please log in to your chess.com account.")
input("After you have successfully logged in, press Enter here to continue...")

# Once the user is logged in, get the cookies
try:
    # Adding a small delay to ensure all cookies are set after login
    time.sleep(2)
    cookies = driver.get_cookies()
    
    # Save the cookies to a file
    with open(COOKIE_FILE, 'wb') as file:
        pickle.dump(cookies, file)
        
    print(f"\n✅ Success! Cookies have been saved to '{COOKIE_FILE}'.")
    print("You can now close this script and the Chrome window.")

except Exception as e:
    print(f"\n❌ An error occurred while saving cookies: {e}")

finally:
    # Wait for a final confirmation before closing, so the user can see the message.
    input("\nPress Enter to close the browser.")
    driver.quit()