from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

def web_scrap_page(url):
    driver = webdriver.Chrome("/home/unal/modelo_rinas/chromedriver")
    driver.get(url)
    driver.find_element_by_id('btnIngreso').click()
    return driver

STOPWORDS_ADDRESS = ['VIA','AVENIDA', 'VEREDA', 'AUTOPISTA', 'AV', 'KILOMETRO', 'KM', 'ESTE']

def web_scrap_address(driver,address):
    if (not any(word in address for word in STOPWORDS_ADDRESS)) and has_numbers(address):
        wait = WebDriverWait(driver, 15)
        search_field = wait.until(EC.presence_of_element_located((By.ID, 'panelSearch')))
        search_field.send_keys(address)
        try:
            wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT,'Dirección')))
            search_field.send_keys(Keys.ARROW_DOWN + Keys.RETURN)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME,'contentPane')))
            result = driver.find_element_by_class_name('contentPane').get_attribute('textContent')
            return result
        except TimeoutException:
            return "Not found"
    else:
        return "Not found"
    
def delete_address(driver,address):
    search_field = driver.find_element_by_id('panelSearch')
    for i in range(len(address)):
        search_field.send_keys(Keys.BACKSPACE)
        
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)