import time
from datetime import datetime
from enum import StrEnum

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import Select, WebDriverWait
from camera import Camera


class ServiceRequestStatusCodes(StrEnum):
    OPEN = "614110001"
    IN_PROGRESS = "614110002"
    CANCEL = "614110000"
    CLOSED = "614110003"


class ServiceRequest:
    """
    Handles filing 311 service requests and checking the status of requests that have been filed.

    Attributes:
        status_key (str): API key for checking service request status.
        url (str): URL for the 311 service request portal.
        driver (webdriver.Chrome): Selenium WebDriver instance for web interactions.
        waiter (WebDriverWait): WebDriverWait instance for handling wait conditions.
    """

    def __init__(self, email: str, password: str, status_key: str):
        self.status_key = status_key
        self.url = "https://portal.311.nyc.gov/article/?kanumber=KA-01986"
        self.driver = webdriver.Chrome()
        self.waiter = WebDriverWait(self.driver, 10.0)
        self._login(email, password)

    def _login(self, email: str, password: str):
        """
        Logs into the 311 portal to avoid captcha on the submit page.

        Args:
            email (str): Email address for login.
            password (str): Password for login.
        """

        self.driver.get(self.url)
        signin_link = self.waiter.until(
            ec.visibility_of_element_located((By.XPATH, "//*[text()='Sign In']"))
        )
        signin_link.click()

        email_field = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "logonIdentifier"))
        )
        email_field.send_keys(email)
        self.driver.find_element(By.ID, "password").send_keys(password)
        self.driver.find_element(By.ID, "next").click()

    def submit_service_request(self, camera: Camera):
        """
        Submits a service request via browser for a given camera.

        Args:
            camera (Camera): The camera object for which to submit the service request.

        Returns:
            str: The service request number.
        """

        current_datetime = datetime.now()
        observed_datetime = current_datetime.strftime("%m/%d/%Y %I:%M %p")

        self.driver.get(self.url)

        accordion_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "accordion-card-title"))
        )
        accordion_button.click()
        self.driver.implicitly_wait(0.5)
        self.driver.find_element(
            By.XPATH, "//*[text()='Report illegal parking.']"
        ).click()

        # Begin 311 service request submission
        # Step 1
        problem_detail = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "n311_problemdetailid_select"))
        )
        select = Select(problem_detail)
        select.select_by_visible_text(camera.descriptor)

        datetime_field = self.waiter.until(
            ec.visibility_of_element_located(
                (By.ID, "n311_datetimeobserved_datepicker_description")
            )
        )
        datetime_field.send_keys(observed_datetime)

        description_field = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "n311_description"))
        )
        description_field.send_keys(camera.description_txt)

        next_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "NextButton"))
        )
        next_button.click()

        # Step 2
        select_address_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "SelectAddressWhere"))
        )
        select_address_button.click()
        time.sleep(2)

        addy_search_box = self.driver.find_element(By.ID, "address-search-box-input")
        addy_search_box.send_keys(camera.address)
        time.sleep(1)
        addy_search_box.send_keys(Keys.DOWN)
        addy_search_box.send_keys(Keys.ENTER)
        time.sleep(1)

        select_address_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "SelectAddressMap"))
        )
        select_address_button.click()

        time.sleep(1)
        self.driver.find_element(By.ID, "NextButton").click()

        # Step 3
        next_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "NextButton"))
        )
        next_button.click()

        # Step 4
        next_button = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "NextButton"))
        )
        next_button.click()
        sr_number = self.waiter.until(
            ec.visibility_of_element_located((By.ID, "n311_name"))
        )

        return sr_number.get_attribute("value")

    def check_status(self, sr_number: str):
        """
        Checks the status of a service request.

        Args:
            sr_number (str): The service request number to check.

        Returns:
            dict: The service request status information.
        """

        url = f"https://api.nyc.gov/public/api/GetServiceRequest?srnumber={sr_number}"
        headers = {
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.status_key,
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        sr_status = response.json()
        print(sr_status)

        if "Error" in sr_status:
            raise ValueError(f"SR check returned an error.")

        return sr_status
