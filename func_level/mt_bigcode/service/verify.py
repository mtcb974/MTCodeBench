import requests
from typing import Dict, Any, Optional

class BigCodeEvalClient:
    # Singleton mode
    _instance = None

    def __new__(cls, base_url: str = "http://localhost:8000"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.base_url = base_url.rstrip('/')
        return cls._instance
    
    def verify_code(
        self,
        code: str,
        test: str,
    ) -> Dict[str, Any]:
        """
        Verify if the code is correct
        
        :param code: The code to test
        :param test: The test code
        :return: A dictionary containing the execution result, including status and details fields
        :raises: If the request fails or the server returns an error, an exception will be raised
        """
        url = f"{self.base_url}/verify/"
        payload = {
            "code": code,
            "test": test,
            "entry_point": "task_func"
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # If the response status code is not 200, throw an HTTPError
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def check_service_health(self) -> bool:
        """
        Check if the service is running normally
        
        :return: If the service is normal, return True, otherwise return False
        """
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

def verify_llm_testgen(code:str,test:str):
    client = BigCodeEvalClient()
    
    # Check if the service is running normally
    if not client.check_service_health():
        print("The service is not available, please check if the server is running")
        raise ValueError("The service is not available, please check if the server is running")

    try:
        # Verify the code
        result = client.verify_code(code, test)
        print("Verification result:", result)

        return result.get("status"),result.get("details")
    except Exception as e:
        print("Error during verification:", str(e))