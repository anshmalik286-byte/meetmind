import os
from dotenv import load_dotenv
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

load_dotenv()

api_key = os.getenv("SivlVzhJtqf6p3C23e1IDP7dYHMSj8lzj6E79jpQ9Q8y")
url = os.getenv("https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/0a8e7584-dc2f-45b9-a5e5-081a92265f14")

authenticator = IAMAuthenticator(api_key)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)

print(stt.list_models().get_result())