
import os
from dotenv import load_dotenv

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

def add(x, y):
    return x + y
def test_add_positive_numbers():
    assert add(2, 3) == 5
