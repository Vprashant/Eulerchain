import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import google.generativeai as genai
from .base_llm_reader import BaseLLMReader

class APIConfig(BaseModel):
    api_key: str
    default_prompt: Optional[str] = Field(
        None,
        description="Default prompt if no custom prompt is provided."
    )

class GeminiProResponse(BaseModel):
    text: str

class GeminiProReader(BaseLLMReader):
    def __init__(self, api_key: str, default_prompt: Optional[str] = None):
        try:
            self.config = APIConfig(api_key=api_key, default_prompt=default_prompt)
            os.environ['GOOGLE_API_KEY'] = self.config.api_key
            genai.configure(api_key=self.config.api_key)
        except ValidationError as e:
            print(f"[EXC]: API key validation error. {e}")
        except Exception as e:
            print(f"[EXC]: Exception occurred while configuring the API key. {e}")

    def read(self, text: str, custom_prompt: Optional[str] = None) -> Optional[GeminiProResponse]:
        prompt = custom_prompt if custom_prompt else self.config.default_prompt
        if not prompt:
            print("----", text)
            prompt = (
                f"Extract relationships between entities in the following text:\n\n{text}\n\n"
                "Provide the relationships in the format: Source -> Target [Relationship]"
            )

        try:
            response = genai.GenerativeModel('models/gemini-pro').generate_content(prompt)
            print(f"Response: {response.text}")
            if response and response.text:
                return GeminiProResponse(text=response.text)
            else:
                print("[EXC]: Unexpected response structure")
                return None
        except Exception as e:
            print(f"[EXC]: Exception occurred during API call. {e}")
            return None
