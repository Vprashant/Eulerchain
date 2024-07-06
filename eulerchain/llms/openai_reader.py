import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from openai import OpenAI
from .base_llm_reader import BaseLLMReader

class OpenAIConfig(BaseModel):
    api_key: str
    model_id: str = "text-davinci-003"  
    default_prompt: Optional[str] = Field(
        None,
        description="Default prompt if no custom prompt is provided."
    )

class OpenAIResponse(BaseModel):
    text: str  

class OpenAIReader(BaseLLMReader):
    def __init__(self, api_key: str, 
                 model_id: str = "text-davinci-003", 
                 default_prompt: Optional[str] = None):
        try:
            self.config = OpenAIConfig(api_key=api_key, model_id=model_id, default_prompt=default_prompt)
            os.environ['OPENAI_API_KEY'] = self.config.api_key
            self.openai = OpenAI(api_key=self.config.api_key)
        except ValidationError as e:
            print(f"[EXC]: Configuration validation error. {e}")
        except Exception as e:
            print(f"[EXC]: Exception occurred while setting up the API. {e}")

    def read(self, text: str, custom_prompt: Optional[str] = None) -> Optional[OpenAIResponse]:
        prompt = custom_prompt if custom_prompt else self.config.default_prompt
        if not prompt: 
            prompt = text

        try:
            response = self.openai.Completion.create(
                model=self.config.model_id,
                prompt=prompt,
                max_tokens=150,  
                n=1,
                stop=None
            )
            return OpenAIResponse(text=response.choices[0].text.strip())
        except Exception as e:
            print(f"[EXC]: Exception occurred during API call. {e}")
            return None
