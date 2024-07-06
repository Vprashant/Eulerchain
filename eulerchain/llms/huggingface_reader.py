import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from .base_llm_reader import BaseLLMReader

class HuggingFaceConfig(BaseModel):
    api_key: str
    model_id: str = "gpt2"  
    default_prompt: Optional[str] = Field(
        None,
        description="Default prompt if no custom prompt is provided."
    )

class HuggingFaceResponse(BaseModel):
    text: str  

class HuggingFaceReader(BaseLLMReader):
    def __init__(self, api_key: str, model_id: str = "gpt2", default_prompt: Optional[str] = None):
        try:
            self.config = HuggingFaceConfig(api_key=api_key, model_id=model_id, default_prompt=default_prompt)
            os.environ['HUGGING_FACE_API_KEY'] = self.config.api_key
            
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        except ValidationError as e:
            print(f"[EXC]: Configuration validation error. {e}")
        except Exception as e:
            print(f"[EXC]: Exception occurred while setting up the model. {e}")

    def read(self, text: str, 
             custom_prompt: Optional[str] = None) -> Optional[HuggingFaceResponse]:
        prompt = custom_prompt if custom_prompt else self.config.default_prompt
        if not prompt:  
            prompt = text

        try:
            results = self.generator(prompt, max_length=50, num_return_sequences=1)  
            return HuggingFaceResponse(text=results[0]['generated_text'])
        except Exception as e:
            print(f"[EXC]: Exception occurred during text generation. {e}")
            return None
