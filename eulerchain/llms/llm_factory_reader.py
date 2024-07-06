from typing import Type, Optional, Dict
from .huggingface_reader import HuggingFaceReader
from .google_gemini_reader import GeminiProReader
from .openai_reader import OpenAIReader
from .base_llm_reader import BaseLLMReader

class LLMReaderFactory:
    reader_map: Dict[str, Type[BaseLLMReader]] = {
        'hugging_face': HuggingFaceReader,
        'gemini_pro': GeminiProReader,
        'openai': OpenAIReader
    }

    @staticmethod
    def get_llm_reader(reader_type: str, **config) -> BaseLLMReader:
        reader_class = LLMReaderFactory.reader_map.get(reader_type)
        if not reader_class:
            raise ValueError(f"Unknown LLM reader type: {reader_type}")
        try:
            return reader_class(**config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM reader '{reader_type}': {str(e)}")

if __name__ == "__main__":
   
    openai_reader = LLMReaderFactory.get_llm_reader(
        'openai',
        api_key='your_openai_api_key_here',
        model_id='text-davinci-003'
    )


    result = openai_reader.read("Example prompt for the OpenAI model.")
    print(result.text if result else "No response received.")
