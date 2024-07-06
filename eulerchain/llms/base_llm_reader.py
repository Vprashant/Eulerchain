from abc import ABC, abstractmethod

class BaseLLMReader(ABC):

    @abstractmethod
    def read(self, input_text):
        """Process the input text and return the model's output."""
        pass
