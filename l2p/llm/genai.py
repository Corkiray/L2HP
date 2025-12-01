from retry import retry
from typing_extensions import override
from .base import BaseLLM

class GenAIClient(BaseLLM):
    def __init__(
        self,
        model: str, 
        provider: str = "google.genai",
        api_key: str | None = None,
        max_tokens: int = 4096,
        ) -> None:
    
        self.provider = provider

        # attempt to import necessary google modules
        try:
            from google import genai
            from google.api_core import retry
        except ImportError:
            raise ImportError(
                "The 'google.genai' library is required for Gemini but is not installed. "
                "Install it using: `pip install google-generativeai`."
            )


        # super().__init__(model, api_key)
        self.client = genai.Client(api_key=api_key)

        self.max_tokens = max_tokens
        self.in_tokens = 0
        self.out_tokens = 0
        self.model = model

        ### Automated retry
        # This codelab sends a lot of requests, so set up an automatic retry
        # that ensures your requests are retried when per-minute quota is reached.
        is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
            genai.models.Models.generate_content = retry.Retry(
            predicate=is_retriable)(genai.models.Models.generate_content)
        
    @override
    def query(
        self, 
        prompt: str
    ) -> str | None:
        """Generate a response from the Google GenAI model based on the prompt."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        
        return response.text

    
    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[system_prompt, prompt]
        )
        return response.text
    
    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens
    
    def reset_tokens(self):
        self.in_tokens = 0
        self.out_tokens = 0
        
    @override
    def valid_models(self) -> list[str]:
        """Returns a list of valid model engines."""
        try:
            return ['gemini-2.0-flash']
        except KeyError:
            return []
