from retry import retry
from typing_extensions import override
from .base import BaseLLM

class InferenceClient(BaseLLM):
    def __init__(
        self, 
        model: str, 
        provider: str = None,
        api_key: str | None = None,
        max_tokens: int = 16384,
        ) -> None:
        
        self.provider = provider

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "The 'huggingface_hub' library is required for but is not installed. "
            )

        # super().__init__(model, api_key)
        self.client = InferenceClient(provider=provider, api_key=api_key)
        
        self.max_tokens = max_tokens
        self.in_tokens = 0
        self.out_tokens = 0
        self.model = model

    @override
    def query(self, prompt: str) -> str | None:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content

    @override
    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {   "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content

    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens

    def reset_tokens(self):
        self.in_tokens = 0
        self.out_tokens = 0