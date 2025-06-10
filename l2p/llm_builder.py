"""
This file contains code for calling LLMs and saving raw model outputs 
Currently, this builder class contains the generic method to run any LLMs. 
It also offers extension to OpenAI and Huggingface, but can generalise to any third-party LLM store.
"""

import logging, functools
from retry import retry
from abc import ABC, abstractmethod
from typing_extensions import override

LOG: logging.Logger = logging.getLogger(__name__)

def require_llm(func):
    """
    Decorator to check if an LLM instance is provided and catch errors.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # check if LLM instance is provided
        model = kwargs.get("model", None)
        if model is None:
            # attempt tp get model from positional argument
            for arg in args:
                if isinstance(arg, LLM):
                    model = arg
                    break

        if not model:
            raise ValueError("An LLM instance must be provided to use this method.")

        # try-catch to ensure LLM is even used properly
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                f"An error occurred in {func.__name__}: {e}.\n You must provide an LLM engine OR proper configuration to use L2P. Refer to https://github.com/AI-Planning/l2p."
            )
            raise

    return wrapper


class LLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in self.valid_models():
            LOG.warning(
                f"{model} is not in the valid model list for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(self, prompt: str) -> str | None:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM
        Returns:
            str: The response from the LLM
        """
        pass
    
    @abstractmethod
    def reset_tokens(self):
        """
        Reset the token counters for the LLM.
        """
        pass

    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract methody to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM
            prompt (str): The prompt to send to the LLM
        Returns:
            str: The response from the LLM
        """
        return self.query(system_prompt + "\n" + prompt)

    def valid_models(self) -> list[str]:
        """
        List of valid model parameters, e.g., 'gpt4o-mini' for GPT
        """
        return []


class OPENAI(LLM):
    """Accessing OpenAI"""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client=None,
        stop=None,
        max_tokens=4e3,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=0,
    ) -> None:

        # attempt to import necessary OPENAI modules
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' library is required for OPENAI but is not installed. "
                "Install it using: `pip install openai`."
            )

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The 'tiktoken' library is required for token processing but is not installed. "
                "Install it using: `pip install tiktoken`."
            )

        # call the parent class constructor to handle model and api_key
        super().__init__(model, api_key)

        # initialize the OpenAI client or use the one provided
        self.client = client if client else OpenAI(api_key=api_key)

        # store other parameters
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        self.context_length = {
            "gpt-3.5-turbo-0125": 16e3,  # 16k tokens
            "gpt-3.5-turbo-instruct": 4e3,  # 4k tokens
            "gpt-4-1106-preview": 128e3,  # 128k tokens
            "gpt-4-turbo-2024-04-09": 128e3,  # 128k tokens
            "gpt-4": 8192,  # ~8k tokens
            "gpt-4-32k": 32768,  # ~32k tokens
            "gpt-4o": 32768,  # ~32k tokens
            "gpt-4o-mini": 32768,  # ~32k tokens
        }[model]

        self.max_tokens = max_tokens if max_tokens is not None else self.context_length
        self.tok = tiktoken.get_encoding("cl100k_base")  # For GPT3.5+
        self.in_tokens = 0
        self.out_tokens = 0

    @retry(tries=2, delay=60)
    def connect_openai(
        self,
        client,
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
    ):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

    @override
    def query(
        self,
        prompt: str,
        messages=None,
        end_when_error=False,
        max_retry=5,
        est_margin=200,
    ) -> str:
        if prompt is None and messages is None:
            raise ValueError("Prompt and messages cannot both be None")
        if messages is not None:
            messages = messages
        else:
            messages = [{"role": "user", "content": prompt}]

        # calculate # of tokens to request. At most self.max_tokens, and prompt + request < self.context_length
        current_tokens = int(
            sum([len(self.tok.encode(m["content"])) for m in messages])
        )  # estimate current usage
        requested_tokens = int(
            min(self.max_tokens, self.context_length - current_tokens - est_margin)
        )  # request with safety margin
        print(
            f"Requesting {requested_tokens} tokens from {self.model} (estimated {current_tokens - est_margin} prompt tokens with a safety margin of {est_margin} tokens)"
        )
        self.in_tokens += current_tokens

        # request response
        n_retry = 0
        conn_success = False
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                print(f"[INFO] connecting to the LLM ({requested_tokens} tokens)...")
                response = self.connect_openai(
                    client=self.client,
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=requested_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                )
                llm_output = response.choices[
                    0
                ].message.content  # response['choices'][0]['message']['content']
                conn_success = True
            except Exception as e:
                print(f"[ERROR] LLM error: {e}")
                if end_when_error:
                    break
        if not conn_success:
            raise ConnectionError(
                f"Failed to connect to the LLM after {max_retry} retries"
            )

        response_tokens = len(self.tok.encode(llm_output))  # Estimate response tokens
        self.out_tokens += response_tokens

        return llm_output

    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens

    def reset_tokens(self):
        self.in_tokens = 0
        self.out_tokens = 0

    @override
    def valid_models(self) -> list[str]:
        """
        List of valid model parameters for OpenAI.
        Returns a list of valid OpenAI model names.
        """
        return [
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-instruct",
            "gpt-4-1106-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4o",
            "gpt-4o-mini",
        ]


class HUGGING_FACE(LLM):
    def __init__(self, model_path: str, max_tokens=4e3, temperature=0.01, top_p=0.9):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.in_tokens = 0
        self.out_tokens = 0
        
        self._load_transformers()

    def _load_transformers(self):

        # attempt to import the `transformers` library
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required for HUGGING_FACE but is not installed. "
                "Install it using: `pip install transformers`."
            )

        # attempt to import `AutoTokenizer` from `transformers`
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers.AutoTokenizer' module is required but is not installed properly. "
                "Ensure that the 'transformers' library is installed correctly."
            )

        # attempt to import the `torch` library
        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required for HUGGING_FACE but is not installed. "
                "Install it using: `pip install torch`."
            )

        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            # Check if the model_path is valid by trying to load it
            self.model = transformers.pipeline(
                "text-generation",
                model=self.model_path,
                model_kwargs={"torch_dtype": dtype},
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except OSError as e:
            # if model_path is not found, raise an error
            raise ValueError(
                f"Model path '{self.model_path}' could not be found. Please ensure the model exists."
            )
        except Exception as e:
            # catch any other exceptions and raise a generic error
            raise RuntimeError(f"An error occurred while loading the model: {str(e)}")

    # Retry decorator to handle retries on request
    @retry(tries=2, delay=60)
    def connect_huggingface(self, input, temperature, max_tokens, top_p, numSample):
        if self.model is None or self.tokenizer is None:
            self._load_transformers()

        if numSample > 1:
            responses = []
            sequences = self.model(
                input,
                do_sample=True,
                top_k=1,
                num_return_sequences=numSample,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            for seq in sequences:
                response = seq["generated_text"]
                responses.append(response)
            return responses

        else:
            sequences = self.model(
                input,
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            seq = sequences[0]
            response = seq["generated_text"]

            return response

    @override
    def query(self, prompt: str, numSample=1, max_retry=3, est_margin=200) -> str:
        if prompt is None:
            raise ValueError("Prompt cannot be None")

        # Estimate current usage of tokens
        current_tokens = len(self.tokenizer.encode(prompt))
        requested_tokens = min(
            self.max_tokens, self.max_tokens - current_tokens - est_margin
        )

        print(
            f"Requesting {requested_tokens} tokens from {self.model} (estimated {current_tokens - est_margin} prompt tokens with a safety margin of {est_margin} tokens)"
        )

        # Retry logic for Hugging Face request
        n_retry = 0
        conn_success = False
        while not conn_success and n_retry < max_retry:
            n_retry += 1
            try:
                print(
                    f"[INFO] Connecting to Hugging Face model ({requested_tokens} tokens)..."
                )
                llm_output = self.connect_huggingface(
                    input=prompt,
                    temperature=self.temperature,
                    max_tokens=requested_tokens,
                    top_p=self.top_p,
                    numSample=numSample,
                )
                conn_success = True
            except Exception as e:
                print(f"[ERROR] Hugging Face error: {e}")
                if n_retry >= max_retry:
                    raise ConnectionError(
                        f"Failed to connect to the Hugging Face model after {max_retry} retries"
                    )

        # Token management
        response_tokens = len(self.tokenizer.encode(llm_output))
        self.out_tokens += response_tokens
        self.in_tokens += current_tokens

        return llm_output

    def get_tokens(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens

    def reset_tokens(self):
        self.in_tokens = 0
        self.out_tokens = 0


class InferenceClient(LLM):
    def __init__(self, provider: str, model: str, api_key: str, max_tokens: int) -> None:
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "The 'huggingface_hub' library is required for but is not installed. "
            )

        self.client = InferenceClient(provider=provider, api_key=api_key)
        self.in_tokens = 0
        self.out_tokens = 0

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


class GeminiClient(LLM):
    def __init__(self, model_name: str, api_key: str, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens
        self.in_tokens = 0
        self.out_tokens = 0
        self.model = model_name

        try:
            from google import genai
            from google.api_core import retry
        except ImportError:
            raise ImportError(
                "The 'google.genai' library is required for Gemini but is not installed. "
                "Install it using: `pip install google-generativeai`."
            )

        self.client = genai.Client(api_key=api_key)
        is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
            genai.models.Models.generate_content = retry.Retry(
            predicate=is_retriable)(genai.models.Models.generate_content)
        ### Automated retry
        # This codelab sends a lot of requests, so set up an automatic retry
        # that ensures your requests are retried when per-minute quota is reached.
        

    def query(self, prompt: str) -> str | None:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
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
