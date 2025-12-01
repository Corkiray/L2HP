import os
from l2p.llm.base import BaseLLM


class HumanLLM(BaseLLM):
    def __init__(self, prompt_path: str = None, response_path: str = None, thinking_time_path: str = None) -> None:
        """
        Initialize with a list of responses to simulate the LLM's outputs.
        """
        self.prompt_path = prompt_path
        self.response_path = response_path
        self.thinking_time_path = thinking_time_path
        self.output = ""

    def query(self, prompt: str):
        """
        Instead of querying an actual LLM, write the prompt to a file and wait for user input.
        The user is expected to write the response in another file.
        """
        print("Writing prompt in the following path:", self.prompt_path)
        print("Please check the prompt and provide the response in the following path:", self.response_path)
        with open(self.prompt_path, 'w') as file:
            file.write(prompt)
        # If response file does not exist, prompt the user to provide it
        if not os.path.exists(self.response_path):
            with open(self.response_path, 'w') as file:
                file.write("")  # Create or clear the response file
                thinking_time = input("Please, indicate the thinking time used (in seconds) and press Enter after providing the response to continue: ")
            with open(self.thinking_time_path, 'w') as file:
                file.write(thinking_time)

        with open(self.response_path, 'r') as file:
            self.output = file.read()
        return self.output

    def reset_tokens(self):
        """
        Placeholder for resetting tokens; not needed for testing.
        """
        pass