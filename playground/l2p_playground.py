"""Inference client for L2P Playground."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from l2p.llm_builder import InferenceClient
from l2p.utils import load_file
from l2p.domain_builder import DomainBuilder
from mySecrets import hf_token
from mySecrets import working_dir

os.chdir(working_dir)

domain_builder = DomainBuilder()

API_KEY = hf_token

llm = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324",
    provider="nebius", api_key=hf_token, max_tokens=500)

# retrieve prompt information
BASE_PATH='tests/usage/prompts/domain/'
domain_desc = load_file(f'{BASE_PATH}blocksworld_domain.txt')
extract_predicates_prompt = load_file(f'{BASE_PATH}extract_predicates.txt')
types = load_file(f'{BASE_PATH}types.json')
action = load_file(f'{BASE_PATH}action.json')

# extract predicates via LLM
predicates, llm_output = domain_builder.extract_predicates(
    model=llm,
    domain_desc=domain_desc,
    prompt_template=extract_predicates_prompt,
    types=types,
    nl_actions={action['action_name']: action['action_desc']}
    )

# format key info into PDDL strings
PREDICATE_STR = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

print(f"PDDL domain predicates:\n{PREDICATE_STR}")
