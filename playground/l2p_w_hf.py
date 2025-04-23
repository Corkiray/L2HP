"""Inference client for L2P Playground."""
import sys
import os
sys.path.append(os.getcwd())
from l2p.utils import load_file
from l2p.domain_builder import DomainBuilder
from mySecrets import hf_token
from mySecrets import working_dir
from l2p.llm_builder import HUGGING_FACE


os.chdir(working_dir)

domain_builder = DomainBuilder()

API_KEY = hf_token


llm = HUGGING_FACE(model_path="tiiuae/Falcon3-7B-Instruct")

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
