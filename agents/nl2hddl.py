import os, json, sys
WORKING_DIR = "/mnt/homeGPU/ipuerta/L2P_HTN"

os.chdir(WORKING_DIR)
sys.path.append(os.getcwd())
print(os.getcwd())
from mySecrets import hf_token
from l2p.llm_builder import InferenceClient
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_parser import prune_predicates, format_types

API_KEY = hf_token

def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    with open(file_path, 'r') as file:
        if ext == '.json': return json.load(file)
        else: return file.read().strip()

domain_builder = DomainBuilder()

model = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324",
    provider="nebius", api_key=hf_token, max_tokens=500)

# load in assumptions
domain_desc = load_file(r'tests/usage/prompts/domain/blocksworld_domain.txt')
extract_predicates_prompt = load_file(r'tests/usage/prompts/domain/extract_predicates.txt')
types = load_file(r'tests/usage/prompts/domain/types.json')
action = load_file(r'tests/usage/prompts/domain/action.json')

# extract predicates via LLM
predicates, llm_output = domain_builder.extract_predicates(
    model=model,
    domain_desc=domain_desc,
    prompt_template=extract_predicates_prompt,
    types=types,
    nl_actions={action['action_name']: action['action_desc']}
    )

# format key info into PDDL strings
predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

print(f"PDDL domain predicates:\n{predicate_str}")