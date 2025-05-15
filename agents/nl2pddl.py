import os, json, sys
WORKING_DIR = "/mnt/homeGPU/ipuerta/l2p-htn"

os.chdir(WORKING_DIR)
sys.path.append(os.getcwd())
print(os.getcwd())
from mySecrets import hf_token
from l2p.llm_builder import InferenceClient, HUGGING_FACE
from l2p.main_builder import MainBuilder
from l2p.utils.pddl_parser import prune_predicates, format_types
from tests.mock_llm import MockLLM

API_KEY = hf_token

REQUIEREMENTS = [":strips", ":typing"]

def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    with open(file_path, 'r') as file:
        if ext == '.json': return json.load(file)
        else: return file.read().strip()

builder = MainBuilder("bloques", "bloques_problem")
builder.requirements = REQUIEREMENTS

# model = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324",
#     provider="nebius", api_key=hf_token, max_tokens=500)

# model = HUGGING_FACE(model_path="Qwen/Qwen2.5-7B-Instruct-1M")
model = MockLLM(
    [
        load_file(
            "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/llm_output.txt"
        )
    ]
)

# load in assumptions
domain_desc = load_file(r'tests/usage/prompts/domain/blocksworld_domain.txt')
extract_pddl_domain_and_problem_prompt = load_file(r'tests/usage/prompts/main_builder/extract_pddl_domain_and_problem.txt')

# extract predicates via LLM
output_list = builder.extract_domain_and_problem(
    model=model,
    task_desc=domain_desc,
    prompt_template=extract_pddl_domain_and_problem_prompt,
    )

# for element in output_list[:-1]:
#     print(f"Element type: {type(element)}")
#     print(element)
    
print(builder.get_domain())
print(builder.get_problem())
