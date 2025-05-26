import os, json, sys
WORKING_DIR = "/mnt/homeGPU/ipuerta/l2p-htn"

os.chdir(WORKING_DIR)
sys.path.append(os.getcwd())
print(os.getcwd())
from mySecrets import hf_token
from l2p.llm_builder import InferenceClient, HUGGING_FACE
from l2p.main_builder import MainBuilder
from l2p.utils.pddl_parser import prune_predicates, format_types
from l2p.utils.planners_interface import UP_Planner
from tests.mock_llm import MockLLM

# from up_siadex import SIADEXEngine
# env = up.environment.get_environment()
# env.factory.add_engine('siadex', __name__, "SIADEXEngine")

API_KEY = hf_token
REQUIEREMENTS = [":strips", ":typing", ":hierarchy", ":negative-preconditions", ":conditional-effects",]
DOMAIN_PATH = "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/domain.hddl"
PROBLEM_PATH = "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/problem.hddl"
PLAN_PATH = "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/plan.txt"


#       Auxiliar Functions
def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    with open(file_path, 'r') as file:
        if ext == '.json': return json.load(file)
        else: return file.read().strip()




#       Initializations and Loadings
planner = UP_Planner('aries')  # FastDownward planner
builder = MainBuilder("bloques", "bloques_problem", isHTN=True, requirements=REQUIEREMENTS)

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
extract_hddl_domain_and_problem_prompt = load_file(r'templates/model_templates/extract_hddl_model.txt')


#       Execution

# extract predicates via LLM
output_list = builder.extract_domain_and_problem(
    model=model,
    task_desc=domain_desc,
    prompt_template=extract_hddl_domain_and_problem_prompt,
    )

# Save the domain and problem to files
with open(DOMAIN_PATH, "w") as file:
    file.write(builder.get_domain('HDDL'))
with open(PROBLEM_PATH, "w") as file:
    file.write(builder.get_problem())


# Run planner
plan = planner.solve(DOMAIN_PATH, PROBLEM_PATH)

# Write generated plan into folder
with open(PLAN_PATH, "w") as file:
    file.write(plan)