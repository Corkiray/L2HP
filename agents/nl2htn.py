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
from l2p import *

from up_siadex import SIADEXEngine

import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.model.htn.hierarchical_problem import HierarchicalProblem, Task, Method
from unified_planning.io import PDDLReader
from unified_planning.io import PDDLWriter
from unified_planning.io.hpdl.hpdl_reader import HPDLReader
from unified_planning.engines.results import PlanGenerationResultStatus

API_KEY = hf_token
REQUIEREMENTS = [":strips", ":typing"]
PROBLEM_PATH = "../unified-planning/unified_planning/test/pddl/hpdl/jacob/problem.hpdl"
DOMAIN_PATH = "../unified-planning/unified_planning/test/pddl/hpdl/jacob/domain.hpdl"
PLAN_PATH = "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/"


#       Auxiliar Functions
def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    with open(file_path, 'r') as file:
        if ext == '.json': return json.load(file)
        else: return file.read().strip()




#       Initializations and Loadings
planner = FastDownward(planner_path="/mnt/homeGPU/ipuerta/l2p-htn/downward/downward/fast-downward.py")  # FastDownward planner
builder = MainBuilder("bloques", "bloques_problem", isHTN=True)
builder.requirements = REQUIEREMENTS

# model = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324",
#     provider="nebius", api_key=hf_token, max_tokens=500)

# model = HUGGING_FACE(model_path="Qwen/Qwen2.5-7B-Instruct-1M")
model = MockLLM(
    [
        load_file(
            "/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/llm_output.hpdl.txt"
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

# for element in output_list[:-1]:
#     print(f"Element type: {type(element)}")
#     print(element)
# print(builder.get_domain())
# print(builder.get_problem())

# Save the domain and problem to files
# with open(DOMAIN_PATH, "w") as file:
#     file.write(builder.get_domain())
# with open(PROBLEM_PATH, "w") as file:
#     file.write(builder.get_problem())

reader = HPDLReader()
problem = reader.parse_problem(DOMAIN_PATH, PROBLEM_PATH)
print(problem)

with OneshotPlanner(name='siadex') as p:
    result = p.solve(problem)
    # print(result)
    if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
        print(f'{p.name()} found a valid plan!')
        print(f'The plan is: \n')
        print(f"_"*50)
        for i,a in enumerate(result.plan.timed_actions):
            action_name = a[1].action.name
            action_params = [p.name for p in a[1].action.parameters]
            if a[1].action.name in action_status:
                action_name = action_status[action_name]
            print(f"""Start: {a[0]} - End: {a[0] + a[2]} \nAction: {action_name}""")
            for param,value in list(zip(action_params,a[1].actual_parameters)):
                print(f"    - {param}: {value}")    
                
            print(f"_"*50)
    else:
        print('No plan found!')
# Run planner
# plan_name = "htnplan_" + builder.domain_name + "_" + builder.problem_name + ".htn"
# _, output = planner.run_fast_downward(domain_file=DOMAIN_PATH, problem_file=PROBLEM_PATH)

# Write generated plan into folder
with open(PLAN_PATH+plan_name, "w") as file:
    file.write(output)