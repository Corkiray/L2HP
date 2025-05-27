"""
This file is used to run experiments with different agents and datasets.
"""
import os, json
from l2p.dataset_builder import PlanBenchDataset
from l2p.planner_builder import UP_Planner
from l2p.model_builder import ModelBuilder
from tests.mock_llm import MockLLM
from agents.nl2htn import NL2HTNAgent
from mySecrets import hf_token

#       Constants and Configurations
REQUIEREMENTS = [":strips", ":typing", ":hierarchy", ":negative-preconditions", ":conditional-effects",]
RESULTS_PATH = "results/"
TEMPLATE_PATH = "templates/model_templates/extract_hddl_model.txt"


#       Initializations and Loadings
dataset = PlanBenchDataset()
planner = UP_Planner('aries')
builder = ModelBuilder("domain_name", "problem_name", isHTN=True, requirements=REQUIEREMENTS)

with open('/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/llm_output.txt', 'r') as file:
    mock_response = file.read()
llm = MockLLM([mock_response])

# llm = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324",
#     provider="nebius", api_key=hf_token, max_tokens=1024)
# llm = HUGGING_FACE(model_path="Qwen/Qwen2.5-7B-Instruct-1M")

with open(TEMPLATE_PATH, 'r') as file:
    extract_hddl_domain_and_problem_prompt =  file.read().strip()

agent = NL2HTNAgent(
    llm=llm,
    builder=builder,
    planner=planner,
    prompt_template=extract_hddl_domain_and_problem_prompt
)


for task in dataset.data_dict.values():
    print(f"Running task: {task['name']}")
    domain_path = RESULTS_PATH + task['name'] + "_domain.hddl"
    problem_path = RESULTS_PATH + task['name'] + "_problem.hddl"
    plan_path = RESULTS_PATH + task['name'] + "_plan.txt"

    # Extract domain and problem using the agent
    error_trace, execution_flag = agent.run(task['desc'], domain_path, problem_path, plan_path)
    
    if execution_flag != 0:
        print(f"Error in task {task['name']}: {error_trace}")
        continue
    print(f"Domain and problem extracted for task {task['name']}.")