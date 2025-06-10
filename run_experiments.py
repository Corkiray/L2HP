"""
This file is used to run experiments with different agents and datasets.
"""
import os, json
from l2p.dataset_builder import PlanBenchDataset
from l2p.planner_builder import UP_Planner
from l2p.model_builder import ModelBuilder
from l2p.llm_builder import GeminiClient
from agents.nl2htn import NL2HTNAgent
from mySecrets import GeminiApi3_token as token
from unified_planning.environment import get_environment


# ====== Configurations, Constants, Initializations and Loadings ======
mode = "hddl"  # or "hddl"

get_environment().error_used_name = False  # Allow same names for different elements

if mode == "pddl":
    REQUIEREMENTS = [":strips", ":typing", ":negative-preconditions", ":conditional-effects",]
    RESULTS_PATH = "results/pddl/"
    TEMPLATE_PATH = "templates/model_templates/extract_pddl_model.txt"
    planner = UP_Planner('fast-downward')
    isHTN = False

elif mode == "hddl":
    REQUIEREMENTS = [":strips", ":typing", ":hierarchy", ":negative-preconditions", ":conditional-effects",]
    RESULTS_PATH = "results/hddl/"
    TEMPLATE_PATH = "templates/model_templates/extract_hddl_model.txt"
    planner = UP_Planner('aries')
    isHTN = True


llm = GeminiClient('gemini-2.0-flash', api_key=token)
dataset = PlanBenchDataset()
builder = ModelBuilder("domain_placeholder", "problem_placeholder", isHTN=isHTN, requirements=REQUIEREMENTS)
# with open('/mnt/homeGPU/ipuerta/l2p-htn/tests/usage/prompts/main_builder/llm_output.txt', 'r') as file:
#     mock_response = file.read()
# llm = MockLLM([mock_response])

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


# ====== Running the tasks ======

results_summary = {
    "total_tasks": len(dataset.data_dict),
    "successful_tasks": 0,
    "failed_tasks": 0,
    "-1": 0,  # Error flag
    "0": 0,   # Success flag
    "1": 0,   # Error in extraction
    "2": 0,   # Error in processing
    "3": 0,   # Error in planing
    "4": 0,   # Plan unsolvable
}

for task in dataset.data_dict.values():
    task_directory = RESULTS_PATH + task['name'] + "/"
    
    # Create task directory if it doesn't exist
    if not os.path.exists(task_directory):
        os.makedirs(task_directory)
    else: # Skip if the directory already exists
        print(f"Directory {task_directory} already exists. Skipping task {task['name']}.")
        continue
    
    print(f"Running task: {task['name']}")
    domain_path = task_directory + task['name'] + ".domain.hddl"
    problem_path = task_directory + task['name'] + ".problem.hddl"
    plan_path = task_directory + task['name'] + ".plan.txt"
    response_path = task_directory + task['name'] + ".llm_response.txt"
    log_path = task_directory + task['name'] + ".log.txt"

    # Extract domain and problem using the agent
    error_trace, execution_flag = agent.run(task['desc'], domain_path, problem_path, plan_path, response_path)
    
    if execution_flag != 0:
        results_summary["failed_tasks"] += 1
        results_summary[str(execution_flag)] += 1
        with open(log_path, "w") as log_file:
            log_file.write(f"Execution Flag: {execution_flag}\nError in task execution: {error_trace}")
        print(f"Error in task {task['name']}")
    else:
        results_summary["successful_tasks"] += 1
        with open(log_path, "w") as log_file:
            log_file.write(f"Execution Flag: {execution_flag}\nTask executed successfully.")
        print(f"Execution successful for task {task['name']}")
        
    with open(RESULTS_PATH + "results_summary.json", "w") as summary_file:
        json.dump(results_summary, summary_file, indent=4)
    print(f"Results summary: {results_summary}")


#  ===== Final Comparation ======

DOMAINS = ["blocksworld", "depots", "logistics", "mystery_blocksworld", "obfuscated_deceptive_logistics"]

results_by_domain = {domain: {  "N": 0,
                                "parsing_error": 0,
                                "syntax_error": 0, 
                                "void_plan": 0, 
                                "found_plan": 0,
                                "incorrect_plan": 0, 
                                "correct_plan": 0}    for domain in DOMAINS}
for directory in os.listdir(RESULTS_PATH):
    if not os.path.isdir(RESULTS_PATH + directory):
        continue
    
    for domain in DOMAINS:
        if directory.startswith(domain):
            results_dict = results_by_domain[domain]
            break
    
    with open(RESULTS_PATH + directory + f'/{directory}.log.txt', "r") as log_file:
        results_dict["N"] += 1
        
        log_content = log_file.read()
        if "Execution Flag: 1" in log_content:
            results_dict["parsing_error"] += 1
        elif "Execution Flag: 3" in log_content:
            results_dict["syntax_error"] += 1
        elif "Execution Flag: 4" in log_content:
            results_dict["void_plan"] += 1
        elif "Execution Flag: 0" in log_content:
            results_dict["found_plan"] += 1
            with open(RESULTS_PATH + directory + f"/{directory}.plan.txt", "r") as plan_file:
                plan = plan_file.read().strip()
            print(f"Comparing ground truth and generated plan for {directory}")
            print(f"\tGround Truth {dataset.data_dict[directory]['ground_truth']}")
            print(f"\tGenerated Plan {plan}")
        else:
            print(f"Unknown execution flag in {directory}, Review the log file.")

print("Results by domain:")
for domain, results in results_by_domain.items():
    print(f"{domain}: {results}")