"""
This file is used to run experiments with different agents and datasets.
"""
import os, json, sys, traceback
import instructor
from unified_planning.environment import get_environment
from google import genai
from google.api_core import retry
from instructor import from_genai
from google import genai
from instructor import from_genai
import outlines

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from l2hp.agents.UltimateModeler import UltimatePDDLModeler
from l2p.dataset_builder import PlanBenchDataset
from l2p.planner_builder import UP_Planner
from mySecrets import GeminiApi_token1 as token
from l2p.llm.genai import GenAIClient as LLM

# ====== Configurations, Constants, Initializations and Loadings ======

# ---- Constants ----

LANGUAGE_TO_MODEL = "hddl"  # "hddl" or "pddl"
PARSER_API = "outlines"  # "outlines" or "instructor"
THINKING_MODE = False  # If True, use the think_about_task function to generate a thinking process before modeling
MODEL =  "gemini-robotics-er-1.5-preview" # LLM model to use. // "gemini-2.5-flash-lite" or "gemini-robotics-er-1.5-preview"
RESULTS_PATH = 'results_ultimateModeler_outlines'

DOMAINS = ["blocksworld", "depots", "logistics", "mystery_blocksworld", "obfuscated_deceptive_logistics"]

if LANGUAGE_TO_MODEL == "pddl":
    REQUIREMENTS = [":strips", ":typing", ":negative-preconditions", ":conditional-effects",]
    RESULTS_PATH = RESULTS_PATH+"/pddl/"
    TEMPLATE_PATH = "templates/l2hp_templates/think_about_pddl.txt"
    planner = UP_Planner('fast-downward')
    isHTN = False
elif LANGUAGE_TO_MODEL == "hddl":
    REQUIREMENTS = [":strips", ":typing", ":hierarchy", ":negative-preconditions", ":conditional-effects",]
    RESULTS_PATH = RESULTS_PATH+"/hddl/"
    TEMPLATE_PATH = "templates/l2hp_templates/think_about_hddl.txt"
    planner = UP_Planner('aries')
    isHTN = True
else:
    print("Invalid mode selected. Choose either 'pddl' or 'hddl'.")
    sys.exit(1)


# ---- Configurations ----

get_environment().error_used_name = False  # Allow same names for different elements
get_environment().credits_stream = None  # Disable credits logging



# ---- LLM and Parser Client ----
### Automated retry
# This codelab sends a lot of requests, so set up an automatic retry
# that ensures your requests are retried when per-minute quota is reached.
# is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
# client.create = retry.Retry(predicate=is_retriable)(client.create)

# LLM Reasoner Client
llm = LLM(MODEL, api_key=token)

# LLM Parser Client
if PARSER_API == "outlines":
    class Outlines_Client():
        def __init__(self, model_name: str):
            self.model = outlines.from_gemini(genai.Client(api_key=token), model_name)
            
        def create(self, messages, response_model = None):
            if response_model is None:
                return self.model(messages)
            output = self.model(messages, response_model)
            return response_model.model_validate_json(output)
    parser_client = Outlines_Client(MODEL)
elif PARSER_API == "instructor":
    parser_client = instructor.from_provider(f"google/{MODEL}", api_key=token) # , mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
else:
    print("Invalid parser API selected. Choose either 'outlines' or 'instructor'.")
    sys.exit(1)
    
# ---- Dataset Loading ----
dataset = PlanBenchDataset()


# ===== Solver Functions =====

def think_about_task(task_desc: str, response_path = None) -> str:
    with open(TEMPLATE_PATH, 'r') as file:
        prompt_template =  file.read().strip()
    
    prompt = prompt_template.replace("{task_desc}", task_desc)
    response = llm.query(prompt)

    if response_path is not None:
        with open(response_path, "w") as file:
                file.write(response) # type: ignore
    return response

modeler = UltimatePDDLModeler(parser_client, requirements=REQUIREMENTS)
def solve_task(task, domain_path, problem_path, plan_path, thinking_process=""):
    """
    Extracts the domain and problem from the task description using the UltimatePDDLModeler agent,
    then solves the planning problem using the specified planner.
    :param task: The task dictionary containing 'name' and 'desc'.
    :param client: The LLM client to use for the agent.
    :param domain_path: The file path to save the extracted domain.
    :param problem_path: The file path to save the extracted problem.
    :param plan_path: The file path to save the generated plan.
    :return: Tuple of (status_string, exit_code)
    """       
    
    task_desc = task['desc']
    if thinking_process != "":
        task_desc = f"{task_desc}\n\nThinking process:\n{thinking_process}"
    
    status_msg, exec_flag = modeler.generate_model(task_desc)
    if exec_flag != 0:
        return status_msg, exec_flag
       
    # Get domain and problem strings
    domain_str = modeler.generate_domain(f"{task['name']}_domain")
    problem_str = modeler.generate_problem(f"{task['name']}_problem", f"{task['name']}_domain")
        
    # Save the domain and problem to files
    with open(domain_path, "w") as file:
        file.write(domain_str)
    with open(problem_path, "w") as file:
        file.write(problem_str)
    
    # Run planner
    try:
        plan = planner.solve(domain_path, problem_path)
    except Exception as e:
        return f"Error running planner: {e}\n" + traceback.format_exc(), 3
        
    # Write generated plan into folder
    if plan is not None and plan.strip() != "":
        with open(plan_path, "w") as file:
            file.write(plan)
    else:
        return "Error, no plan found.\n" + traceback.format_exc(), 4
    
    return "Plan generated successfully", 0


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

tasks = dataset.data_dict.values()
tasks = [dataset.data_dict[domain+str(instance)] for instance in range(10,101,10) for domain in DOMAINS]


for task in tasks:
    task_directory = RESULTS_PATH + task['name'] + "/"
    
    if os.path.exists(task_directory): # Skip if the directory already exists
        print(f"Directory {task_directory} already exists. Skipping task {task['name']}.")
        continue
    
    # Create task directory if it doesn't exist
    print(f"Running task: {task['name']}")
    os.makedirs(task_directory)
    domain_path = task_directory + task['name'] + ".domain.hddl"
    problem_path = task_directory + task['name'] + ".problem.hddl"
    plan_path = task_directory + task['name'] + ".plan.txt"
    model_path = task_directory + task['name'] + ".model.json"
    log_path = task_directory + task['name'] + ".log.txt"
    response_path = task_directory + task['name'] + ".llm_response.txt"
    prompt_path = task_directory + task['name'] + ".prompt.txt"

    # Extract domain and problem using the agent
    
    response = ""
    if THINKING_MODE:
        response = think_about_task(task['desc'], response_path)
    modeler = UltimatePDDLModeler(parser_client, requirements=REQUIREMENTS)
    error_trace, execution_flag = solve_task(task, domain_path, problem_path, plan_path, thinking_process=response)
    # input('Press Enter to continue...')
    
    # Save the model to a JSON file
    with open(model_path, "w") as model_file:
        json.dump(modeler.to_dict(), model_file, indent=4)
    
    # Generate log file if there was an error and update results summary
    if execution_flag != 0:
        results_summary["failed_tasks"] += 1
        results_summary[str(execution_flag)] += 1
        with open(log_path, "w") as log_file:
            log_file.write(f"Execution Flag: {execution_flag}\nError in task execution: {error_trace}")
        print(f"Error in task {task['name']}")
    
    # Generate log file for successful execution and update results summary
    else:
        results_summary["successful_tasks"] += 1
        with open(plan_path, "r") as plan_file:
            generated_plan = plan_file.read().strip()
        with open(log_path, "w") as log_file:
            log_file.write(f"Execution Flag: {execution_flag}\nTask executed successfully.")
            log_file.write(f"\nGenerated Plan:\n{generated_plan}\n")
            log_file.write(f"\nGround Truth:\n{task['ground_truth']}\n")
        print(f"Execution successful for task {task['name']}")
    
    # Save intermediate results summaryz
    with open(RESULTS_PATH + "results_summary.json", "w") as summary_file:
        json.dump(results_summary, summary_file, indent=4)
    print(f"Results summary: {results_summary}")


#  ===== Final Comparison ======

results_by_domain = {domain: {
    "total": 0,
    "successful": 0,
    "parsing_error": 0,
    "planner_error": 0,
    "no_plan_found": 0,
    "success_rate": 0.0
} for domain in DOMAINS}

print("\n" + "="*80)
print("FINAL COMPARISON - RESULTS BY DOMAIN")
print("="*80 + "\n")

for directory in os.listdir(RESULTS_PATH):
    if directory == "results_summary.json" or not os.path.isdir(RESULTS_PATH + directory):
        continue
    
    # Find which domain this task belongs to
    domain_found = None
    for domain in DOMAINS:
        if directory.startswith(domain):
            domain_found = domain
            break
    
    if domain_found is None:
        continue
    
    results_dict = results_by_domain[domain_found]
    log_path = RESULTS_PATH + directory + f'/{directory}.log.txt'
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found for {directory}")
        continue
    
    results_dict["total"] += 1
    
    with open(log_path, "r") as log_file:
        log_content = log_file.read()
        
        if "Execution Flag: 0" in log_content:
            results_dict["successful"] += 1
            plan_path = RESULTS_PATH + directory + f"/{directory}.plan.txt"
            
            if os.path.exists(plan_path):
                with open(plan_path, "r") as plan_file:
                    generated_plan = plan_file.read().strip()
                
                ground_truth = dataset.data_dict[directory]['ground_truth']

                print(f"  Ground Truth: {ground_truth}")
                print(f"  Generated:    {generated_plan}")
                print()
        
        elif "Execution Flag: 1" in log_content:
            results_dict["parsing_error"] += 1
            print(f"✗ Task: {directory} - PARSING ERROR")
        
        elif "Execution Flag: 3" in log_content:
            results_dict["planner_error"] += 1
            print(f"✗ Task: {directory} - PLANNER ERROR")
        
        elif "Execution Flag: 4" in log_content:
            results_dict["no_plan_found"] += 1
            print(f"✗ Task: {directory} - NO PLAN FOUND")
        
        else:
            print(f"? Task: {directory} - UNKNOWN ERROR")

# Calculate success rates and print summary
print("\n" + "="*80)
print("SUMMARY BY DOMAIN")
print("="*80 + "\n")
print(f"{'Domain':<30} {'Total':<8} {'Success':<10} {'Parsing':<10} {'Planner':<10} {'No Plan':<10} {'Rate':<8}")
print("-"*86)

total_all = 0
successful_all = 0

for domain in DOMAINS:
    results = results_by_domain[domain]
    if results["total"] > 0:
        success_rate = (results["successful"] / results["total"]) * 100
    else:
        success_rate = 0.0
    
    results["success_rate"] = success_rate
    
    print(f"{domain:<30} {results['total']:<8} {results['successful']:<10} "
          f"{results['parsing_error']:<10} {results['planner_error']:<10} "
          f"{results['no_plan_found']:<10} {success_rate:>6.1f}%")
    
    total_all += results["total"]
    successful_all += results["successful"]

print("-"*86)
overall_rate = (successful_all / total_all * 100) if total_all > 0 else 0
print(f"{'TOTAL':<30} {total_all:<8} {successful_all:<10} {'':<10} {'':<10} {'':<10} {overall_rate:>6.1f}%")
print("="*86 + "\n")

# Save detailed results
with open(RESULTS_PATH + "final_comparison.json", "w") as f:
    json.dump(results_by_domain, f, indent=4)

print(f"Detailed results saved to {RESULTS_PATH}final_comparison.json")
print()
print("="*80)
print("Results by domain:")
for domain, results in results_by_domain.items():
    print(f"{domain}: {results}")
print("="*80)