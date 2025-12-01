import os, json, sys
import traceback
from typing import Tuple

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from l2p.llm.base import BaseLLM as LLM
from l2p.model_builder import ModelBuilder
from l2p.planner_builder import Planner

class simpleNL2HTNAgent:
    """
    An agent that uses a language model to extract domain and problem from a task description,
    and then runs a planner to generate a plan.
    """
    
    def __init__(self, prompt_template: str, llm: LLM, builder: ModelBuilder, planner: Planner) -> None:
        """
        Initializes the NL2HTNAgent with a language model, a builder for the domain and problem,
        and a planner to generate plans.
        :param prompt_template: The template to use for prompting the LLM.
        :param llm: The language model to use for extracting the domain and problem.
        :param builder: The builder to use for constructing the domain and problem.
        :param planner: The planner to use for generating plans.
        """
        self.llm = llm
        self.builder = builder
        self.prompt_template = prompt_template
        self.planner = planner

    def run(self, task_desc: str, domain_path: str, problem_path: str, plan_path: str, response_path: str | None = None) -> Tuple[str, int]:
        """
        Runs the agent to extract the domain and problem, and then runs the planner.
        :param task_desc: The task description to run the agent on.
        :domain_path: Path to save the domain file.
        :problem_path: Path to save the problem file.
        :response_path: Path to save the LLM response.
        """

        # Extract the domain and problem using the LLM
        try:
            self.builder.formalize_domain_and_task(
                model=self.llm,
                task_desc=task_desc,
                prompt_template=self.prompt_template,
            )
            # print("Predicates:", self.builder.predicates)
        except Exception as e:
            return f"Error extracting domain and problem: {e}\n" + traceback.format_exc(), 1
        finally:
            if response_path is not None:
                # Save the LLM response to a file
                try:
                    with open(response_path, "w") as file:
                        file.write(self.builder.llm_response) # type: ignore
                except Exception as e:
                    return f"Error saving LLM response: {e}\n" + traceback.format_exc(), -1

        # Process the outputs to get domain and problem
        try:
            domain_str = self.builder.get_domain()
            problem_str = self.builder.get_problem()
        except Exception as e:
            return f"Error processing domain and problem: {e}\n" + traceback.format_exc(), 2
        
        # Save the domain and problem to files
        with open(domain_path, "w") as file:
            file.write(domain_str)
        with open(problem_path, "w") as file:
            file.write(problem_str)
        
        # Run planner
        try:
            plan = self.planner.solve(domain_path, problem_path)
        except Exception as e:
            return f"Error running planner: {e}\n" + traceback.format_exc(), 3
        
        # Write generated plan into folder
        try:
            with open(plan_path, "w") as file:
                file.write(plan)

        except Exception as e:
            return f"Error, no plan found: {e}\n" + traceback.format_exc(), 4
        
        return f"Plan generated successfully", 0



if __name__ == "__main__":
    from tests.human_llm import HumanLLM as LLM
    from l2p.dataset_builder import PlanBenchDataset
    from l2p.planner_builder import UP_Planner
    from l2p.model_builder import ModelBuilder
    from unified_planning.environment import get_environment
    get_environment().error_used_name = False  # Allow same names for different elements

    REQUIEREMENTS = [":strips", ":typing", ":hierarchy", ":negative-preconditions", ":conditional-effects",]
    TEMPLATE_PATH = "templates/l2hp_templates/extract_hddl_model.txt"
    RESULTS_PATH = "results_simpleNL2HTN_chatGPT/"
    THINKING_TIME_PATH = "results_simpleNL2HTN_chatGPT/blocksworld10/thinking_time.txt"
    
    PROMPT_PATH = "results_simpleNL2HTN_chatGPT/blocksworld10/prompt.txt"
    RESPONSE_PATH = "results_simpleNL2HTN_chatGPT/blocksworld10/ChatGPT_response.txt"
    llm = LLM(prompt_path=PROMPT_PATH, response_path=RESPONSE_PATH)  # Human-in-the-loop LLM for testing
    
    planner = UP_Planner('aries')
    dataset = PlanBenchDataset()
    builder = ModelBuilder("domain_placeholder", "problem_placeholder", isHTN=True, requirements=REQUIEREMENTS)

    with open(TEMPLATE_PATH, 'r') as file:
        extract_hddl_domain_and_problem_prompt =  file.read().strip()

    agent = simpleNL2HTNAgent(
        llm=llm,
        builder=builder,
        planner=planner,
        prompt_template=extract_hddl_domain_and_problem_prompt
    )
    
    # ====== Running the tasks ======

    DOMAINS = ["blocksworld", "depots", "logistics", "mystery_blocksworld", "obfuscated_deceptive_logistics"]
    tasks = [dataset.data_dict[domain+str(instance)] for instance in range(10,101,10) for domain in DOMAINS]

    # tasks = [dataset.data_dict['blocksworld10']]  # For testing purposes, only run one task

    results_summary = {
        "total_tasks": len(tasks),
        "successful_tasks": 0,
        "failed_tasks": 0,
        "-1": 0,  # Error flag
        "0": 0,   # Success flag
        "1": 0,   # Error in extraction
        "2": 0,   # Error in processing
        "3": 0,   # Error in planing
        "4": 0,   # Plan unsolvable
    }
        
    for task in tasks:
        
        task_directory = RESULTS_PATH + task['name'] + "/"

            # Create task directory if it doesn't exist
        print(f"Running task: {task['name']}")
        if not os.path.exists(task_directory):
            os.makedirs(task_directory)
        domain_path = task_directory + task['name'] + ".domain.hddl"
        problem_path = task_directory + task['name'] + ".problem.hddl"
        plan_path = task_directory + task['name'] + ".plan.txt"
        response_path = task_directory + task['name'] + ".llm_response.txt"
        log_path = task_directory + task['name'] + ".log.txt"

        agent.llm.prompt_path = task_directory + task['name'] + ".prompt.txt"
        agent.llm.response_path = task_directory + task['name'] + ".chatGPT_response.txt"
        agent.llm.thinking_time_path = task_directory + task['name'] + ".thinking_time.txt"
        # Extract domain and problem using the agent
        error_trace, execution_flag = agent.run(task['desc'], domain_path, problem_path, plan_path, response_path)

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
            with open(log_path, "w") as log_file:
                log_file.write(f"Execution Flag: {execution_flag}\nTask executed successfully.")
            print(f"Execution successful for task {task['name']}")
        
        # Save intermediate results summaryz
        with open(RESULTS_PATH + "results_summary.json", "w") as summary_file:
            json.dump(results_summary, summary_file, indent=4)
        print(f"Results summary: {results_summary}")


    #  ===== Final Comparation ======


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