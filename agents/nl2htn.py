import os, json, sys
from typing import Tuple
# WORKING_DIR = "/mnt/homeGPU/ipuerta/l2p-htn"

# os.chdir(WORKING_DIR)
# sys.path.append(os.getcwd())
# print(os.getcwd())
# from l2p.utils.pddl_parser import prune_predicates, format_types
# from tests.mock_llm import MockLLM

from l2p.llm_builder import LLM
from l2p.model_builder import ModelBuilder
from l2p.planner_builder import Planner


class NL2HTNAgent:
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

    def run(self, task_desc: str, domain_path: str, problem_path: str, plan_path: str) -> Tuple[str, int]:
        """
        Runs the agent to extract the domain and problem, and then runs the planner.
        :param task_desc: The task description to run the agent on.
        :domain_path: Path to save the domain file.
        :problem_path: Path to save the problem file.
        """
        
        # Extract the domain and problem using the LLM
        try:
            llm_output = self.builder.extract_domain_and_problem(
                model=self.llm,
                task_desc=task_desc,
                prompt_template=self.prompt_template,
            )
            # print("LLM Output:", llm_output)
        except Exception as e:
            return f"Error extracting domain and problem: {e}", 1
        
        # Process the outputs to get domain and problem
        try:
            domain_str = self.builder.get_domain()
            problem_str = self.builder.get_problem()     
        except Exception as e:
            return f"Error processing domain and problem: {e}", 2
        
        # Save the domain and problem to files
        with open(domain_path, "w") as file:
            file.write(domain_str)
        with open(problem_path, "w") as file:
            file.write(problem_str)
        
        # Run planner
        try:
            plan = self.planner.solve(domain_path, problem_path)
        except Exception as e:
            return f"Error running planner: {e}", 3
        
        # Write generated plan into folder
        with open(plan_path, "w") as file:
            file.write(plan)
        
        
        
        return f"Plan generated successfully", 0




