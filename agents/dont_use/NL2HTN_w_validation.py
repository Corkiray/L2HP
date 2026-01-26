import os, json, sys
import traceback
from typing import Tuple

from l2p.llm.base import BaseLLM as LLM
from l2p.model_builder import ModelBuilder
from l2p.planner_builder import Planner
from l2p.utils.pddl_validator import SyntaxValidator

class NL2HTNAgentWithValidation:
    """
    An agent that uses a language model to extract domain and problem from a task description,
    and then runs a planner to generate a plan. Now with validation step.
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
        self.validation_functions = [
            "validate_type",
            "validate_format_types", # Done
            "validate_cyclic_types", # Done
            # "validate_constant_types", No sense here
            # "validate_format_functions", No sense here
            "validate_types_predicates", # Done
            # "validate_duplicate_predicates", No sense here
            "validate_overflow_predicates",
            "validate_format_predicates", # Done
            "validate_pddl_action",
            "validate_params",
            "validate_usage_action",
            "validate_task_objects", # Done
            "validate_task_states", 
            "validate_header", # Done
            "validate_duplicate_headers", # Done
            "validate_unsupported_keywords" # Done
        ]
        
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

        # Run validations
        available_args = {
            "domain_str": self.builder.get_domain(),
            "problem_str": self.builder.get_problem(),
            "task_desc": task_desc
        }
        
        is_valid_response = False
        while not is_valid_response:
            validation_results = run_validations(self.validation_functions, available_args)
            for validator_name, valid, message in validation_results:
                if not valid:
                    return f"Validation failed ({validator_name}): {message}", -1


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
    
    
def get_validator_parameters(validator_func) -> list[str]:
    """
    Gets the required parameters for a validator function using introspection.
    """
    import inspect
    params = inspect.signature(validator_func).parameters
    return [name for name in params.keys() if name != 'self']

    
def validate(syntax_validator: SyntaxValidator, error_type: str, available_args: dict) -> tuple[bool, str]:
    """
    Dynamically calls a validator function with the correct arguments.
    
    Args:
        syntax_validator: Instance of SyntaxValidator containing validation methods
        error_type: Name of the validation function to call
        available_args: Dictionary of available arguments to pass
    
    Returns:
        tuple[bool, str]: Validation result and message
    """
    validator = getattr(syntax_validator, error_type, None)
    if not validator:
        return False, f"Validator {error_type} not found"
    
    required_args = get_validator_parameters(validator)
    args_to_pass = {arg: available_args.get(arg) for arg in required_args}
    
    return validator(**args_to_pass)


def run_validations(validators_list: list[str], available_args: dict) -> list[tuple[str, bool, str]]:
    """
    Runs all specified validations with the provided arguments.
    
    Args:
        available_args: Dictionary containing all possible arguments for validators
    
    Returns:
        list[tuple[str, bool, str]]: List of (validator_name, success flag, error message) tuples
    """
    results = []
    
    syntax_validator = SyntaxValidator(error_types=validators_list)
    for error_type in validators_list:
        valid, message = validate(syntax_validator, error_type, available_args)
        results.append((error_type, valid, message))
    return results



        # all_validator_args = [
        #     "types",
        #     "llm_response",
        
              # Type Validations
        #     "target_type",
        #     "claimed_type", 
        #     "constants",
        
              # Function Validations
        #     "functions",
        
              # Predicate Validations
        #     "predicates",
        #     "curr_predicates",
        #     "new_predicates",
        #     "limit",
        
              # Action Validations
        #     "pddl",
        #     "action_params",
        #     "part",
        #     "parameters",
        #     "extract_new_preds",
        
              # Task Validations

        #     "objects",
        #     "states",
        #     "state_type"
        # ]