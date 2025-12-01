from typing import Tuple
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from l2p.prompt_builder import PromptBuilder
from l2p.utils.pddl_parser import load_file, load_files, prune_predicates
from l2p.llm.genai import GenAIClient as LLM
from l2p.model_builder import ModelBuilder
from l2p.planner_builder import Planner, UP_Planner
from mySecrets import GeminiApiStudent_token as token
from unified_planning.environment import get_environment

class iterativeNL2HTNAgent:
    """
    An agent that uses a language model to iteratively extract and formalize an HTN planning domain
    """
    
    def __init__(self, llm: LLM, builder: ModelBuilder, planner: Planner, htn_mode = True) -> None:
        """
        Initializes the NL2HTNAgent with a language model, a builder for the domain and problem,
        and a planner to generate plans.
        :param prompt_template: The template to use for prompting the LLM.
        :param llm: The language model to use for extracting the domain and problem.
        :param builder: The builder to use for constructing the domain and problem.
        :param planner: The planner to use for generating plans.
        """
        self.llm = llm
        builder.isHTN = htn_mode
        self.builder = builder
        self.planner = planner
        self.prompt_builder = PromptBuilder()
        self.depuration_mode = False
        self.htn_mode = htn_mode
    
    def generate_prompt(
        self, role_path: str | None = None, examples_path: str | None = None, task_path: str | None = None
    ) -> str:

        # load in files
        if role_path:
            role = load_file(role_path)
            self.prompt_builder.set_role(role=role)
        if examples_path:
            examples = load_files(examples_path)
            for ex in examples:
                self.prompt_builder.set_examples(example=ex)
        if task_path:
            task = load_file(task_path)
            self.prompt_builder.set_task(task=task)

        return self.prompt_builder.generate_prompt()
      
    def run(self, task_desc: str, domain_path: str, problem_path: str, plan_path: str, response_path: str | None = None) -> Tuple[str, int]:
        """
        Runs the agent to extract the domain and problem, and then runs the planner.
        :param task_desc: The task description to run the agent on.
        :domain_path: Path to save the domain file.
        :problem_path: Path to save the problem file.
        :response_path: Path to save the LLM response.
        """
        # A. EXTRACTION PHASE
        
        # 1. Type Extraction
        types, llm_output, validation_info = self.builder.formalize_types(
            model=self.llm,
            domain_desc=task_desc,
            prompt_template=self.generate_prompt(
                role_path="paper_reconstructions/nl2plan/prompts/type_extraction/role.txt",
                examples_path="paper_reconstructions/nl2plan/prompts/type_extraction/examples",
                task_path="paper_reconstructions/nl2plan/prompts/type_extraction/task.txt",
            )
        )
        
        assert(validation_info[0]), f"Type extraction failed validation: {validation_info[1]}"
        self.builder.types = types
        
        # 2. Hierarchy Construction
        hierarchy, llm_output, validation_info = self.builder.formalize_type_hierarchy(
            model=self.llm,
            domain_desc=task_desc,
            types=self.builder.types,
            prompt_template=self.generate_prompt(
                role_path="paper_reconstructions/nl2plan/prompts/hierarchy_construction/role.txt",
                examples_path="paper_reconstructions/nl2plan/prompts/hierarchy_construction/examples",
                task_path="paper_reconstructions/nl2plan/prompts/hierarchy_construction/task.txt"
            )
        )        
        
        if self.depuration_mode:
            print("Type Hierarchy LLM Output:\n", llm_output, "\n-------------------------\n")
            print("Type Hierarchy:", hierarchy)
            input("Press Enter to continue...")

        assert(validation_info[0]), f"Hierarchy construction failed validation: {validation_info[1]}"
        self.builder.type_hierarchy = hierarchy
           
           
        if self.htn_mode:   
            # 3. Tasks Construction
            # 4. Tasks Construction
            tasks, llm_output  = self.builder.formalize_HPDLTasks(
                model=self.llm,
                domain_desc=task_desc,
                types=self.builder.type_hierarchy,
                prompt_template=self.generate_prompt(
                    role_path="templates/l2hp_templates/htn_task_extraction/role.txt",
                    examples_path="templates/l2hp_templates/htn_task_extraction/examples",
                    task_path="templates/l2hp_templates/htn_task_extraction/task.txt",
                )
            )
            
            assert(len(tasks) > 0), "No tasks were extracted."
            
            nl_tasks_list = [f"{name} {' '.join([f'{param} - {ptype}' for param, ptype in task['params'].items()])}: {task['desc']}" for name, task in tasks.items()]
            self.builder.tasks = tasks
            
            if self.depuration_mode:
                print("NL Tasks LLM Output:\n", llm_output, "\n-------------------------\n")
                print("NL Tasks List:")
                for nl_task in nl_tasks_list:
                    print('\t-', nl_task)
                input("Press Enter to continue...")
            
            # 5. Methods Extraction
            prompt = self.generate_prompt(
                role_path="templates/l2hp_templates/htn_method_extraction/role.txt",
                examples_path="templates/l2hp_templates/htn_method_extraction/examples",
                task_path="templates/l2hp_templates/htn_method_extraction/task.txt",
            )
                
            for task_name, task in self.builder.tasks.items():
                methods_dict, llm_output = self.builder.extract_methods_list(
                    model=self.llm,
                    domain_desc=task_desc,
                    types=self.builder.type_hierarchy,
                    tasks=nl_tasks_list,
                    task=task,
                    prompt_template=prompt
                )
                
                assert(len(methods_dict) > 0), f"No methods were extracted for task {task_name}."
                
                task['methods'] = [{'name': method_name, 'desc': method_desc} for method_name, method_desc in methods_dict.items()]
                
                self.builder.tasks[task_name] = task
                
                if self.depuration_mode:
                    print(f"NL Methods LLM Output for task {task_name}:\n", llm_output, "\n-------------------------\n")
                    print(f"NL Methods for task {task_name}:")
                    for method_name, method_desc in methods_dict.items():
                        print(f"\t- {method_name}: {method_desc}")
                    input("Press Enter to continue...")
                                    
            # 6. Methods Construction
            predicates = []
            nl_actions = {}
            
            prompt = self.generate_prompt(
                role_path="templates/l2hp_templates/htn_method_construction/role.txt",
                examples_path="templates/l2hp_templates/htn_method_construction/examples",
                task_path="templates/l2hp_templates/htn_method_construction/task.txt",
            )
            
            methods = []
            for task_name, task in self.builder.tasks.items():
                formalized_methods = []
                # For each method in the task
                for method in task['methods']:
                    method_formal, new_predicates, new_actions, llm_output = self.builder.formalize_method(
                        model=self.llm,
                        domain_desc=task_desc,
                        types=self.builder.type_hierarchy,
                        method_name=method['name'],
                        method_desc=method['desc'],
                        method_task=task,
                        tasks_list=nl_tasks_list,
                        predicates_list=predicates,
                        nl_actions=nl_actions,
                        prompt_template=prompt,
                    )
                    
                    formalized_methods.append(method_formal)
                    predicates.extend(new_predicates)
                    # predicates = prune_predicates(predicates, formalized_methods)
                    nl_actions.update(new_actions)
                
                task['methods'] = formalized_methods
                methods.extend(formalized_methods)

            self.builder.predicates = predicates
            
            if self.depuration_mode:
                print("Last LLM Methods Construction Output:\n", llm_output, "\n-------------------------\n")
                print("\nFormalized HTN Methods:")
                if len(methods) > 0:
                    for method in methods:
                        print(f"\n{method['name']}: {method['desc']}")
                        print(f"\tPARAMETERS: {method['params']}")
                        print(f"\tPRECONDITIONS: {method['preconditions']}")
                        print(f"\tTASK: {method['task']}")
                        print(f"\tSUBTASKS: {method['ordered_subtasks']}")
                else:
                    print("No methods formalized.")
                print("\nUpdated Predicates:")
                for predicate in predicates:
                    print(f"\t- {predicate['clean']} : {predicate['desc']}")
                print("\nUpdated NL Actions:")
                for action_name, action_desc in nl_actions.items():
                    print(f"\t- {action_name}: {action_desc}")
                input("\nPress Enter to continue...")


        if not self.htn_mode:
            # 7. Actions Extraction
            nl_actions, llm_output  = self.builder.extract_nl_actions(
                model=self.llm,
                domain_desc=task_desc,
                types=self.builder.type_hierarchy,
                prompt_template=self.generate_prompt(
                    role_path="paper_reconstructions/nl2plan/prompts/action_extraction/role.txt",
                    examples_path="paper_reconstructions/nl2plan/prompts/action_extraction/examples",
                    task_path="paper_reconstructions/nl2plan/prompts/action_extraction/task.txt",
                )
            )
        
            assert(len(nl_actions) > 0), "No actions were extracted."
            
            if self.depuration_mode:
                print("NL Actions LLM Output:\n", llm_output, "\n-------------------------\n")
                print("NL Actions:")
                for action_name, action_desc in nl_actions.items():
                    print(f"\t- {action_name}: {action_desc}")
                input("Press Enter to continue...")
        
        # 8. Actions Construction
        actions = []
        predicates = []
        
        nl_actions_list = [f"{name}: {desc}" for name, desc in nl_actions.items()]
        prompt = self.generate_prompt(
            role_path="paper_reconstructions/nl2plan/prompts/action_construction/role.txt",
            examples_path="paper_reconstructions/nl2plan/prompts/action_construction/examples",
            task_path="paper_reconstructions/nl2plan/prompts/action_construction/task.txt",
        )
        for action_name, action_desc in nl_actions.items():
            action, new_predicates, llm_output, validation_info = self.builder.formalize_pddl_action(
                model=self.llm,
                domain_desc=task_desc,
                types=self.builder.type_hierarchy,
                action_name=action_name,
                action_desc=action_desc,
                action_list=nl_actions_list,
                predicates=predicates,
                extract_new_preds=True,
                prompt_template=prompt,
            )
                        
            actions.append(action)
            predicates.extend(new_predicates)
            predicates = prune_predicates(predicates, actions)

            assert(validation_info[0]), f"Action {action_name} construction failed validation: {validation_info[1]}"

        self.builder.actions = actions
        self.builder.predicates = predicates
        
        if self.depuration_mode:
            print("Last LLM Action Construction Output:\n", llm_output, "\n-------------------------\n")
            print("PDDL Predicates:")
            for predicate in predicates:
                print(f"\t- {predicate['clean']} : {predicate['desc']}")
            print("PDDL Actions:")
            for action in actions:
                print(f"\n{action['name']}:")
                print(f"\tPARAMETERS: {action['params']}")
                print(f"\tPRECONDITIONS: {action['preconditions']}")
                print(f"\tEFFECTS: {action['effects']}")
            input("Press Enter to continue...")

        # 9. Predicates Extraction
        ## Not needed as predicates are extracted during action construction.

        # 10. Objects Extraction
        # 11. Initial State Extraction
        # 12. Goal Extraction
        objects, initial, goal, llm_output, validation_info = self.builder.formalize_task(
            model=self.llm,
            problem_desc=task_desc,
            types=types,
            predicates=predicates,
            prompt_template=self.generate_prompt(
                role_path="paper_reconstructions/nl2plan/prompts/task_extraction/role.txt",
                examples_path="paper_reconstructions/nl2plan/prompts/task_extraction/examples",
                task_path="paper_reconstructions/nl2plan/prompts/task_extraction/task.txt",
            ),
        )

        assert(validation_info[0]), f"Task extraction failed validation: {validation_info[1]}"

        self.builder.objects = objects
        self.builder.initial = initial
        self.builder.goal = goal
        
        if self.depuration_mode:
            print("Task LLM Output:\n", llm_output, "\n-------------------------\n")
            print("Objects:", objects)
            print("Initial State:", initial)
            print("Goal State:", goal)
            input("Press Enter to continue...")
             
        # B. BUILDING PHASE
        # Domain file
        domain = self.builder.get_domain()

        # Problem file
        problem = self.builder.get_problem()
        
        if self.depuration_mode:
            print("PDDL Domain:\n", domain)
            print("PDDL Problem:\n", problem)
            input("Press Enter to continue...")
        
        # C. PLANNING PHASE
        # Plan Generation
        self.planner.solve_str(domain=domain, problem=problem)  # type: ignore
        
        print(self.planner.get_plan())

    
if __name__ == "__main__":
    get_environment().error_used_name = False  # Allow same names for different elements

    planner = UP_Planner('aries')
    llm = LLM('gemini-2.0-flash', api_key=token)
    builder = ModelBuilder(domain_name="blocks-world", problem_name="move-blocks", requirements=[':typing'])
    agent = iterativeNL2HTNAgent(llm=llm, builder=builder, planner=planner)
    agent.depuration_mode = True

    agent.run(
        task_desc="Move the red block from A to B using a robot arm.",
        domain_path="output/domain.pddl",
        problem_path="output/problem.pddl",
        plan_path="output/plan.txt",
        response_path="output/llm_response.txt"
    )