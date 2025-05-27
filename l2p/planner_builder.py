from abc import ABC, abstractmethod
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.io import PDDLReader
from .utils.pddl_planner import FastDownward as FD

class Planner(ABC):
    
    @abstractmethod
    def solve(self, domain_path, problem_path):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_plan(self):
        raise NotImplementedError("This method should be overridden by subclasses")
     
class UP_Planner(Planner):
    def __init__(self, planner='aries'):
        """
        Initializes the UP_Planner with a specific planner name.
        :param planner: The name of the planner to use (default is 'aries').
        """
        super().__init__()
        self.planner_name = planner
        self.plan = None
        
    def solve(self, domain_path, problem_path = None):
        reader = PDDLReader()
        if problem_path is not None:
            problem = reader.parse_problem(domain_path, problem_path)
        else:
            problem = reader.parse_problem(domain_path)
            
        result = OneshotPlanner(name=self.planner_name, problem_kind=problem.kind).solve(problem)
        if result.plan is not None:
            plan = str(result.plan)
            print("Plan:", plan)
            self.plan = plan
            return plan
        else:
            print("ERROR: Not plan found:", result.status)
            return None
    
    def get_plan(self):
        return self.plan
    
class FastDownward(Planner, FD):
    def __init__(self, planner_path):
        """
        Initializes the Fast Downward planner with the path to the planner executable.
        :param planner_path: The path to the FastDownward planner executable.
        """
        super().__init__(planner_path=planner_path)
        self.planner_name = 'FastDownward'
        self.plan = None

    def solve(self, domain_path: str, problem_path: str, search_alg: str = "lama-first"):
        """
        Runs the FastDownward planner on the given domain and problem files.
        :param domain_path: Path to the PDDL domain file.
        :param problem_path: Path to the PDDL problem file.
        :return: The plan as a string if found, otherwise None.
        """
        is_success, plan = super().run_fast_downward(domain_file=domain_path, problem_file=problem_path, search_alg = search_alg)
        if is_success:
            print("Plan:", plan)
            self.plan = plan
            return plan
        else:
            print("ERROR: Not plan found")
            return None

    def get_plan(self):
        return self.plan