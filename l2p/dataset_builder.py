"""
This file contains the builder for experiments with the different datasets.
It provides functionality to extract and process datasets, 
particularly for model and plan generation tasks.     
"""

from abc import ABC, abstractmethod
from typing import TypedDict
import pandas as pd

class NLTask(TypedDict):
    """
    TypedDict for natural language tasks.
    Contains task name, description, and ground truth.
    """
    name: str
    desc: str
    ground_truth: str

class Dataset(ABC):
    """
    Abstract base class for datasets.
    Provides methods to extract and process dataset files.
    """
    data_dict: dict[str, NLTask]

    def __init__(self) -> None:
        """
        Initializes the dataset.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
 
  
class PlanBenchDataset(Dataset):
    """
    Dataset class for PlanBench dataset.
    Provides methods to extract and process the dataset files.
    """

    def __init__(self, subset_parquet: str = 'task_1_plan_generation/train-00000-of-00001-f765a1b29ae17c5a.parquet') -> None:
        """
        Initializes the PlanBench dataset.
        """
        df = pd.read_parquet("hf://datasets/tasksource/planbench/" + subset_parquet)
        self.data_dict = self.preprocess_dataset(df)
  
    def preprocess_dataset(self, df: pd.DataFrame) -> dict[str, NLTask]:
        """
        Reformats the dataset from a DataFrame to a dictionary of tasks.
        :param df: The DataFrame containing the dataset.
        :return: A dictionary with task names as keys and nl_task as values.
        """
        tasks = {}
        for _, row in df.iterrows():
            task_name = row['domain'] + str(row['instance_id'])
            task_desc = self.substract_desc(row['query'])
            ground_truth = row['ground_truth_plan']
            tasks[task_name] = {
                'name': task_name,
                'desc': task_desc,
                'ground_truth': ground_truth
            }
        return tasks

    def substract_desc(self, task_desc: str) -> str:
        """
        Extracts the description from a raw planbench task string.
        :param task_desc: The task description string.
        :return: The extracted description.
        """
        domain = task_desc.split("[STATEMENT]")[0]
        problem = task_desc.split("[STATEMENT]")[2].split("My plan is as follows:")[0]

        return domain + problem
    