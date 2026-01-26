"""
Advanced Experimentation Script for PDDLModeler Framework

This script provides a comprehensive experimentation platform for the PDDLModeler
with features including:
- Automated batch processing of tasks from PlanBench dataset
- Detailed metrics collection and analysis
- Progress tracking and resumable execution (skips completed tasks)
- Comparative analysis across domains
- CSV export for further analysis
- Comprehensive error handling and logging

═══════════════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════════════

1. Run with defaults:
    python pddl_modeler_experimentation.py

2. Run quick test (1 task per domain):
    Modify line in __main__: max_tasks_per_domain=1

3. Run for specific model:
    Modify line in __main__: model_name="qwen2.5"

4. Run HTN experiments:
    Modify line in __main__: is_htn=True

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION SYSTEM
═══════════════════════════════════════════════════════════════════════════════

This script uses a two-level configuration system:

Level 1: GLOBAL DEFAULTS (DEFAULT_* constants at top of file)
  - Used by ExperimentConfig when no explicit value provided
  - Modify these to change ALL experiment runs
  - Example: DEFAULT_OLLAMA_MODEL = "mistral"

Level 2: INSTANCE CONFIGURATION (ExperimentConfig in __main__)
  - Overrides global defaults for this run
  - Modify ExperimentConfig() call in __main__
  - Example: ExperimentConfig(model_name="qwen2.5")

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION CATEGORIES
═══════════════════════════════════════════════════════════════════════════════

Model & LLM:
  - model_name: Which LLM to use
  - ollama_host: Server address
  - structured_output_mode: How to generate structured output

Generation Strategy:
  - is_htn: Classical PDDL vs Hierarchical Task Networks
  - use_analysis: Analyze problem before generation
  - think_mode: Two-step reasoning (think then generate)
  - max_retries: How many times to retry on failure

Dataset:
  - domains: Which planning domains to test
  - instance_range_start/end/step: Which instances within each domain
  - max_tasks_per_domain: Limit tasks per domain (useful for testing)

Execution:
  - resumable_execution: Skip already completed tasks
  - force_rerun: Re-run even completed tasks
  - planner_timeout: How long to wait for plan generation

Output & Storage:
  - results_dir: Where to save results
  - save_metrics_csv: Export results to CSV
  - save_summary_json: Export summary to JSON

Display:
  - show_progress: Show progress updates
  - show_timing: Display timing information
  - console_width: Width for output formatting

═══════════════════════════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

# Quick test with small dataset:
config = ExperimentConfig(
    domains=["blocksworld"],
    max_tasks_per_domain=1,
    show_progress=True
)

# Full experiment with custom model:
config = ExperimentConfig(
    model_name="qwen2.5",
    is_htn=False,
    max_tasks_per_domain=None,  # Use all available instances
    track_metrics=True,
    save_intermediate=True
)

# Production HTN with extensive logging:
config = ExperimentConfig(
    is_htn=True,
    model_name="llama3.1",
    log_level="DEBUG",
    save_generation_logs=True,
    track_metrics=True,
    collect_error_stats=True,
    resumable_execution=True
)

Author: Generated for PDDLModeler Framework
"""

import json
import sys
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from l2hp.agents.UltimateModeler import (
    PDDLModeler, 
    create_structured_client,
    LogLevel
)
from l2p.dataset_builder import PlanBenchDataset
from l2p.planner_builder import UP_Planner
from unified_planning.environment import get_environment


# =============================================================================
# GLOBAL CONFIGURATION CONSTANTS
# =============================================================================
# All framework defaults are centralized here for easy customization.
# Modify these values to change default behavior across the entire script.
#
# CONSTANT CATEGORIES:
#   - Ollama connection: Server address and model name
#   - Generation settings: HTN mode, analysis, thinking
#   - Execution control: Retry policy, timeouts
#   - Logging: Verbosity and output files
#   - Dataset: Domains and instances
#   - Planner: Solver selection and timeouts
#   - Output: Results directory and formats
#   - Display: Formatting and progress indicators
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CONNECTION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_MODEL = "gemma3"  # Model name (llama3.1, qwen2.5, mistral, gemma3, etc.)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # Ollama server URL
DEFAULT_STRUCTURED_OUTPUT_MODE = "instructor"  # "instructor", "outlines", or "raw_ollama"

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_IS_HTN = False  # True for HTN/HDDL, False for classical PDDL
DEFAULT_USE_ANALYSIS = True  # Enable initial problem analysis step
DEFAULT_THINK_MODE = True  # Enable two-step thinking (think → JSON generation)
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts on generation failure

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING AND DEBUGGING
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LOG_LEVEL = "INFO"  # QUIET, INFO, DEBUG, TRACE (PDDLModeler verbosity)
DEFAULT_INTERACTIVE_MODE = False  # Enable interactive mode (pause at each step)
DEFAULT_SAVE_GENERATION_LOGS = True  # Save PDDLModeler logs for each task (.generation.log)
DEFAULT_SAVE_EXECUTION_LOG = True  # Save this script's execution logs for each task (.execution.log)
DEFAULT_TRACK_METRICS = True  # Track generation metrics and timing (PDDLModeler)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DOMAINS = [
    "blocksworld",
    "depots",
    "logistics",
    "mystery_blocksworld",
    "obfuscated_deceptive_logistics"
]
DEFAULT_INSTANCE_RANGE_START = 10  # First instance number (inclusive)
DEFAULT_INSTANCE_RANGE_END = 101  # Last instance number (exclusive)
DEFAULT_INSTANCE_STEP = 1  # Step size for instances (e.g., 10, 20, 30...)
DEFAULT_MAX_TASKS_PER_DOMAIN = None  # None = no limit, or set max (e.g., 5 for testing)


# =========================== [WARNING] TESTING CONFIGURATION. REMOVE OR COMMENT OUT FOR FULL EXPERIMENTS] ============================
# DEFAULT_DOMAINS = [
#     "blocksworld",
#     "depots",
# ]
# DEFAULT_INSTANCE_RANGE_START = 10  # First instance number (inclusive)
# DEFAULT_INSTANCE_RANGE_END = 101  # Last instance number (exclusive)
# DEFAULT_INSTANCE_STEP = 10  # Step size for instances (e.g., 10, 20, 30...)
# DEFAULT_MAX_TASKS_PER_DOMAIN = 3  # None = no limit, or set max (e.g., 5 for testing)
# =====================================================================================================================================


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION CONTROL
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PLANNER_NAME = "fast-downward"  # "fast-downward" (PDDL) or "aries" (HTN)
DEFAULT_PLANNER_TIMEOUT = 60  # Seconds to wait for planner
DEFAULT_CONTINUE_ON_ERROR = True  # Continue to next task if one fails
DEFAULT_RESUMABLE_EXECUTION = True  # Skip already completed tasks
DEFAULT_FORCE_RERUN = False  # If True, re-run even completed tasks

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = "results_from_testing_outlines"  # Base results directory
DEFAULT_RESULTS_DIR_SUFFIX_MODE = True  # Append "_pddl" or "_htn" to results_dir
DEFAULT_SAVE_MODEL_JSON = True  # Save generated model as JSON
DEFAULT_SAVE_DOMAIN_PDDL = True  # Save PDDL domain file
DEFAULT_SAVE_PROBLEM_PDDL = True  # Save PDDL problem file
DEFAULT_SAVE_PLAN = True  # Save generated plan file
DEFAULT_SAVE_METRICS_CSV = True  # Save detailed results to CSV
DEFAULT_SAVE_SUMMARY_JSON = True  # Save summary to JSON
DEFAULT_SAVE_INTERMEDIATE = True  # Save results after each task

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY AND FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SHOW_PROGRESS = True  # Show progress bars and live updates
DEFAULT_SHOW_TIMING = True  # Display timing information
DEFAULT_SHOW_METRICS_DETAIL = True  # Show detailed metrics in output
DEFAULT_PROGRESS_INTERVAL = 1  # Update progress every N tasks
DEFAULT_CONSOLE_WIDTH = 80  # Width for console separators

# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLING AND RECOVERY
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SKIP_GENERATION_ERRORS = False  # Skip task on generation error
DEFAULT_SKIP_VALIDATION_ERRORS = False  # Skip task on validation error
DEFAULT_SKIP_PLANNER_ERRORS = False  # Skip task on planner error
DEFAULT_SKIP_NO_PLAN_FOUND = False  # Skip task if no plan found
DEFAULT_COLLECT_ERROR_STATS = True  # Collect and report error statistics


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Centralized configuration for experiments.
    
    All values default to the global constants defined at the top of this file.
    Override any value when creating an ExperimentConfig instance.
    
    Usage:
        # Use all defaults
        config = ExperimentConfig()
        
        # Override specific values
        config = ExperimentConfig(
            model_name="qwen2.5",
            is_htn=True,
            domains=["blocksworld", "logistics"]
        )
    """
    
    # ─── Model Configuration ───
    model_name: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    structured_output_mode: str = DEFAULT_STRUCTURED_OUTPUT_MODE
    
    # ─── Generation Settings ───
    is_htn: bool = DEFAULT_IS_HTN
    use_analysis: bool = DEFAULT_USE_ANALYSIS
    think_mode: bool = DEFAULT_THINK_MODE
    max_retries: int = DEFAULT_MAX_RETRIES
    
    # ─── Logging and Debugging ───
    log_level: str = DEFAULT_LOG_LEVEL
    interactive_mode: bool = DEFAULT_INTERACTIVE_MODE
    save_generation_logs: bool = DEFAULT_SAVE_GENERATION_LOGS
    save_execution_log: bool = DEFAULT_SAVE_EXECUTION_LOG
    track_metrics: bool = DEFAULT_TRACK_METRICS
    
    # ─── Dataset Configuration ───
    domains: List[str] = field(default_factory=lambda: DEFAULT_DOMAINS.copy())
    instance_range_start: int = DEFAULT_INSTANCE_RANGE_START
    instance_range_end: int = DEFAULT_INSTANCE_RANGE_END
    instance_step: int = DEFAULT_INSTANCE_STEP
    max_tasks_per_domain: Optional[int] = DEFAULT_MAX_TASKS_PER_DOMAIN
    
    # ─── Execution Control ───
    planner_name: str = DEFAULT_PLANNER_NAME
    planner_timeout: int = DEFAULT_PLANNER_TIMEOUT
    continue_on_error: bool = DEFAULT_CONTINUE_ON_ERROR
    resumable_execution: bool = DEFAULT_RESUMABLE_EXECUTION
    force_rerun: bool = DEFAULT_FORCE_RERUN
    
    # ─── Output Configuration ───
    results_dir: str = DEFAULT_RESULTS_DIR
    results_dir_suffix_mode: bool = DEFAULT_RESULTS_DIR_SUFFIX_MODE
    save_model_json: bool = DEFAULT_SAVE_MODEL_JSON
    save_domain_pddl: bool = DEFAULT_SAVE_DOMAIN_PDDL
    save_problem_pddl: bool = DEFAULT_SAVE_PROBLEM_PDDL
    save_plan: bool = DEFAULT_SAVE_PLAN
    save_metrics_csv: bool = DEFAULT_SAVE_METRICS_CSV
    save_summary_json: bool = DEFAULT_SAVE_SUMMARY_JSON
    save_intermediate: bool = DEFAULT_SAVE_INTERMEDIATE
    
    # ─── Display and Formatting ───
    show_progress: bool = DEFAULT_SHOW_PROGRESS
    show_timing: bool = DEFAULT_SHOW_TIMING
    show_metrics_detail: bool = DEFAULT_SHOW_METRICS_DETAIL
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL
    console_width: int = DEFAULT_CONSOLE_WIDTH
    
    # ─── Error Handling ───
    skip_generation_errors: bool = DEFAULT_SKIP_GENERATION_ERRORS
    skip_validation_errors: bool = DEFAULT_SKIP_VALIDATION_ERRORS
    skip_planner_errors: bool = DEFAULT_SKIP_PLANNER_ERRORS
    skip_no_plan_found: bool = DEFAULT_SKIP_NO_PLAN_FOUND
    collect_error_stats: bool = DEFAULT_COLLECT_ERROR_STATS
    
    def __post_init__(self):
        """Validate and set dependent configurations."""
        # Auto-adjust planner for HTN mode
        if self.is_htn and self.planner_name == "fast-downward":
            self.planner_name = "aries"
            print(f"  → Auto-adjusted planner to 'aries' for HTN mode")
        
        # Add mode suffix to results directory if enabled
        if self.results_dir_suffix_mode:
            mode_suffix = "_htn" if self.is_htn else "_pddl"
            if not self.results_dir.endswith(mode_suffix):
                self.results_dir = self.results_dir.rstrip('/') + mode_suffix
        
        # Create results directory
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate domain list
        if not self.domains:
            print("⚠️  WARNING: No domains specified, using defaults")
            self.domains = DEFAULT_DOMAINS.copy()
        
        # Validate instance range
        if self.instance_range_end <= self.instance_range_start:
            raise ValueError(f"instance_range_end ({self.instance_range_end}) must be > instance_range_start ({self.instance_range_start})")
        
        if self.instance_step <= 0:
            raise ValueError(f"instance_step must be positive, got {self.instance_step}")
    
    def get_instance_range(self) -> Tuple[int, int, int]:
        """Get instance range as (start, end, step) tuple."""
        return (self.instance_range_start, self.instance_range_end, self.instance_step)


# =============================================================================
# METRICS AND RESULTS TRACKING
# =============================================================================

@dataclass
class TaskResult:
    """Results for a single task execution."""
    task_name: str
    domain: str
    instance: int
    
    # Execution status
    success: bool
    exit_code: int  # 0=success, 1=generation_error, 2=validation_error, 3=planner_error, 4=no_plan
    error_message: str = ""
    
    # Timing metrics
    total_time: float = 0.0
    generation_time: float = 0.0
    planning_time: float = 0.0
    
    # Generation metrics (from PDDLModeler.metrics if available)
    num_types: int = 0
    num_predicates: int = 0
    num_actions: int = 0
    num_objects: int = 0
    num_tasks: int = 0  # HTN only
    num_methods: int = 0  # HTN only
    num_initial_facts: int = 0
    num_goal_facts: int = 0
    
    # LLM metrics (if available)
    total_llm_calls: int = 0
    successful_llm_calls: int = 0
    failed_llm_calls: int = 0
    avg_attempts_per_step: float = 0.0
    
    # Plan quality
    plan_length: int = 0
    plan_found: bool = False
    
    # File paths
    domain_file: str = ""
    problem_file: str = ""
    plan_file: str = ""
    log_file: str = ""
    model_file: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ResultsTracker:
    """Track and analyze results across all tasks."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[TaskResult] = []
        self.start_time = time.time()
    
    def add_result(self, result: TaskResult):
        """Add a task result."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.results)
        if total == 0:
            return {"total_tasks": 0}
        
        successful = sum(1 for r in self.results if r.success)
        
        summary = {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": total - successful,
            "success_rate": successful / total * 100,
            "avg_total_time": sum(r.total_time for r in self.results) / total,
            "avg_generation_time": sum(r.generation_time for r in self.results) / total,
            "avg_planning_time": sum(r.planning_time for r in self.results if r.plan_found) / max(1, sum(1 for r in self.results if r.plan_found)),
            "total_elapsed_time": time.time() - self.start_time,
            
            # Error breakdown
            "generation_errors": sum(1 for r in self.results if r.exit_code == 1),
            "validation_errors": sum(1 for r in self.results if r.exit_code == 2),
            "planner_errors": sum(1 for r in self.results if r.exit_code == 3),
            "no_plan_found": sum(1 for r in self.results if r.exit_code == 4),
            
            # Average model sizes
            "avg_types": sum(r.num_types for r in self.results) / total,
            "avg_predicates": sum(r.num_predicates for r in self.results) / total,
            "avg_actions": sum(r.num_actions for r in self.results) / total,
            "avg_objects": sum(r.num_objects for r in self.results) / total,
        }
        
        if self.config.is_htn:
            summary["avg_tasks"] = sum(r.num_tasks for r in self.results) / total
            summary["avg_methods"] = sum(r.num_methods for r in self.results) / total
        
        return summary
    
    def get_domain_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by domain."""
        domain_results = {}
        
        for domain in self.config.domains:
            domain_tasks = [r for r in self.results if r.domain == domain]
            if not domain_tasks:
                continue
            
            total = len(domain_tasks)
            successful = sum(1 for r in domain_tasks if r.success)
            
            domain_results[domain] = {
                "total": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": successful / total * 100 if total > 0 else 0,
                "avg_time": sum(r.total_time for r in domain_tasks) / total,
                "generation_errors": sum(1 for r in domain_tasks if r.exit_code == 1),
                "planner_errors": sum(1 for r in domain_tasks if r.exit_code == 3),
                "no_plan_found": sum(1 for r in domain_tasks if r.exit_code == 4),
            }
        
        return domain_results
    
    def save_to_csv(self, filepath: str):
        """Save detailed results to CSV."""
        if not self.results:
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
    
    def save_summary(self, filepath: str):
        """Save summary to JSON."""
        summary = {
            "config": asdict(self.config),
            "overall": self.get_summary(),
            "by_domain": self.get_domain_summary(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# TASK EXECUTION
# =============================================================================

class TaskExecutor:
    """Execute individual tasks with the PDDLModeler."""
    
    def __init__(self, config: ExperimentConfig, dataset: PlanBenchDataset):
        self.config = config
        self.dataset = dataset
        
        # Initialize planner
        self.planner = UP_Planner(config.planner_name)
        
        # Configure Unified Planning environment
        env = get_environment()
        env.error_used_name = False
        env.credits_stream = None
        
        print(f"\n{'='*80}")
        print(f"TASK EXECUTOR INITIALIZED")
        print(f"{'='*80}")
        print(f"  Model: {config.model_name}")
        print(f"  Mode: {'HTN (HDDL)' if config.is_htn else 'Classical (PDDL)'}")
        print(f"  Structured Output: {config.structured_output_mode}")
        print(f"  Think Mode: {config.think_mode}")
        print(f"  Analysis: {config.use_analysis}")
        print(f"  Planner: {config.planner_name}")
        print(f"  Log Level: {config.log_level}")
        print(f"{'='*80}\n")
    
    def create_modeler(self, task_name: str) -> PDDLModeler:
        """Create a fresh PDDLModeler instance for a task."""
        # Create structured output client
        client = create_structured_client(
            model=self.config.model_name,
            host=self.config.ollama_host,
            mode=self.config.structured_output_mode,
            max_retries=self.config.max_retries
        )
        
        # Create log file path if enabled
        log_file = None
        if self.config.save_generation_logs:
            task_dir = Path(self.config.results_dir) / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(task_dir / f"{task_name}.generation.log")
        
        # Create modeler
        modeler = PDDLModeler(
            client=client,
            is_htn=self.config.is_htn,
            max_retries=self.config.max_retries,
            use_analysis=self.config.use_analysis,
            log_level=LogLevel[self.config.log_level.upper()],
            interactive=self.config.interactive_mode,
            log_file=log_file,
            track_metrics=self.config.track_metrics,
            think_mode=self.config.think_mode
        )
        
        return modeler
    
    def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a single task and return results."""
        task_name = task['name']
        
        # Determine domain name and extract instance number
        domain = None
        instance = 0
        for d in self.config.domains:
            if task_name.startswith(d):
                domain = d
                # Extract instance number (digits at the end)
                instance_str = task_name[len(d):]
                if instance_str.isdigit():
                    instance = int(instance_str)
                break
        
        print(f"\n{'─'*80}")
        print(f"🔹 TASK: {task_name}")
        print(f"{'─'*80}")
        
        # Initialize result
        result = TaskResult(
            task_name=task_name,
            domain=domain or "unknown",
            instance=instance,
            success=False,
            exit_code=-1
        )
        
        # Setup file paths
        task_dir = Path(self.config.results_dir) / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        extension = ".hddl" if self.config.is_htn else ".pddl"
        result.domain_file = str(task_dir / f"{task_name}.domain{extension}")
        result.problem_file = str(task_dir / f"{task_name}.problem{extension}")
        result.plan_file = str(task_dir / f"{task_name}.plan.txt")
        result.log_file = str(task_dir / f"{task_name}.execution.log")
        result.model_file = str(task_dir / f"{task_name}.model.json")
        
        task_start = time.time()
        
        try:
            # PHASE 1: Generate PDDL/HDDL model
            print(f"  📝 Generating model...")
            gen_start = time.time()
            
            modeler = self.create_modeler(task_name)
            status_msg, exit_code = modeler.generate_full_model(
                task['desc'],
                show_metrics=False  # We'll extract metrics separately
            )
            
            result.generation_time = time.time() - gen_start
            
            if exit_code != 0:
                result.exit_code = 1
                result.error_message = f"Generation failed: {status_msg}"
                self._save_error_log(result, modeler)
                print(f"  ❌ Generation failed: {status_msg}")
                return result
            
            # Extract generation metrics
            self._extract_metrics(result, modeler)
            
            # Validate model
            print(f"  ✓ Model generated ({result.generation_time:.2f}s)")
            print(f"    Types: {result.num_types}, Predicates: {result.num_predicates}, "
                  f"Actions: {result.num_actions}, Objects: {result.num_objects}")
            
            is_valid, validation_msg = modeler.builder.validate()
            if not is_valid:
                result.exit_code = 2
                result.error_message = f"Validation failed: {validation_msg}"
                self._save_error_log(result, modeler)
                print(f"  ❌ Validation failed: {validation_msg}")
                return result
            
            print(f"  ✓ Model validated")
            
            # Save PDDL files
            domain_str, problem_str = modeler.get_pddl(
                domain_name=f"{task_name}_domain",
                problem_name=f"{task_name}_problem"
            )
            
            with open(result.domain_file, 'w') as f:
                f.write(domain_str)
            with open(result.problem_file, 'w') as f:
                f.write(problem_str)
            
            # Save model JSON
            if self.config.save_model_json:
                with open(result.model_file, 'w') as f:
                    json.dump(modeler.to_dict(), f, indent=2)
            
            # PHASE 2: Run planner
            print(f"  🎯 Running planner...")
            plan_start = time.time()
            
            try:
                plan = self.planner.solve(result.domain_file, result.problem_file)
                result.planning_time = time.time() - plan_start
                
                if plan and plan.strip():
                    result.plan_found = True
                    # Plan format is like "[action1(...), action2(...), ...]" or line-separated
                    # Count actions by splitting appropriately
                    plan_str = plan.strip()
                    if plan_str.startswith('[') and plan_str.endswith(']'):
                        # Format: [action1, action2, ...]
                        result.plan_length = plan_str.count(',') + 1 if ',' in plan_str else (1 if plan_str != '[]' else 0)
                    else:
                        # Line-separated format
                        result.plan_length = len([l for l in plan_str.split('\n') if l.strip() and not l.startswith(';')])
                    
                    with open(result.plan_file, 'w') as f:
                        f.write(plan)
                    
                    print(f"  ✓ Plan found ({result.planning_time:.2f}s, {result.plan_length} steps)")
                    
                    # Success!
                    result.success = True
                    result.exit_code = 0
                else:
                    result.exit_code = 4
                    result.error_message = "No plan found"
                    print(f"  ❌ No plan found")
            
            except Exception as e:
                result.exit_code = 3
                result.error_message = f"Planner error: {str(e)}"
                result.planning_time = time.time() - plan_start
                print(f"  ❌ Planner error: {str(e)}")
        
        except Exception as e:
            result.exit_code = 1
            result.error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            print(f"  ❌ Unexpected error: {str(e)}")
        
        finally:
            result.total_time = time.time() - task_start
            
            # Save execution log
            self._save_execution_log(result, task)
            
            print(f"  ⏱️  Total time: {result.total_time:.2f}s")
            print(f"{'─'*80}")
        
        return result
    
    def _extract_metrics(self, result: TaskResult, modeler: PDDLModeler):
        """Extract metrics from the modeler."""
        # Model size metrics
        result.num_types = len(modeler.builder.type_names)
        result.num_predicates = len(modeler.builder.predicate_names)
        result.num_actions = len(modeler.builder.action_names)
        result.num_objects = len(modeler.builder.object_names)
        
        if self.config.is_htn:
            result.num_tasks = len(modeler.builder.task_names)
            result.num_methods = len(modeler.builder.method_names)
        
        if modeler.raw_initial:
            result.num_initial_facts = len(modeler.raw_initial.facts)
        
        if modeler.raw_goal:
            if hasattr(modeler.raw_goal, 'facts'):
                result.num_goal_facts = len(modeler.raw_goal.facts)
            elif hasattr(modeler.raw_goal, 'tasks'):
                result.num_goal_facts = len(modeler.raw_goal.tasks)
        
        # LLM metrics (if tracking enabled)
        if self.config.track_metrics and modeler.metrics.steps:
            metrics_dict = modeler.metrics.to_dict()
            result.total_llm_calls = metrics_dict.get('total_steps', 0)
            result.successful_llm_calls = metrics_dict.get('successful_steps', 0)
            result.failed_llm_calls = metrics_dict.get('failed_steps', 0)
            
            if result.total_llm_calls > 0:
                total_attempts = sum(step.get('attempts', 0) for step in metrics_dict.get('steps', []))
                result.avg_attempts_per_step = total_attempts / result.total_llm_calls
    
    def _save_execution_log(self, result: TaskResult, task: Dict[str, Any]):
        """Save execution log."""
        if not self.config.save_execution_log:
            return
        
        with open(result.log_file, 'w') as f:
            f.write(f"Task Execution Log\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Task: {result.task_name}\n")
            f.write(f"Domain: {result.domain}\n")
            f.write(f"Instance: {result.instance}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"\nExit Code: {result.exit_code}\n")
            f.write(f"Success: {result.success}\n")
            
            if result.error_message:
                f.write(f"\nError Message:\n{result.error_message}\n")
            
            f.write(f"\nTiming:\n")
            f.write(f"  Total: {result.total_time:.2f}s\n")
            f.write(f"  Generation: {result.generation_time:.2f}s\n")
            f.write(f"  Planning: {result.planning_time:.2f}s\n")
            
            f.write(f"\nModel Statistics:\n")
            f.write(f"  Types: {result.num_types}\n")
            f.write(f"  Predicates: {result.num_predicates}\n")
            f.write(f"  Actions: {result.num_actions}\n")
            f.write(f"  Objects: {result.num_objects}\n")
            
            if self.config.is_htn:
                f.write(f"  Tasks: {result.num_tasks}\n")
                f.write(f"  Methods: {result.num_methods}\n")
            
            f.write(f"  Initial Facts: {result.num_initial_facts}\n")
            f.write(f"  Goal Facts: {result.num_goal_facts}\n")
            
            if result.plan_found:
                f.write(f"\nPlan:\n")
                f.write(f"  Length: {result.plan_length}\n")
                f.write(f"  File: {result.plan_file}\n")
            
            if 'ground_truth' in task:
                f.write(f"\nGround Truth Plan:\n{task['ground_truth']}\n")
    
    def _save_error_log(self, result: TaskResult, modeler: Optional[PDDLModeler] = None):
        """Save error log with details."""
        with open(result.log_file, 'w') as f:
            f.write(f"Error Log\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Task: {result.task_name}\n")
            f.write(f"Domain: {result.domain}\n")
            f.write(f"Instance: {result.instance}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"\nExit Code: {result.exit_code}\n")
            f.write(f"Error: {result.error_message}\n")
            
            if modeler:
                # Save current model state for debugging
                f.write(f"\n{'='*80}\n")
                f.write(f"MODEL STATE AT FAILURE:\n")
                f.write(f"{'='*80}\n")
                f.write(f"Types: {list(modeler.builder.type_names)}\n")
                f.write(f"Predicates: {list(modeler.builder.predicate_names)}\n")
                f.write(f"Actions: {list(modeler.builder.action_names)}\n")
                f.write(f"Objects: {list(modeler.builder.object_names)}\n")
                
                if self.config.is_htn:
                    f.write(f"Tasks: {list(modeler.builder.task_names)}\n")
                    f.write(f"Methods: {list(modeler.builder.method_names)}\n")
                
                # Save metrics if available
                if self.config.track_metrics:
                    metrics_summary = modeler.get_metrics_summary()
                    if metrics_summary:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"GENERATION METRICS:\n")
                        f.write(f"{'='*80}\n")
                        f.write(metrics_summary)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiments(config: ExperimentConfig):
    """Run the full experiment suite."""
    sep = "=" * config.console_width
    
    print(f"\n{sep}")
    print(f"PDDL MODELER EXPERIMENTATION FRAMEWORK")
    print(f"{sep}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results Directory: {config.results_dir}")
    print(f"{sep}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = PlanBenchDataset()
    print(f"✓ Dataset loaded: {len(dataset.data_dict)} tasks available\n")
    
    # Build task list from configuration
    # Format: domain + instance_num (e.g., "blocksworld10", "depots20")
    selected_tasks = []
    for instance_num in range(config.instance_range_start, config.instance_range_end, config.instance_step):
        for domain in config.domains:
            domain_count = 0
            # Respect max_tasks_per_domain if set
            if config.max_tasks_per_domain and domain_count >= config.max_tasks_per_domain:
                break
            
            task_name = f"{domain}{instance_num}"
            if task_name in dataset.data_dict:
                selected_tasks.append(dataset.data_dict[task_name])
                domain_count += 1
            else:
                if config.show_progress:
                    print(f"  ⚠️  Task '{task_name}' not found in dataset")
    
    if not selected_tasks:
        print("❌ No tasks selected! Check your domain names and instance range.")
        return
    
    print(f"Selected {len(selected_tasks)} tasks across {len(config.domains)} domains\n")
    
    # Initialize tracker and executor
    tracker = ResultsTracker(config)
    executor = TaskExecutor(config, dataset)
    
    # Track skipped tasks separately
    skipped_count = 0
    
    # Process tasks
    for i, task in enumerate(selected_tasks, 1):
        if config.show_progress:
            print(f"\n[{i}/{len(selected_tasks)}] Processing: {task['name']}")
        
        # Check if already completed (for resumable execution)
        task_dir = Path(config.results_dir) / task['name']
        execution_log = task_dir / f"{task['name']}.execution.log"
        
        if config.resumable_execution and execution_log.exists() and not config.force_rerun:
            if config.show_progress:
                print(f"  ⏭️  Skipping (already completed)")
            skipped_count += 1
            
            # Try to load previous result for accurate statistics
            try:
                result = _load_previous_result(task, task_dir, config)
                if result:
                    tracker.add_result(result)
            except Exception as e:
                if config.show_progress:
                    print(f"  ⚠️  Could not load previous result: {e}")
            continue
        
        # Execute task
        result = executor.execute_task(task)
        tracker.add_result(result)
        
        # Save intermediate results (in case of interruption)
        if config.save_intermediate:
            if config.save_metrics_csv:
                csv_path = Path(config.results_dir) / "results_detailed.csv"
                tracker.save_to_csv(str(csv_path))
            
            if config.save_summary_json:
                summary_path = Path(config.results_dir) / "results_summary.json"
                tracker.save_summary(str(summary_path))
        
        # Print progress
        if config.show_progress and i % config.progress_interval == 0:
            summary = tracker.get_summary()
            print(f"\n  📊 Progress: {summary['successful_tasks']}/{summary['total_tasks']} successful "
                  f"({summary['success_rate']:.1f}%)")
    
    # Final summary
    print(f"\n{sep}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{sep}")
    
    if skipped_count > 0 and config.show_progress:
        print(f"\n⏭️  Skipped {skipped_count} previously completed tasks")
    
    summary = tracker.get_summary()
    if summary['total_tasks'] == 0:
        print("\n⚠️  No tasks were executed (all were skipped or none selected)")
        return
    
    if config.show_progress:
        print(f"\nOverall Results:")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Successful: {summary['successful_tasks']} ({summary['success_rate']:.1f}%)")
        print(f"  Failed: {summary['failed_tasks']}")
        
        if config.collect_error_stats:
            print(f"    - Generation Errors: {summary['generation_errors']}")
            print(f"    - Validation Errors: {summary['validation_errors']}")
            print(f"    - Planner Errors: {summary['planner_errors']}")
            print(f"    - No Plan Found: {summary['no_plan_found']}")
        
        if config.show_timing:
            print(f"\nTiming:")
            print(f"  Total Elapsed: {summary['total_elapsed_time']:.1f}s ({summary['total_elapsed_time']/60:.1f}m)")
            print(f"  Avg per Task: {summary['avg_total_time']:.1f}s")
            print(f"  Avg Generation: {summary['avg_generation_time']:.1f}s")
            print(f"  Avg Planning: {summary['avg_planning_time']:.1f}s")
        
        if config.show_metrics_detail:
            print(f"\nModel Statistics (averages):")
            print(f"  Types: {summary['avg_types']:.1f}")
            print(f"  Predicates: {summary['avg_predicates']:.1f}")
            print(f"  Actions: {summary['avg_actions']:.1f}")
            print(f"  Objects: {summary['avg_objects']:.1f}")
        
        # Domain breakdown
        print(f"\n{'─'*config.console_width}")
        print(f"Results by Domain:")
        print(f"{'─'*config.console_width}")
        print(f"{'Domain':<35} {'Total':<8} {'Success':<10} {'Rate':<10}")
        print(f"{'─'*config.console_width}")
        
        domain_summary = tracker.get_domain_summary()
        for domain, stats in sorted(domain_summary.items()):
            print(f"{domain:<35} {stats['total']:<8} {stats['successful']:<10} {stats['success_rate']:>6.1f}%")
        
        print(f"{sep}\n")
    
    # Save final results
    if config.save_metrics_csv:
        csv_path = Path(config.results_dir) / "results_detailed.csv"
        tracker.save_to_csv(str(csv_path))
        if config.show_progress:
            print(f"✓ Detailed results saved to: {csv_path}")
    
    if config.save_summary_json:
        summary_path = Path(config.results_dir) / "results_summary.json"
        tracker.save_summary(str(summary_path))
        if config.show_progress:
            print(f"✓ Summary saved to: {summary_path}")
    
    if config.show_progress:
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{sep}\n")


def _load_previous_result(task: Dict[str, Any], task_dir: Path, config: ExperimentConfig) -> Optional[TaskResult]:
    """
    Load a TaskResult from a previously completed task directory.
    Used for resumable execution to maintain accurate statistics.
    """
    task_name = task['name']
    log_path = task_dir / f"{task_name}.execution.log"
    
    if not log_path.exists():
        return None
    
    # Parse the execution log to reconstruct the result
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Determine domain and instance
    domain = None
    instance = 0
    for d in config.domains:
        if task_name.startswith(d):
            domain = d
            instance_str = task_name[len(d):]
            if instance_str.isdigit():
                instance = int(instance_str)
            break
    
    # Parse exit code
    exit_code = -1
    if "Exit Code: 0" in log_content:
        exit_code = 0
    elif "Exit Code: 1" in log_content:
        exit_code = 1
    elif "Exit Code: 2" in log_content:
        exit_code = 2
    elif "Exit Code: 3" in log_content:
        exit_code = 3
    elif "Exit Code: 4" in log_content:
        exit_code = 4
    
    # Check for plan
    plan_path = task_dir / f"{task_name}.plan.txt"
    plan_found = plan_path.exists() and plan_path.stat().st_size > 0
    
    result = TaskResult(
        task_name=task_name,
        domain=domain or "unknown",
        instance=instance,
        success=(exit_code == 0),
        exit_code=exit_code,
        plan_found=plan_found
    )
    
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# 
# This script uses a hierarchical configuration system:
# 1. Global defaults are defined at the top as DEFAULT_* constants
# 2. ExperimentConfig dataclass uses these defaults
# 3. Override any default by passing parameters to ExperimentConfig()
#
# Examples:
#
#   # Use all defaults
#   config = ExperimentConfig()
#   run_experiments(config)
#
#   # Override specific model settings
#   config = ExperimentConfig(
#       model_name="qwen2.5",
#       structured_output_mode="instructor"
#   )
#
#   # Test mode: small dataset
#   config = ExperimentConfig(
#       domains=["blocksworld"],
#       max_tasks_per_domain=2,
#       instance_range_start=10,
#       instance_range_end=31,
#       instance_step=10
#   )
#
#   # HTN experiments with custom planner settings
#   config = ExperimentConfig(
#       is_htn=True,
#       planner_name="aries",
#       planner_timeout=120,
#       think_mode=True,
#       save_generation_logs=True,
#       track_metrics=True
#   )
#
#   # Production: all features enabled, resumable
#   config = ExperimentConfig(
#       resumable_execution=True,
#       force_rerun=False,
#       save_intermediate=True,
#       collect_error_stats=True,
#       show_metrics_detail=True
#   )
#
# To change global defaults for ALL experiments, modify the DEFAULT_* constants
# at the top of this file before creating any ExperimentConfig.

import argparse
import pandas as pd

def visualize_results(results_dir, show_summary=False, show_domain=False, show_details=False, show_errors=False, filter_domain=None, filter_success=None, console_width=80, save=None):
    """
    Visualiza resultados de experimentos guardados en results_dir.
    - show_summary: muestra el resumen general
    - show_domain: muestra resumen por dominio
    - show_details: muestra detalles de cada tarea
    - filter_domain: si se especifica, solo muestra ese dominio
    - filter_success: None (todos), True (solo exitosos), False (solo fallidos)
    """        
    
    import re
    import os
    from pathlib import Path
    import io
    output_buffer = io.StringIO()

    # --- Resolver results_dir al principio ---
    tried_dirs = []
    if not results_dir or not Path(results_dir).exists():
        print(f"[visualize_results] El directorio {results_dir} no existe. Intentando con sufijos _htn/_pddl...")
        default_dir = DEFAULT_RESULTS_DIR
        use_suffix = DEFAULT_RESULTS_DIR_SUFFIX_MODE
        found = False
        if use_suffix:
            # Elegir sufijo según DEFAULT_IS_HTN
            if DEFAULT_IS_HTN:
                htn_path = Path(default_dir + "_htn")
                tried_dirs.append(str(htn_path))
                if htn_path.exists():
                    print(f"[visualize_results] Usando directorio de resultados HTN: {htn_path}")
                    results_dir = str(htn_path)
                    found = True
                else:
                    print(f"[visualize_results] ADVERTENCIA: No existe el directorio HTN: {htn_path}. Intentando con _pddl...")
                    pddl_path = Path(default_dir + "_pddl")
                    tried_dirs.append(str(pddl_path))
                    if pddl_path.exists():
                        print(f"[visualize_results] Usando directorio de resultados PDDL: {pddl_path}")
                        results_dir = str(pddl_path)
                        found = True
                    else:
                        print(f"[visualize_results] ADVERTENCIA: Tampoco existe el directorio PDDL: {pddl_path}.")
            else:
                pddl_path = Path(default_dir + "_pddl")
                tried_dirs.append(str(pddl_path))
                if pddl_path.exists():
                    print(f"[visualize_results] Usando directorio de resultados PDDL: {pddl_path}")
                    results_dir = str(pddl_path)
                    found = True
                else:
                    print(f"[visualize_results] ADVERTENCIA: No existe el directorio PDDL: {pddl_path}. Intentando con _htn...")
                    htn_path = Path(default_dir + "_htn")
                    tried_dirs.append(str(htn_path))
                    if htn_path.exists():
                        print(f"[visualize_results] Usando directorio de resultados HTN: {htn_path}")
                        results_dir = str(htn_path)
                        found = True
                    else:
                        print(f"[visualize_results] ADVERTENCIA: Tampoco existe el directorio HTN: {htn_path}.")
        if not found:
            # Intentar el directorio base sin sufijo
            base_path = Path(default_dir)
            tried_dirs.append(str(base_path))
            if base_path.exists():
                print(f"[visualize_results] ADVERTENCIA: Usando directorio base sin sufijo: {base_path}")
                results_dir = str(base_path)
                found = True
            else:
                print(f"[visualize_results] ADVERTENCIA: Tampoco existe el directorio base sin sufijo: {base_path}.")
        if not found:
            print(f"[visualize_results] ERROR: No se encontró ningún directorio de resultados válido. Intentados: {tried_dirs}. Abortando visualización.")
            return

    # Resolver save_path
    save_path = None
    if save is not None:
        if isinstance(save, str) and save != '':
            # --save PATH
            save_path = save
        else:
            # --save (sin argumento): generar path sintético
            suffix = "visualize_"
            if show_errors:
                suffix += "errors.log"
            elif show_details:
                suffix += "details.log"
            elif show_domain:
                suffix += "domain.log"
            elif show_summary:
                suffix += "summary.log"
            else:
                suffix += "output.log"
            save_path = str(Path(results_dir) / suffix)


    # Redefinir print para capturar la salida si se va a guardar
    def print_out(*args, **kwargs):
        print(*args, **kwargs, file=output_buffer)
        print(*args, **kwargs)

    if save_path is not None:
        _print = print_out
    else:
        _print = print

    from datetime import timedelta
    sep = "=" * console_width
    summary_path = Path(results_dir) / "results_summary.json"
    csv_path = Path(results_dir) / "results_detailed.csv"

    
    def extract_error_blocks(log_text):
        """
        Extrae bloques de error de un execution log.
        Devuelve dict con info de error_message y traceback (si existen).
        Para traceback, incluye:
            - exception_from_raise: tipo y bloque tras el último raise ExceptionName(
            - summary_from_raise: ExceptionName: mensaje correspondiente
            - last_exception: última ExceptionName: mensaje del traceback
        """

        def classify_traceback_subgroup(msg):
            """
            Clasifica el mensaje de error en un subgrupo temático para traceback.
            """
            if not msg:
                return "other"
            msg = msg.lower()
            # Tipos de subgrupos temáticos
            if "does not exist for object" in msg or "does not exist for predicate" in msg or "does not exist for action" in msg:
                return "missing_type_or_predicate"
            if "does not exist in action" in msg:
                return "missing_predicate_in_action"
            if "is not defined in action" in msg:
                return "undefined_parameter_in_action"
            if "missing types that were identified" in msg:
                return "missing_type_from_analysis"
            if "cannot create type" in msg:
                return "type_creation_error"
            if "expected action name" in msg:
                return "expected_action_name"
            if "pydantic validation failed" in msg:
                return "pydantic_validation"
            if "is not well-formed" in msg:
                return "malformed_expression"
            if "has arity" in msg:
                return "wrong_arity"
            if "failed to generate valid" in msg:
                return "llm_generation_failure"
            if "not exist for" in msg:
                return "missing_type_or_predicate"
            if "not exist in action" in msg:
                return "missing_predicate_in_action"
            if "not defined in action" in msg:
                return "undefined_parameter_in_action"
            if "does not exist" in msg:
                return "missing_type_or_predicate"
            if "is not defined" in msg:
                return "undefined_parameter_in_action"
            if "validation error" in msg:
                return "pydantic_validation"
            if "parent type does not exist" in msg:
                return "type_creation_error"
            if "is not well-formed" in msg:
                return "malformed_expression"
            if "after 3 attempts" in msg:
                return "llm_generation_failure"
            return "other"

        def classify_llm_error_subgroup(msg):
            """
            Clasifica los errores de generación LLM por tipo de elemento (action, method, type, object, predicate, etc).
            """
            if not msg:
                return "others"
            prefix = "Error: Failed to generate valid "
            suffix = " after 3 attempts:"
            if msg.startswith(prefix) and msg.endswith(suffix):
                middle = msg[len(prefix):-len(suffix)].strip()
                # Puede ser "action pick_up" o solo "type"
                element_type = middle.split()[0].lower() if middle else "others"
                if element_type in {"action", "method", "types", "objects", "predicates", "type", "object", "predicate", "initial", "goal"}:
                    return element_type
                else:
                    return "others"
            return "others"

        result = {}
        # --- Error Message ---
        m = re.search(r"Error Message:\n(.*?)(?:\n\w|\n\n|\Z)", log_text, re.DOTALL)
        if m:
            error_msg_block = m.group(1).strip()
            lines = error_msg_block.splitlines()
            for line in lines:
                if ':' in line:
                    type_part, msg_part = line.split(':', 1)
                    type_part = type_part.strip()
                    msg_part = msg_part.strip()
                    # Usar el nuevo filtro para errores LLM
                    if type_part.lower() in {"error", "unknown"}:
                        subgroup = classify_llm_error_subgroup(msg_part)
                        result['error_message'] = {
                            'type': lines[0].split(':', 1)[0].strip() if ':' in lines[0] else 'Generation failed',
                            'message': msg_part,
                            'block': error_msg_block,
                            'subgroup': subgroup
                        }
                    else:
                        subgroup = classify_llm_error_subgroup(msg_part)
                        result['error_message'] = {
                            'type': type_part,
                            'message': msg_part,
                            'block': error_msg_block,
                            'subgroup': subgroup
                        }
                    break
            else:
                for line in lines:
                    if line.strip():
                        subgroup = classify_llm_error_subgroup(line.strip())
                        result['error_message'] = {
                            'type': 'Unknown',
                            'message': line.strip(),
                            'block': error_msg_block,
                            'subgroup': subgroup
                        }
                        _print(f"[extract_error_blocks] Error Message sin tipo detectado: {line.strip()}")
                        break
        else:
            _print("[extract_error_blocks] No se encontró Error Message en el log.")


        # --- Traceback ---
        traceback_pattern = r"Traceback:(.*?)(?:\n\n\n|\Z)"
        matches = list(re.finditer(traceback_pattern, log_text, re.DOTALL))
        if matches:
            tb_match = matches[-1]
        else:
            tb_match = None
            _print("[extract_error_blocks] No se encontró Traceback en el log.")
        if tb_match:
            tb_block = tb_match.group(1).strip()
            if not tb_block:
                _print(f"[extract_error_blocks][ADVERTENCIA] El bloque de traceback está vacío.")
            tb_lines = tb_block.splitlines()
            raise_idx = None
            raise_name = None
            for i in range(len(tb_lines)-1, -1, -1):
                m = re.match(r"^\s*raise ([A-Za-z_][A-Za-z0-9_\.]*)\(", tb_lines[i])
                if m:
                    raise_idx = i
                    raise_name = m.group(1)
                    break
            exception_block = None
            summary_from_raise = None
            type_found = None
            msg_found = None
            subgroup_tb = None
            if raise_idx is not None:
                block_lines = []
                for j in range(raise_idx, len(tb_lines)):
                    block_lines.append(tb_lines[j])
                    if tb_lines[j].strip() == '' and j > raise_idx:
                        break
                exception_block = '\n'.join(block_lines).strip()
                for line in tb_lines:
                    m = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\s*:\s*(.*)", line)
                    if m and raise_name in m.group(1):
                        type_found = m.group(1)
                        msg_found = m.group(2)
                        subgroup_tb = classify_traceback_subgroup(msg_found)
                        break
                if type_found:
                    summary_from_raise = msg_found
            last_exc_type = None
            last_exc_msg = None
            subgroup_last = None
            for line in reversed(tb_lines):
                match = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\s*:\s*(.*)", line.strip())
                if match:
                    last_exc_type = match.group(1)
                    last_exc_msg = match.group(2)
                    subgroup_last = classify_traceback_subgroup(last_exc_msg)
                    break
            result['traceback'] = {
                'exception_from_raise': {
                    'type': type_found if type_found else None,
                    'block': exception_block if exception_block else None,
                    'subgroup': subgroup_tb if subgroup_tb else None
                } if exception_block else None,
                'summary_from_raise': {
                    'type': type_found if type_found else None,
                    'message': summary_from_raise if summary_from_raise else None,
                    'subgroup': subgroup_tb if subgroup_tb else None
                } if type_found and summary_from_raise else None,
                'last_exception': {
                    'type': last_exc_type if last_exc_type else None,
                    'message': last_exc_msg if last_exc_msg else None,
                    'subgroup': subgroup_last if subgroup_last else None
                } if last_exc_type else None,
                'block': tb_block
            }
            if not type_found and exception_block:
                result['traceback']['unknown_block'] = exception_block
        else:
            print("[extract_error_blocks] No se encontró Traceback en el log.")
        return result

    def summarize_errors(results_dir, console_width=80, filter_domain=None):
        from pathlib import Path
        sep = '=' * console_width
        # --- Inicialización de estructuras ---
        error_summary = {'traceback': {}, 'error_message': {}}
        file_info = {}
        msg_count = {}
        em_msg_count = {}
        type_count = {}
        subgroup_count = {}
        domain_count = {}
        em_type_count = {}
        em_subgroup_count = {}
        em_domain_count = {}
        all_types = set()
        em_types = set()
        unknown_blocks = []
        total_files = 0

        _print(f"[summarize_errors] Buscando execution logs en {results_dir}")
        for log_path in Path(results_dir).rglob('*.execution.log'):
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    log_text = f.read()
            except Exception as e:
                _print(f"[summarize_errors] ERROR al leer {log_path}: {e}")
                continue
            total_files += 1
            # Extraer dominio y tarea
            domain = None
            task = None
            m = re.search(r"^Domain: (.+)$", log_text, re.MULTILINE)
            if m:
                domain = m.group(1)
            m = re.search(r"^Task: (.+)$", log_text, re.MULTILINE)
            if m:
                task = m.group(1)
            info = extract_error_blocks(log_text)
            file_info[str(log_path)] = info
            # --- Error Message ---
            em = info.get('error_message')
            if em:
                etype = em.get('type') or 'UnknownError'
                esub = em.get('subgroup') or 'other'
                emsg = em.get('message') or ''
                error_summary['error_message'].setdefault(etype, {})
                error_summary['error_message'][etype].setdefault(esub, {})
                error_summary['error_message'][etype][esub].setdefault(emsg, {})
                error_summary['error_message'][etype][esub][emsg].setdefault(domain or 'unknown', []).append(task or str(log_path))
                em_types.add(etype)
                em_type_count[etype] = em_type_count.get(etype, 0) + 1
                em_subgroup_count.setdefault(etype, {})
                em_subgroup_count[etype][esub] = em_subgroup_count[etype].get(esub, 0) + 1
                em_msg_count[emsg] = em_msg_count.get(emsg, 0) + 1
                if domain:
                    em_domain_count.setdefault((etype, esub), set()).add(domain)
            # --- Traceback: summary_from_raise ---
            tb = info.get('traceback')
            if tb:
                if tb.get('summary_from_raise'):
                    etype = tb['summary_from_raise'].get('type') or 'UnknownError'
                    esub = tb['summary_from_raise'].get('subgroup') or 'other'
                    emsg = tb['summary_from_raise'].get('message') or ''
                    error_summary['traceback'].setdefault(etype, {})
                    error_summary['traceback'][etype].setdefault(esub, {})
                    error_summary['traceback'][etype][esub].setdefault(emsg, {})
                    error_summary['traceback'][etype][esub][emsg].setdefault(domain or 'unknown', []).append(task or str(log_path))
                    all_types.add(etype)
                    type_count[etype] = type_count.get(etype, 0) + 1
                    subgroup_count.setdefault(etype, {})
                    subgroup_count[etype][esub] = subgroup_count[etype].get(esub, 0) + 1
                    msg_count[emsg] = msg_count.get(emsg, 0) + 1
                    if domain:
                        domain_count.setdefault((etype, esub), set()).add(domain)
                if 'unknown_block' in tb:
                    unknown_blocks.append((log_path, domain))

        # 1. Listas detalladas
        _print(f"\n{sep}\nLISTA DE ERRORES POR TRACEBACK (summary_from_raise)\n{sep}")
        for etype, subdict in sorted(error_summary['traceback'].items(), key=lambda x: -sum(len(t) for sd in x[1].values() for md in sd.values() for t in md.values())):
            _print(f"\n{etype}:")
            for esub, msgdict in sorted(subdict.items(), key=lambda x: -sum(len(t) for md in x[1].values() for t in md.values())):
                _print(f"  > Subgrupo: {esub}")
                for emsg, domdict in sorted(msgdict.items(), key=lambda x: -sum(len(t) for t in x[1].values())):
                    _print(f"    - {emsg}")
                    for dom, tasks in sorted(domdict.items(), key=lambda x: -len(x[1])):
                        if filter_domain and dom != filter_domain:
                            continue
                        _print(f"      [{dom}] {len(tasks)} ocurrencia(s): {', '.join(tasks[:5])}{' ...' if len(tasks)>5 else ''}")

        _print(f"\n{sep}\nLISTA DE ERRORES POR ERROR MESSAGE\n{sep}")
        for etype, subdict in sorted(error_summary['error_message'].items(), key=lambda x: -sum(len(t) for sd in x[1].values() for md in sd.values() for t in md.values())):
            _print(f"\n{etype}:")
            for esub, msgdict in sorted(subdict.items(), key=lambda x: -sum(len(t) for md in x[1].values() for t in md.values())):
                _print(f"  > Subgrupo: {esub}")
                for emsg, domdict in sorted(msgdict.items(), key=lambda x: -sum(len(t) for t in x[1].values())):
                    _print(f"    - {emsg}")
                    for dom, tasks in sorted(domdict.items(), key=lambda x: -len(x[1])):
                        if filter_domain and dom != filter_domain:
                            continue
                        _print(f"      [{dom}] {len(tasks)} ocurrencia(s): {', '.join(tasks[:5])}{' ...' if len(tasks)>5 else ''}")

        # 2. Distribuciones de mensajes de error
        _print(f"\nDistribución de mensajes de error (Traceback):")
        for m_, v in sorted(msg_count.items(), key=lambda x: -x[1]):
            _print(f"  - '{m_}': {v} ocurrencias")
        _print(f"\nDistribución de mensajes de error (Error Message):")
        for m_, v in sorted(em_msg_count.items(), key=lambda x: -x[1]):
            _print(f"  - '{m_}': {v} ocurrencias")

        # 3. Distribuciones de errores por tipo y subgrupo
        _print(f"\nDistribución de errores por tipo y subgrupo (Traceback):")
        for t, _ in sorted(type_count.items(), key=lambda x: -x[1]):
            _print(f"  - {t}: {type_count[t]} ocurrencias")
            if t in subgroup_count:
                for sub, _ in sorted(subgroup_count[t].items(), key=lambda x: -x[1]):
                    doms = ', '.join(sorted(domain_count.get((t, sub), [])))
                    _print(f"      > {sub}: {subgroup_count[t][sub]} ocurrencias en dominios: {doms}")
        _print(f"\nDistribución de errores por tipo y subgrupo (Error Message):")
        for t, _ in sorted(em_type_count.items(), key=lambda x: -x[1]):
            _print(f"  - {t}: {em_type_count[t]} ocurrencias")
            if t in em_subgroup_count:
                for sub, _ in sorted(em_subgroup_count[t].items(), key=lambda x: -x[1]):
                    doms = ', '.join(sorted(em_domain_count.get((t, sub), [])))
                    _print(f"      > {sub}: {em_subgroup_count[t][sub]} ocurrencias en dominios: {doms}")

        # 4. Tablas resumen y resumen final
        _print(f"\n{sep}\nTABLA DE ERRORES POR TIPO, SUBGRUPO Y DOMINIO (Traceback)\n{sep}")
        _print(f"{'Tipo de Error':<30} {'Subgrupo':<30} {'#Ocurrencias':<12} {'   Dominios':<30}")
        _print('-'*100)
        def short_type_name(t):
            return t.split('.')[-1] if '.' in t else t
        for t, _ in sorted(type_count.items(), key=lambda x: -x[1]):
            t_short = short_type_name(t)
            if t in subgroup_count:
                for sub, _ in sorted(subgroup_count[t].items(), key=lambda x: -x[1]):
                    doms = ', '.join(sorted(domain_count.get((t, sub), [])))
                    _print(f"{t_short:<30} {sub:<30} {subgroup_count[t][sub]:<12}    {doms:<30}")
        _print(f"\n{sep}\nTABLA DE ERRORES POR TIPO, SUBGRUPO Y DOMINIO (Error Message)\n{sep}")
        _print(f"{'Tipo de Error':<30} {'Subgrupo':<30} {'#Ocurrencias':<12} {'   Dominios':<30}")
        _print('-'*100)
        for t, _ in sorted(em_type_count.items(), key=lambda x: -x[1]):
            if t in em_subgroup_count:
                for sub, _ in sorted(em_subgroup_count[t].items(), key=lambda x: -x[1]):
                    doms = ', '.join(sorted(em_domain_count.get((t, sub), [])))
                    _print(f"{t:<30} {sub:<30} {em_subgroup_count[t][sub]:<12}    {doms:<30}")
        _print('-'*100)
        _print()
        
        # ─────────────────────────────────────────────────────────────
        # CRUCE DE ERRORES ENTRE TRACEBACK Y ERROR MESSAGE (por subgrupo)
        # ─────────────────────────────────────────────────────────────
        _print(f"{sep}\nCRUCE DE ERRORES ENTRE TRACEBACK Y ERROR MESSAGE (por subgrupo)\n{sep}")
        # 1. Construir matrices de coocurrencia por subgrupo
        #   - Para cada archivo, si tiene ambos errores, anotar subgrupo de cada uno
        #   - Contar coocurrencias y guardar ejemplos
        cooc_matrix = {}  # (tb_sub, em_sub) -> count
        cooc_examples = {}  # (tb_sub, em_sub) -> [(task, domain, log_path)]
        tb_only = {}  # tb_sub -> count
        em_only = {}  # em_sub -> count
        for log_path, info in file_info.items():
            tb_sub = None
            em_sub = None
            domain = None
            task = None
            # Extraer dominio y tarea del path o info
            m_dom = re.search(r"Domain: (.+)", log_path)
            m_task = re.search(r"Task: (.+)", log_path)
            domain = m_dom.group(1) if m_dom else 'unknown'
            task = m_task.group(1) if m_task else log_path
            # Subgrupo Traceback
            tb = info.get('traceback')
            if tb and tb.get('summary_from_raise'):
                tb_sub = tb['summary_from_raise'].get('subgroup') or 'other'
            # Subgrupo Error Message
            em = info.get('error_message')
            if em:
                em_sub = em.get('subgroup') or 'other'
            # Cruce
            if tb_sub and em_sub:
                key = (tb_sub, em_sub)
                cooc_matrix[key] = cooc_matrix.get(key, 0) + 1
                cooc_examples.setdefault(key, []).append((task, domain, log_path))
            elif tb_sub:
                tb_only[tb_sub] = tb_only.get(tb_sub, 0) + 1
            elif em_sub:
                em_only[em_sub] = em_only.get(em_sub, 0) + 1

        # 2. Listar todos los subgrupos presentes
        tb_subgroups = set()
        em_subgroups = set()
        for tb_sub, _ in tb_only.items():
            tb_subgroups.add(tb_sub)
        for em_sub, _ in em_only.items():
            em_subgroups.add(em_sub)
        for tb_sub, em_sub in cooc_matrix.keys():
            tb_subgroups.add(tb_sub)
            em_subgroups.add(em_sub)
        tb_subgroups = sorted(tb_subgroups)
        em_subgroups = sorted(em_subgroups)


        # 3. Imprimir heatmap de coocurrencias con sumatorias
        _print(f"\nHEATMAP DE COOCCURRENCIAS (subgrupo Traceback x subgrupo Error Message):\n")
        # Calcular sumatorias por fila y columna
        row_sums = {tb_sub: sum(cooc_matrix.get((tb_sub, em_sub), 0) for em_sub in em_subgroups) for tb_sub in tb_subgroups}
        col_sums = {em_sub: sum(cooc_matrix.get((tb_sub, em_sub), 0) for tb_sub in tb_subgroups) for em_sub in em_subgroups}
        total_sum_rows = sum(row_sums.values())
        total_sum_cols = sum(col_sums.values())

        # Ordenar subgrupos por totales
        tb_subgroups_sorted = sorted(tb_subgroups, key=lambda tb: -row_sums.get(tb, 0))
        em_subgroups_sorted = sorted(em_subgroups, key=lambda em: -col_sums.get(em, 0))

        # Header
        header = f"{'':<38}" + ''.join([f"{em:<18}" for em in em_subgroups_sorted]) + f"{'Total TB→':<10}"
        _print(header)
        _print('-' * (38 + 18 * len(em_subgroups_sorted) + 10))
        # Filas
        for tb_sub in tb_subgroups_sorted:
            row = f"{tb_sub:<38}"
            for em_sub in em_subgroups_sorted:
                count = cooc_matrix.get((tb_sub, em_sub), 0)
                row += f"{count:<18}"
            row += f"{row_sums[tb_sub]:<10}"
            _print(row)
        # Fila de sumatorias de columna
        sum_row = f"{'Total EM↓':<38}"
        for em_sub in em_subgroups_sorted:
            sum_row += f"{col_sums[em_sub]:<18}"
        # Mostrar total solo si coinciden
        if total_sum_rows == total_sum_cols:
            sum_row += f"{total_sum_rows:<10}"
        else:
            sum_row += f"{'':<10}"
            print('-' * (38 + 18 * len(em_subgroups_sorted) + 10))
            print(f"[ADVERTENCIA] La suma total por filas ({total_sum_rows}) y por columnas ({total_sum_cols}) NO coincide. Revisar matriz de coocurrencias.")
        _print('-' * (38 + 18 * len(em_subgroups_sorted) + 10))
        _print(sum_row)

        # 4. Imprimir totales de subgrupos solo Traceback y solo Error Message
        _print("\nInstancias SOLO en Traceback (sin Error Message):")
        any_tb = False
        for tb_sub in tb_subgroups:
            count = tb_only.get(tb_sub, 0)
            if count:
                _print(f"  - {tb_sub}: {count} ocurrencia(s)")
                any_tb = True
        if not any_tb:
            _print("  (Ninguna)")

        _print("\nInstancias SOLO en Error Message (sin Traceback):")
        any_em = False
        for em_sub in em_subgroups:
            count = em_only.get(em_sub, 0)
            if count:
                _print(f"  - {em_sub}: {count} ocurrencia(s)")
                any_em = True
        if not any_em:
            _print("  (Ninguna)")


        # --- RESUMEN ESTADÍSTICO DETALLADO ---
        _print(f"\n{sep}\nRESUMEN ESTADÍSTICO DE ERRORES Y EJECUCIONES\n{sep}")

        # Instancias con y sin error
        instancias_con_error = set()
        instancias_sin_error = set()
        dominio_stats = {}
        for log_path, info in file_info.items():
            # Extraer dominio e instancia del contenido del log
            domain = None
            task = None
            # Buscar en el contenido del log
            log_text = None
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    log_text = f.read()
            except Exception:
                pass
            if log_text:
                m_dom = re.search(r"^Domain: (.+)$", log_text, re.MULTILINE)
                m_task = re.search(r"^Task: (.+)$", log_text, re.MULTILINE)
                domain = m_dom.group(1) if m_dom else 'unknown'
                task = m_task.group(1) if m_task else log_path
            else:
                domain = 'unknown'
                task = log_path
            # Consideramos error si hay error_message o traceback
            has_error = False
            if info.get('error_message') or (info.get('traceback') and info['traceback'].get('summary_from_raise')):
                has_error = True
            if has_error:
                instancias_con_error.add(task)
            else:
                instancias_sin_error.add(task)
            # Estadísticas por dominio
            if domain not in dominio_stats:
                dominio_stats[domain] = {'total': 0, 'errores': 0, 'correctas': 0}
            dominio_stats[domain]['total'] += 1
            if has_error:
                dominio_stats[domain]['errores'] += 1
            else:
                dominio_stats[domain]['correctas'] += 1

        total_instancias = len(instancias_con_error) + len(instancias_sin_error)
        porcentaje_errores = 100.0 * len(instancias_con_error) / total_instancias if total_instancias else 0.0

        _print(f"Total de archivos analizados: {total_files}")
        _print(f"  - Total de instancias: {total_instancias}")
        _print(f"  - Instancias con error: {len(instancias_con_error)} ({porcentaje_errores:.1f}%)")

        # --- Desglose por tipos de error con porcentajes ---
        total_tb = sum(type_count.values())
        _print(f"\nErrores con Traceback: {total_tb}")
        for t, count in sorted(type_count.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total_tb if total_tb else 0.0
            _print(f"  - {t}: {count} ({pct:.1f}%)")

        total_em = sum(em_type_count.values())
        _print(f"\nErrores con Error Message: {total_em}")
        for t, count in sorted(em_type_count.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total_em if total_em else 0.0
            _print(f"  - {t}: {count} ({pct:.1f}%)")

        # Estadísticas por dominio
        _print(f"\n\nEstadísticas por dominio:")
        _print(f"{'Dominio':<32} {'Total':<8} {'Errores':<8} {'Correctas':<10} {'%Correctas':<10} {'%Errores':<10}")
        _print('-'*console_width)
        for dom, stats in sorted(dominio_stats.items(), key=lambda x: -x[1]['total']):
            total = stats['total']
            err = stats['errores']
            ok = stats['correctas']
            p_ok = 100.0 * ok / total if total else 0.0
            p_err = 100.0 * err / total if total else 0.0
            _print(f"{dom:<32} {total:<8} {err:<8} {ok:<10} {p_ok:>8.1f}% {p_err:>8.1f}%")
        _print('-'*console_width)

        # Archivos con bloque desconocido
        if unknown_blocks:
            _print(f"\nArchivos con unknown_block:")
            for log_path, domain in unknown_blocks:
                _print(f"  - {log_path} (Dominio: {domain})")
        _print(f"\n{sep}\n")

    def _show_summary():
        if not summary_path.exists():
            _print(f"[visualize_results] No se encontró el archivo de resumen: {summary_path}")
        else:
            try:
                with open(summary_path) as f:
                    content = f.read()
                    if not content.strip():
                        _print(f"[visualize_results] El archivo {summary_path} está vacío.")
                    else:
                        summary = json.loads(content)
                        if show_domain:
                            _print(f"\n{sep}\nRESUMEN POR DOMINIO\n{sep}")
                            by_domain = summary.get("by_domain", {})
                            _print(f"{'Domain':<30} {'Total':<8} {'Success':<10} {'Rate':<10}")
                            _print("-"*console_width)
                            for domain, stats in by_domain.items():
                                if filter_domain and domain != filter_domain:
                                    continue
                                _print(f"{domain:<30} {stats['total']:<8} {stats['successful']:<10} {stats['success_rate']:>6.1f}%")
                        if show_summary:
                            _print(f"\n{sep}\nRESUMEN GENERAL\n{sep}")
                            overall = summary.get("overall", {})
                            for k, v in overall.items():
                                if k in ("avg_total_time", "avg_generation_time", "total_elapsed_time", "avg_planning_time"):
                                    try:
                                        _print(f"{k}: {str(timedelta(seconds=float(v)))} ({v:.2f}s)")
                                    except Exception:
                                        _print(f"{k}: {v}")
                                else:
                                    _print(f"{k}: {v}")
            except Exception as e:
                _print(f"[visualize_results] Error leyendo {summary_path}: {e}")

    def _show_details():
        if not csv_path.exists():
            _print(f"[visualize_results] No se encontró el archivo de detalles: {csv_path}")
        else:
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    _print(f"[visualize_results] El archivo {csv_path} está vacío.")
                else:
                    _print(f"\n{sep}\nDETALLES DE TAREAS\n{sep}")
                    if filter_domain:
                        df = df[df['domain'] == filter_domain]
                    if filter_success is not None:
                        df = df[df['success'] == filter_success]
                    for _, row in df.iterrows():
                        _print(f"Tarea: {row['task_name']} | Dominio: {row['domain']} | Instancia: {row['instance']} | Success: {row['success']} | Plan: {row['plan_found']} | Longitud: {row['plan_length']} | Tiempo total: {row['total_time']:.2f}s")
            except Exception as e:
                _print(f"[visualize_results] Error leyendo {csv_path}: {e}")

    if show_summary:
        _show_summary()
    if show_domain:
        _show_summary()  # El resumen por dominio está incluido en el summary.json
    if show_details:
        _show_details()
    if show_errors:
        summarize_errors(results_dir, console_width=console_width, filter_domain=filter_domain)
    
    # Si se pidió guardar, escribir el buffer en el archivo
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(f"\n{'='*console_width}\n")
            f.write(f"VISUALIZACIÓN DE RESULTADOS DE PDDLMODELER\n")
            f.write(f"Nombre del directorio de resultados: {results_dir}\n")
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Información mostrada:\n")
            if show_summary:
                f.write(" - Resumen general\n")
            if show_domain:
                f.write(" - Resumen por dominio\n")
            if show_details:
                f.write(" - Detalles de tareas\n")
            if show_errors:
                f.write(" - Resumen de errores\n")
            f.write(  f"Filtros aplicados:\n")
            if filter_domain:
                f.write(f" - Dominio: {filter_domain}\n")
            if filter_success:
                f.write(f" - Éxito: {filter_success}\n")
            f.write(f"\n{'='*console_width}\n\n")
            f.write(output_buffer.getvalue())
        print(f"\nVisualización guardada en: {save_path}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="PDDLModeler Experimentation & Visualization")
    parser.add_argument('results_dir', type=str, nargs='?', default=DEFAULT_RESULTS_DIR, help='Directorio de resultados a visualizar (posicional, sin flag)')
    parser.add_argument('--visualize', action='store_true', help='Solo visualizar resultados, no ejecutar experimentos')
    parser.add_argument('--show', type=str, nargs='*', default=['summary'], help='Qué mostrar: summary, domain, details, errors')
    parser.add_argument('--filter_domain', type=str, default=None, help='Filtrar por dominio')
    parser.add_argument('--filter_success', type=str, default=None, help='Filtrar por éxito: true/false')
    parser.add_argument('--console_width', type=int, default=DEFAULT_CONSOLE_WIDTH, help='Ancho de consola')
    parser.add_argument('--save', type=str, nargs='*', default=[''], help='Guarda la visualización en un archivo. Si se especifica, usa ese path; si no, genera uno automáticamente en la carpeta de resultados.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.visualize:
        show_summary = 'summary' in args.show
        show_domain = 'domain' in args.show
        show_details = 'details' in args.show
        show_errors  = 'errors' in args.show
        filter_success = None
        if args.filter_success is not None:
            if args.filter_success.lower() in ['true', '1', 'yes']:
                filter_success = True
            elif args.filter_success.lower() in ['false', '0', 'no']:
                filter_success = False

        visualize_results(
            results_dir=args.results_dir,
            show_summary=show_summary,
            show_domain=show_domain,
            show_details=show_details,
            show_errors=show_errors,
            filter_domain=args.filter_domain,
            filter_success=filter_success,
            console_width=args.console_width,
            save=args.save
        )
    else:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTATION SCRIPT FOR PDDLMODELER")
        print(f"{'='*80}")
        print(f"\nTo customize this experiment, modify either:")
        print(f"  1. DEFAULT_* constants at the top of this file (affects all instances)")
        print(f"  2. ExperimentConfig parameters in __main__ (affects this run only)")
        print(f"\n{'='*80}\n")
        config = ExperimentConfig(
            model_name=DEFAULT_OLLAMA_MODEL,
            ollama_host=DEFAULT_OLLAMA_HOST,
            structured_output_mode=DEFAULT_STRUCTURED_OUTPUT_MODE,
            is_htn=DEFAULT_IS_HTN,
            use_analysis=DEFAULT_USE_ANALYSIS,
            think_mode=DEFAULT_THINK_MODE,
            max_retries=DEFAULT_MAX_RETRIES,
            log_level=DEFAULT_LOG_LEVEL,
            save_generation_logs=DEFAULT_SAVE_GENERATION_LOGS,
            save_execution_log=DEFAULT_SAVE_EXECUTION_LOG,
            track_metrics=DEFAULT_TRACK_METRICS,
            domains=DEFAULT_DOMAINS,
            instance_range_start=DEFAULT_INSTANCE_RANGE_START,
            instance_range_end=DEFAULT_INSTANCE_RANGE_END,
            instance_step=DEFAULT_INSTANCE_STEP,
            max_tasks_per_domain=DEFAULT_MAX_TASKS_PER_DOMAIN,
            planner_name=DEFAULT_PLANNER_NAME,
            planner_timeout=DEFAULT_PLANNER_TIMEOUT,
            continue_on_error=DEFAULT_CONTINUE_ON_ERROR,
            resumable_execution=DEFAULT_RESUMABLE_EXECUTION,
            force_rerun=DEFAULT_FORCE_RERUN,
            results_dir=DEFAULT_RESULTS_DIR,
            results_dir_suffix_mode=DEFAULT_RESULTS_DIR_SUFFIX_MODE,
            save_model_json=DEFAULT_SAVE_MODEL_JSON,
            save_domain_pddl=DEFAULT_SAVE_DOMAIN_PDDL,
            save_problem_pddl=DEFAULT_SAVE_PROBLEM_PDDL,
            save_plan=DEFAULT_SAVE_PLAN,
            save_metrics_csv=DEFAULT_SAVE_METRICS_CSV,
            save_summary_json=DEFAULT_SAVE_SUMMARY_JSON,
            save_intermediate=DEFAULT_SAVE_INTERMEDIATE,
            show_progress=DEFAULT_SHOW_PROGRESS,
            show_timing=DEFAULT_SHOW_TIMING,
            show_metrics_detail=DEFAULT_SHOW_METRICS_DETAIL,
            progress_interval=DEFAULT_PROGRESS_INTERVAL,
            console_width=DEFAULT_CONSOLE_WIDTH,
            collect_error_stats=DEFAULT_COLLECT_ERROR_STATS,
        )
        try:
            run_experiments(config)
        except KeyboardInterrupt:
            print("\n\n⚠️  Experiment interrupted by user")
            print("Partial results have been saved.")
            print("Re-run the script to resume from where you left off.")
        except Exception as e:
            print(f"\n\n❌ Fatal error: {e}")
            print(traceback.format_exc())
            sys.exit(1)
