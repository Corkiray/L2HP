"""
Paper: "PROC2PDDL: Open-Domain Planning Representations from Texts" Zhang et al. (2024)
Source code: https://github.com/zharry29/proc2pddl
Run: python3 -m paper_reconstructions.proc2pddl.main
"""

import os
from l2p import *

if __name__ == "__main__":

    UNSUPPORTED_KEYWORDS = ["object", "pddl", "lisp"]

    engine = "gpt-4o"
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_llm = OPENAI(model=engine, api_key=api_key)

    domain_builder = DomainBuilder()
    task_builder = TaskBuilder()
    feedback_builder = FeedbackBuilder()
    planner = FastDownward(planner_path="downward/fast-downward.py")

    # annotated original domain header
    types = load_file("paper_reconstructions/proc2pddl/prompts/types.json")
    predicates = load_file("paper_reconstructions/proc2pddl/prompts/predicates.json")

    with open("paper_reconstructions/proc2pddl/prompts/nl_actions.json", "r") as file:
        nl_actions = json.load(file)

    # retrieve wikihow text
    domain_desc = load_file("paper_reconstructions/proc2pddl/prompts/wikihow.txt")

    # ZPD prompt
    role = load_file("paper_reconstructions/proc2pddl/prompts/zpd_prompt/role.txt")
    technique = load_file(
        "paper_reconstructions/proc2pddl/prompts/zpd_prompt/technique.txt"
    )
    example = load_file(
        "paper_reconstructions/proc2pddl/prompts/zpd_prompt/example.txt"
    )
    task = (
        "here are the actions I want:\n"
        + (str(nl_actions))
        + "\n\nhere are the types I have:\n"
        + pretty_print_dict(types)
        + "\n\nhere are the predicates I have:\n"
        + pretty_print_expression(predicates)
    )
    ZPD_prompt = PromptBuilder(
        role=role, technique=technique, examples=[example], task=task
    )

    # (1) query LLM for ZPD information
    action_descriptions = openai_llm.query(prompt=ZPD_prompt.generate_prompt())

    # PDDL extraction prompt
    role = load_file(
        "paper_reconstructions/proc2pddl/prompts/pddl_translate_prompt/role.txt"
    )
    example = load_file(
        "paper_reconstructions/proc2pddl/prompts/pddl_translate_prompt/example.txt"
    )
    task = load_file(
        "paper_reconstructions/proc2pddl/prompts/pddl_translate_prompt/task.txt"
    )
    task += "\n\nhere are the action descriptions to use:\n" + action_descriptions
    pddl_formalize_prompt = PromptBuilder(role=role, examples=[example], task=task)

    action_list = [
        f"{name}: {desc.rstrip('.')}"
        for action in nl_actions
        for name, desc in action.items()
    ]

    # (2) extract PDDL requirements
    actions, _, llm_response = domain_builder.formalize_pddl_actions(
        model=openai_llm,
        domain_desc=domain_desc,
        prompt_template=pddl_formalize_prompt.generate_prompt(),
        action_list=action_list,
        predicates=predicates,
        types=types,
    )

    types = format_types(types)  # retrieve types
    pruned_types = {
        name: description
        for name, description in types.items()
        if name not in UNSUPPORTED_KEYWORDS
    }  # remove unsupported words

    # # format strings
    # predicate_str = "\n".join([pred["clean"] for pred in predicates])
    # types_str = "\n".join(pruned_types)

    requirements = [
        ":strips",
        ":typing",
        ":equality",
        ":negative-preconditions",
        ":disjunctive-preconditions",
        ":universal-preconditions",
        ":conditional-effects",
    ]

    # generate domain
    pddl_domain = domain_builder.generate_domain(
        domain_name="survive_deserted_island",
        requirements=requirements,
        types=pruned_types,
        predicates=predicates,
        actions=actions,
    )

    domain_file = "paper_reconstructions/proc2pddl/results/domain.pddl"
    with open(domain_file, "w") as f:
        f.write(pddl_domain)

    print("PDDL domain:\n", pddl_domain)

    problem_file = load_file(
        "paper_reconstructions/proc2pddl/results/problems/problem-catch_cook_fish.pddl"
    )

    # run planner
    planner.run_fast_downward(domain_file=domain_file, problem_file=problem_file)
