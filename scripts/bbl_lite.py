from utils.bigbench_utils import load_task

BBL_TASKS = [
    "auto_debugging", "bbq_lite_json", "code_line_description", "conceptual_combinations",
    "conlang_translation", "emoji_movie", "formal_fallacies_syllogisms_negation", "hindu_knowledge",
    "known_unknowns", "language_identification", "linguistics_puzzles", "logic_grid_puzzle",
    "logical_deduction", "misconceptions_russian", "novel_concepts", "operators",
    "parsinlu_reading_comprehension", "play_dialog_same_or_different", "repeat_copy_logic",
    "strange_stories", "strategyqa", "symbol_interpretation", "vitaminc_fact_verification", "winowhy",
]

for task in BBL_TASKS:
    try:
        ex = load_task(task, n=1)[0]
        print(f"{task:40s} n_choices={len(ex['multiple_choice_targets']):2d}")
    except Exception as e:
        print(f"{task:40s} ERROR: {str(e)[:60]}")          