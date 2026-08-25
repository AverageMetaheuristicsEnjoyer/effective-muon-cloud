TASK_GROUPS = {
    "basic_v1": [
        "hellaswag",
        "arc_easy",
    ],
    "basic_v2": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "gsm8k_gold_bpb_5shot",
    ],
}


def _dedupe_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def resolve_downstream_tasks(task_group=None, tasks=None):
    resolved = []

    if task_group:
        if task_group not in TASK_GROUPS:
            available = ", ".join(sorted(TASK_GROUPS))
            raise ValueError(
                f"Unknown downstream task group '{task_group}'. Available groups: {available}"
            )
        resolved.extend(TASK_GROUPS[task_group])

    if tasks:
        resolved.extend(tasks)

    if not resolved:
        resolved.extend(TASK_GROUPS["basic_v1"])

    return _dedupe_preserve_order(resolved)
