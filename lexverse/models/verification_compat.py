from copy import deepcopy


def rename_keys(data, names):
    result = dict(data)
    for old, new in names.items():
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError(f"Conflicting fields: {old} and {new}")
            result[new] = result.pop(old)
    return result


def convert_config_names(data, *, legacy=False):
    result = deepcopy(data)
    names = {"evaluation": "verification"}
    models = {"evaluators": "verifiers"}
    if legacy:
        names = {v: k for k, v in names.items()}
        models = {v: k for k, v in models.items()}
    result = rename_keys(result, names)
    if isinstance(result.get("models"), dict):
        result["models"] = rename_keys(result["models"], models)
    return result


def convert_task_names(data, *, legacy=False):
    return rename_keys(data, {"verification": "evaluation"} if legacy else
                       {"evaluation": "verification"})


def convert_settings_names(data, *, legacy=False):
    result = rename_keys(data, {"verify": "evaluate"} if legacy else {"evaluate": "verify"})
    if "config" in result:
        result["config"] = convert_config_names(result["config"], legacy=legacy)
    return result


def convert_result_names(data, *, legacy=False):
    names = {"evaluation": "verifier_result", "evaluation_ref": "verification_ref",
             "evaluation_status": "verification_status"}
    return rename_keys(data, {v: k for k, v in names.items()} if legacy else names)
