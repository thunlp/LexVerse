from lexverse.utils.identity import identity


def execution_conditions(identity):
    conditions = {key: value for key, value in identity.items() if key not in {"code", "code_commit"}}
    return conditions
