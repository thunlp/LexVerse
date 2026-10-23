import hashlib
import json


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def execution_conditions(identity):
    conditions = {key: value for key, value in identity.items() if key not in {"code", "code_commit"}}
    # Older public runs predate explicit task network settings.
    if conditions.get("network_mode", "public") == "public" and not conditions.get("allowed_hosts"):
        conditions.pop("network_mode", None)
        conditions.pop("allowed_hosts", None)
    return conditions
