from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    name: str                      # short name: "CI", "CR", ...
    alias: str                     # upstream registry alias
    roles: tuple[str, ...]         # order matters: judge-like first
    default_agents: dict[str, str] # role -> upstream agent alias
    default_max_turns: int
    metrics: tuple[str, ...]
    supported: bool = True


SCENARIOS: dict[str, ScenarioSpec] = {
    "CI": ScenarioSpec(
        name="CI",
        alias="J1Bench.Scenario.CI",
        roles=("judge", "plaintiff", "defendant"),
        default_agents={
            "judge": "Agent.Judge.GPT_CI",
            "plaintiff": "Agent.Plaintiff.GPT_CI",
            "defendant": "Agent.Defendant.GPT_CI",
        },
        default_max_turns=50,
        metrics=("REA", "LAW", "JUD", "PFS"),
    ),
    "CR": ScenarioSpec(
        name="CR",
        alias="J1Bench.Scenario.CR",
        roles=("judge", "procurator", "defendant", "lawyer"),
        default_agents={
            "judge": "Agent.Judge.GPT_CR",
            "procurator": "Agent.Procurator.GPT_CR",
            "defendant": "Agent.Defendant.GPT_CR",
            "lawyer": "Agent.Lawyer.GPT_CR",
        },
        default_max_turns=35,
        metrics=("CRI", "SEN", "FINE", "REA", "LAW", "PFS"),
    ),
    "CD": ScenarioSpec(
        name="CD",
        alias="J1Bench.Scenario.CD",
        roles=("lawyer", "specific_character"),
        default_agents={
            "lawyer": "Agent.Lawyer.GPT_CD",
            "specific_character": "Agent.Specific_character.GPT_CD",
        },
        default_max_turns=20,
        metrics=("PLA", "DEF", "CLA", "EVI", "FAC", "DOC", "FOR", "AVE"),
    ),
    "DD": ScenarioSpec(
        name="DD",
        alias="J1Bench.Scenario.DD",
        roles=("lawyer", "specific_character"),
        default_agents={
            "lawyer": "Agent.Lawyer.GPT_DD",
            "specific_character": "Agent.Specific_character.GPT_DD",
        },
        default_max_turns=15,
        metrics=("RES", "DEF", "EVI", "DOC", "FOR", "AVE"),
    ),
    "KQ": ScenarioSpec(
        name="KQ",
        alias="J1Bench.Scenario.KQ",
        roles=("trainee", "general_public"),
        default_agents={
            "trainee": "Agent.Trainee.ConsultGPT",
            "general_public": "Agent.General_public.ConsultGPT",
        },
        default_max_turns=15,
        metrics=("BIN", "NBIN", "AVE"),
    ),
    "LC": ScenarioSpec(
        name="LC",
        alias="J1Bench.Scenario.LC",
        roles=("trainee", "general_public"),
        default_agents={
            "trainee": "Agent.Trainee.LC_GPT",
            "general_public": "Agent.General_public.LC_GPT",
        },
        default_max_turns=10,
        metrics=("BIN", "NBIN", "AVE"),
    ),
}


ALL_SCENARIO_NAMES = tuple(SCENARIOS.keys())
SUPPORTED_SCENARIOS = tuple(n for n, s in SCENARIOS.items() if s.supported)


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        raise KeyError(f"unknown J1Bench scenario: {name!r}; "
                       f"expected one of {ALL_SCENARIO_NAMES}")
    return SCENARIOS[name]


def require_supported(name: str) -> ScenarioSpec:
    spec = get_scenario(name)
    if not spec.supported:
        raise NotImplementedError(
            f"J1Bench scenario {name!r} is registered but not executable in "
            f"this installation. Supported: {sorted(SUPPORTED_SCENARIOS)}."
        )
    return spec
