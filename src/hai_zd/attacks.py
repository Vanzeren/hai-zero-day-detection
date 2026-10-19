"""Attack IDs and documented targets from HAI technical details v4.0, pp. 31-38."""

from __future__ import annotations

ATTACK_TARGETS: dict[str, dict[str, list[str]]] = {
    "21.03": {
        "A101": ["P1_B2016"], "A102": ["P1_B3005"], "A103": ["P1_LCV01D"],
        "A104": ["P2_SCO"], "A105": ["P2_AutoSD"], "A201": ["P2_VTR02"],
        "A202": ["P1_B2016", "P1_PIT01"], "A203": ["P1_LCV01D"],
        "A204": ["P1_B3005", "P1_FT03"], "A205": ["P1_B2016"],
        "A206": ["P1_FCV03D", "P1_FT03"], "A207": ["P2_VTR01"],
        "A208": ["P1_B3004", "P1_LIT01"], "A209": ["P1_B3004"],
        "A210": ["P1_PCV01D", "P1_PIT01"], "A211": ["P2_AutoSD"],
        "A212": ["P2_AutoSD", "P2_SIT01"], "A213": ["P1_LCV01D", "P1_LIT01"],
        "A214": ["P1_PCV01D"], "A215": ["P2_SCO", "P2_SIT01"],
        "A216": ["P1_FCV03D"], "A217": ["P2_RTR"], "A218": ["P1_FCV03D"],
        "A219": ["P3_LCP01D"], "A220": ["P2_LCV01D"],
        "A301": ["P1_LCV01D", "P1_B3005"],
        "A302": ["P1_B2016", "P1_PIT01", "P1_B3005"],
        "A303": ["P1_PCV01D", "P1_LCV01D"],
        "A304": ["P2_AutoSD", "P2_VTR01"], "A305": ["P2_SCO", "P2_VTR02"],
        "A306": ["P1_B2016", "P2_AutoSD"],
        "A307": ["P1_FCV03D", "P2_VTR01"],
        "A308": ["P1_LCV01D", "P1_LIT01", "P2_AutoSD"],
        "A401": ["P1_PCV01D"],
        "A402": ["P1_B3004", "P1_LIT01", "P1_B3005", "P1_FT03", "P2_LCV01D"],
        "A403": ["P1_B3004", "P1_LIT01", "P2_LCV01D"],
        "A404": ["P2_SCO", "P2_SIT01", "P1_LCV01D", "P1_LIT01"],
        "A405": ["P2_AutoSD", "P2_LCV01D"],
        "A501": ["P1_PCV01D", "P2_VTR02"],
        "A502": ["P1_FCV03D", "P1_FT03", "P2_SCO"],
        "A503": ["P1_B3004", "P1_LIT01", "P2_SCO"],
        "A504": ["P1_FCV03D", "P2_LCV01D"],
        "A505": ["P1_B3004", "P2_AutoSD"],
        "A506": ["P2_SCO", "P2_SIT01", "P2_LCV01D"],
        "A507": ["P2_AutoSD", "P2_VTR01"],
        "A508": ["P1_FCV03D", "P1_LCV01D"],
        "A509": ["P1_B2016", "P1_PCV01D"],
        "A510": ["P1_B3004", "P1_LCV01D", "P1_LIT01"],
        "A511": ["P2_RTR", "P2_LCV01D"],
        "A512": ["P1_B3005", "P1_FCV03D", "P1_FT03"],
    },
    "23.05": {
        "A101": ["P1_B2016"], "A102": ["P1_B2016", "P1_PIT01"],
        "A103": ["P1_PCV01D"], "A104": ["P1_PCV01D", "P1_PIT01"],
        "A105": ["P1_PCV01D"], "A106": ["P1_B2016", "P1_PIT01"],
        "A107": ["P1_B2016"], "A108": ["P1_B3005"],
        "A109": ["P1_B3005", "P1_FT03"], "A110": ["P1_FCV03D"],
        "A111": ["P1_FCV03D", "P1_FT03"], "A112": ["P1_FCV03D"],
        "A113": ["P1_B3005", "P1_FT03", "P1_LIT01"], "A114": ["P1_B3005"],
        "A201": ["P1_B3004"], "A202": ["P1_B3004", "P1_LIT01"],
        "A203": ["P1_LCV01D"], "A204": ["P1_LCV01D", "P1_LIT01"],
        "A205": ["P1_LCV01D"],
        "A206": ["P1_B2016", "P1_PIT01", "P1_FIT"], "A207": ["P1_LCV01D"],
        "A208": ["P1_LCV01D", "P1_LIT01", "P1_FIT"],
        "A209": ["P1_PP04"], "A210": ["P1_PP04"],
        "A211": ["P1_PP04", "P1_TIT03"], "A212": ["P1_PP04"],
        "A213": ["P1_PP04_SP"], "A214": ["P1_FCV01D"],
        "A215": ["P1_FCV01D", "P1_TIT01"], "A216": ["P1_FCV01D"],
        "A217": ["P1_B4002"], "A218": ["P1_PCV01D"], "A219": ["P1_FCV03D"],
        "A220": ["1002-14-AVALGEN.R1"], "A221": ["1004-21-LOWMON.R1"],
        "A222": ["1001-09-AVALGEN.R1"], "A223": ["1020-15-FUNCTION.S4"],
        "A224": ["P1_B3004", "P2_AutoSD"], "A225": ["P1_LCV01D", "P2_VTR01"],
        "A226": ["P1_PCV01D", "P1_FCV03D"],
        "A227": ["P1_B3005", "P1_FT03", "P1_B3004"],
        "A228": ["P1_PCV01D", "P1_PIT01", "P2_SCO", "P2_SIT01"],
        "A229": ["P1_B4002", "P1_B2016"],
        "A230": ["P1_FCV01D", "P1_B2016", "P1_PIT01"],
        "A231": ["P1_B3005", "P3_LCP01D"],
        "A232": ["P1_B4002", "P2_AutoSD", "P2_SIT01"],
        "A233": ["P1_LCV01D", "P1_LIT01", "P2_VTR02"],
        "A234": ["P1_PP04_SP"], "A235": ["1003-05-FUNCTION.T9"],
        "A236": ["OCB0004018.T6"], "A237": ["OCB0002006.S5"],
        "A238": ["1002-31-FUNCTION.S5"],
    },
}

# The v4.0 table writes P2_LCV01D under P3-LC-CO2; HAI only contains P3_LCV01D.
TARGET_ALIASES = {"P2_LCV01D": "P3_LCV01D", "P1_PP04_SP": "P1_PP04SP"}


def _ids(first: int, last: int) -> list[str]:
    return [f"A{number}" for number in range(first, last + 1)]


FILE_ATTACKS: dict[str, dict[str, list[str]]] = {
    "21.03": {
        "test1": _ids(101, 105), "test2": _ids(201, 220),
        "test3": _ids(301, 308), "test4": _ids(401, 405),
        "test5": _ids(501, 512),
    },
    "23.05": {"test1": _ids(101, 114), "test2": _ids(201, 238)},
}


def targets(version: str, attack_id: str, canonical: bool = True) -> list[str]:
    result = ATTACK_TARGETS[version][attack_id]
    if canonical:
        return [TARGET_ALIASES.get(tag, tag) for tag in result]
    return result.copy()
