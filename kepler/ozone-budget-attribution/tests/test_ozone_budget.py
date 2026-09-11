"""Sealed verifier for the ozone budget attribution task.

Grading reads only the agent's artifacts under /app/output and the sealed
expectations in /tests/truth.json.  truth.json was produced by an evaluator that
is independent of the reference solution: it aggregates the archive with xarray
whole-array reductions, while the reference walks the intervals explicitly.

Every scored scalar is compared on its own, so a budget whose total happens to
be right because two wrong terms cancel does not pass.  The closure gate
recomputes the residual from the agent's own reported terms against the sealed
endpoint burdens instead of trusting the residual column.
"""

import csv
import json
import math
import os

import pytest

OUTPUT = "/app/output"
TRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truth.json")

SCENARIOS = ["00", "10", "01", "11"]
VOLUMES = ["fixed_urban", "moving_plume"]
WINDOWS = ["W1", "W2", "W3", "W4"]
QUANTITIES = ["M0", "M1", "dM", "A", "K", "P", "L", "C", "D", "G", "residual"]
SERIES = ["00", "10", "01", "11", "d_fire_A0", "d_fire_A1",
          "d_urban_F0", "d_urban_F1", "interaction"]
CONTRAST_COLS = ["d_fire_A0", "d_fire_A1", "d_urban_F0", "d_urban_F1", "interaction"]
RECEPTOR_METRICS = ["mean_ppbv", "peak_ppbv", "exposure_ppbvh",
                    "exposure_over40_ppbvh"]

BUDGET_HEADER = ["case_id", "scenario", "volume", "window", "M0_kg", "M1_kg",
                 "A_kg", "A_lateral_kg", "A_vertical_kg", "K_kg",
                 "K_lateral_kg", "K_vertical_kg", "P_kg", "L_kg", "C_kg",
                 "D_kg", "S_kg", "G_kg", "residual_kg"]
CONTRAST_HEADER = ["contrast_id", "volume", "window", "quantity"] + CONTRAST_COLS
RECEPTOR_HEADER = ["row_id", "receptor_id", "series", "window"] + RECEPTOR_METRICS

COLUMN_TO_TERM = {"M0_kg": "M0", "M1_kg": "M1", "A_kg": "A",
                  "A_lateral_kg": "A_lateral", "A_vertical_kg": "A_vertical",
                  "K_kg": "K", "K_lateral_kg": "K_lateral",
                  "K_vertical_kg": "K_vertical", "P_kg": "P", "L_kg": "L",
                  "C_kg": "C", "D_kg": "D", "S_kg": "S", "G_kg": "G",
                  "residual_kg": "residual"}

RTOL = 1.0e-4
ATOL_SCALE = 2.0e-3
RECEPTOR_ATOL_PPBV = 2.0e-3
RECEPTOR_ATOL_PPBVH = 2.0e-2

CHEM_LABELS = {"net_production", "net_destruction", "near_zero"}
CONTRAST_LABELS = {"more_positive", "more_negative", "near_zero"}
INTERACTION_LABELS = {"amplifying", "damping", "near_zero"}
TERM_NAMES = {"A", "K", "C", "D", "G"}

TRUTH = json.load(open(TRUTH_PATH))
SCALES = TRUTH["scales_kg"]


def _read_csv(name, header):
    path = os.path.join(OUTPUT, name)
    if not os.path.isfile(path):
        return None, f"{path} is missing"
    try:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except Exception as exc:                                  # noqa: BLE001
        return None, f"{path} could not be read: {exc}"
    if not rows:
        return None, f"{path} is empty"
    if [c.strip() for c in rows[0]] != header:
        return None, (f"{path} header is {rows[0]} but must be exactly {header}")
    out = []
    for index, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            return None, f"{path} line {index} has {len(row)} fields, expected {len(header)}"
        out.append(dict(zip(header, row)))
    return out, None


def _number(record, column, where):
    raw = record[column].strip()
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"{where}: column {column} is not a number: {raw!r}")
    if not math.isfinite(value):
        pytest.fail(f"{where}: column {column} is not finite: {raw!r}")
    return value


def _close(got, expected, scale, where, term):
    tol = ATOL_SCALE * scale + RTOL * abs(expected)
    if abs(got - expected) > tol:
        pytest.fail(f"{where}: {term} is {got!r}, reference {expected!r}, "
                    f"difference {abs(got - expected):.6g} exceeds tolerance {tol:.6g}")


@pytest.fixture(scope="module")
def budgets():
    rows, problem = _read_csv("budgets.csv", BUDGET_HEADER)
    if problem:
        pytest.fail(problem)
    return rows


@pytest.fixture(scope="module")
def contrasts():
    rows, problem = _read_csv("contrasts.csv", CONTRAST_HEADER)
    if problem:
        pytest.fail(problem)
    return rows


@pytest.fixture(scope="module")
def receptors():
    rows, problem = _read_csv("receptors.csv", RECEPTOR_HEADER)
    if problem:
        pytest.fail(problem)
    return rows


@pytest.fixture(scope="module")
def diagnosis():
    path = os.path.join(OUTPUT, "diagnosis.json")
    if not os.path.isfile(path):
        pytest.fail(f"{path} is missing")
    try:
        return json.load(open(path))
    except Exception as exc:                                  # noqa: BLE001
        pytest.fail(f"{path} is not valid JSON: {exc}")


@pytest.fixture(scope="module")
def budget_map(budgets):
    table = {}
    for record in budgets:
        key = record["case_id"].strip()
        table[key] = {term: _number(record, column, f"budgets.csv row {key}")
                      for column, term in COLUMN_TO_TERM.items()}
    return table


# --------------------------------------------------------------------------
# Input and output integrity
# --------------------------------------------------------------------------
def test_workflow_delivered():
    directory = os.path.join(OUTPUT, "workflow")
    assert os.path.isdir(directory), f"{directory} is missing"
    run_note = os.path.join(directory, "RUN.txt")
    assert os.path.isfile(run_note), f"{run_note} is missing"
    assert os.path.getsize(run_note) >= 100, (
        f"{run_note} holds only {os.path.getsize(run_note)} bytes; it has to "
        "record the exact invocation and name the inputs and outputs")
    payload = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            payload += os.path.getsize(os.path.join(root, name))
    assert payload >= 200, (
        f"{directory} holds only {payload} bytes; the analysis that produced "
        "the tables has to be delivered with them")


def test_budgets_row_set(budgets):
    seen = [record["case_id"].strip() for record in budgets]
    assert len(seen) == len(set(seen)), "budgets.csv contains duplicate case_id values"
    expected = {f"{s}|{v}|{w}" for s in SCENARIOS for v in VOLUMES for w in WINDOWS}
    assert set(seen) == expected, (
        f"budgets.csv case_id set is wrong; missing {sorted(expected - set(seen))}, "
        f"unexpected {sorted(set(seen) - expected)}")
    for record in budgets:
        key = record["case_id"].strip()
        scenario, volume, window = key.split("|")
        assert record["scenario"].strip() == scenario
        assert record["volume"].strip() == volume
        assert record["window"].strip() == window


def test_contrasts_row_set(contrasts):
    seen = [record["contrast_id"].strip() for record in contrasts]
    assert len(seen) == len(set(seen)), "contrasts.csv contains duplicate ids"
    expected = {f"{v}|{w}|{q}" for v in VOLUMES for w in WINDOWS for q in QUANTITIES}
    assert set(seen) == expected, (
        f"contrasts.csv id set is wrong; missing {sorted(expected - set(seen))}, "
        f"unexpected {sorted(set(seen) - expected)}")


def test_receptors_row_set(receptors):
    seen = [record["row_id"].strip() for record in receptors]
    assert len(seen) == len(set(seen)), "receptors.csv contains duplicate ids"
    expected = set(TRUTH["receptors"])
    assert set(seen) == expected, (
        f"receptors.csv id set is wrong; missing {sorted(expected - set(seen))}, "
        f"unexpected {sorted(set(seen) - expected)}")


# --------------------------------------------------------------------------
# Per-term accuracy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("term", ["M0", "M1", "A", "A_lateral", "A_vertical",
                                  "K", "K_lateral", "K_vertical", "P", "L",
                                  "C", "D", "S", "G"])
def test_budget_term(budget_map, term):
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        _close(row[term], TRUTH["budgets"][key][term], scale,
               f"budgets.csv {key}", term)


def test_transport_split_matches_its_total(budget_map):
    """A and K have to be assembled from the interface fluxes, not inferred.

    A total obtained by subtracting the other terms from the burden change
    cannot be split by interface orientation, so the split is graded too and
    has to agree with the reported total.
    """
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        _close(row["A"], row["A_lateral"] + row["A_vertical"], scale,
               f"budgets.csv {key}", "A_kg against its lateral and vertical parts")
        _close(row["K"], row["K_lateral"] + row["K_vertical"], scale,
               f"budgets.csv {key}", "K_kg against its lateral and vertical parts")


def test_gross_chemistry_is_not_a_radical_diagnostic(budget_map):
    """P and L must be the gross O3 channels, so C has to equal P - L."""
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        _close(row["C"], row["P"] - row["L"], scale,
               f"budgets.csv {key}", "C against the reported P - L")
        assert row["P"] >= -ATOL_SCALE * scale, f"{key}: gross production is negative"
        assert row["L"] >= -ATOL_SCALE * scale, f"{key}: gross destruction is negative"
        assert row["D"] >= -ATOL_SCALE * scale, f"{key}: deposition is negative"


def test_closure_recomputed_from_reported_terms(budget_map):
    """Rebuild the residual from the agent's terms and the sealed endpoints."""
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        truth = TRUTH["budgets"][key]
        rebuilt = (truth["M1"] - truth["M0"]) - (
            row["A"] + row["K"] + row["C"] - row["D"] + row["S"] + row["G"])
        tol = ATOL_SCALE * scale
        assert abs(rebuilt) <= tol, (
            f"budgets.csv {key}: the reported process terms do not close the "
            f"sealed burden change; leftover {rebuilt:.6g} kg exceeds {tol:.6g} kg")


def test_reported_residual(budget_map):
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        _close(row["residual"], TRUTH["budgets"][key]["residual"], scale,
               f"budgets.csv {key}", "residual_kg")


def test_mask_accounting(budget_map):
    """G vanishes on the fixed volume and is not absorbed into transport."""
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        scale = SCALES[f"{volume}|{window}"]
        if volume == "fixed_urban":
            assert abs(row["G"]) <= ATOL_SCALE * scale, (
                f"budgets.csv {key}: the volume does not move, so G must be zero, "
                f"not {row['G']:.6g} kg")
        else:
            expected = TRUTH["budgets"][key]["G"]
            _close(row["G"], expected, scale, f"budgets.csv {key}", "G_kg")
            truth = TRUTH["budgets"][key]
            drift = row["A"] + row["K"] - (truth["A"] + truth["K"])
            budget_tol = (2.0 * ATOL_SCALE * scale
                          + RTOL * (abs(truth["A"]) + abs(truth["K"])))
            assert abs(drift) <= budget_tol, (
                f"budgets.csv {key}: the summed transport terms are off by "
                f"{drift:.6g} kg, which is what folding the mask-change term into "
                "advection or mixing looks like")


def test_additional_source_is_zero(budget_map):
    for key, row in budget_map.items():
        volume, window = key.split("|")[1], key.split("|")[2]
        assert abs(row["S"]) <= ATOL_SCALE * SCALES[f"{volume}|{window}"], (
            f"budgets.csv {key}: the archive applies no ozone source other than "
            f"chemistry, so S must be zero, not {row['S']:.6g} kg")


# --------------------------------------------------------------------------
# Counterfactual consistency
# --------------------------------------------------------------------------
@pytest.mark.parametrize("column", CONTRAST_COLS)
def test_contrast_values(contrasts, column):
    for record in contrasts:
        key = record["contrast_id"].strip()
        volume, window, _quantity = key.split("|")
        scale = SCALES[f"{volume}|{window}"]
        got = _number(record, column, f"contrasts.csv row {key}")
        _close(got, TRUTH["contrasts"][key][column], scale,
               f"contrasts.csv {key}", column)


def test_contrasts_agree_with_the_reported_budgets(contrasts, budget_map):
    combos = {"d_fire_A0": ("10", "00"), "d_fire_A1": ("11", "01"),
              "d_urban_F0": ("01", "00"), "d_urban_F1": ("11", "10")}
    for record in contrasts:
        key = record["contrast_id"].strip()
        volume, window, quantity = key.split("|")
        scale = SCALES[f"{volume}|{window}"]
        own = {}
        for scenario in SCENARIOS:
            row = budget_map[f"{scenario}|{volume}|{window}"]
            own[scenario] = (row["M1"] - row["M0"]) if quantity == "dM" else row[quantity]
        for column, (plus, minus) in combos.items():
            got = _number(record, column, f"contrasts.csv row {key}")
            _close(got, own[plus] - own[minus], scale,
                   f"contrasts.csv {key}", f"{column} against budgets.csv")
        got = _number(record, "interaction", f"contrasts.csv row {key}")
        _close(got, own["11"] - own["01"] - own["10"] + own["00"], scale,
               f"contrasts.csv {key}", "interaction against budgets.csv")


# --------------------------------------------------------------------------
# Receptor diagnostics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("metric", RECEPTOR_METRICS)
def test_receptor_metric(receptors, metric):
    atol = RECEPTOR_ATOL_PPBVH if metric.startswith("exposure") else RECEPTOR_ATOL_PPBV
    for record in receptors:
        key = record["row_id"].strip()
        got = _number(record, metric, f"receptors.csv row {key}")
        expected = TRUTH["receptors"][key][metric]
        tol = atol + RTOL * abs(expected)
        if abs(got - expected) > tol:
            pytest.fail(f"receptors.csv {key}: {metric} is {got!r}, reference "
                        f"{expected!r}, difference {abs(got - expected):.6g} "
                        f"exceeds tolerance {tol:.6g}")


def test_receptor_contrasts_are_differences_of_metrics(receptors):
    table = {}
    for record in receptors:
        key = record["row_id"].strip()
        rid, series, window = key.split("|")
        values = {m: _number(record, m, f"receptors.csv row {key}")
                  for m in RECEPTOR_METRICS}
        table[(rid, series, window)] = values
    combos = {"d_fire_A0": ("10", "00"), "d_fire_A1": ("11", "01"),
              "d_urban_F0": ("01", "00"), "d_urban_F1": ("11", "10")}
    for (rid, series, window) in list(table):
        if series not in combos and series != "interaction":
            continue
        for metric in RECEPTOR_METRICS:
            atol = (RECEPTOR_ATOL_PPBVH if metric.startswith("exposure")
                    else RECEPTOR_ATOL_PPBV)
            if series == "interaction":
                expected = (table[(rid, "11", window)][metric]
                            - table[(rid, "01", window)][metric]
                            - table[(rid, "10", window)][metric]
                            + table[(rid, "00", window)][metric])
            else:
                plus, minus = combos[series]
                expected = (table[(rid, plus, window)][metric]
                            - table[(rid, minus, window)][metric])
            got = table[(rid, series, window)][metric]
            tol = 4.0 * atol + RTOL * abs(expected)
            assert abs(got - expected) <= tol, (
                f"receptors.csv {rid}|{series}|{window}: {metric} is {got!r} but "
                f"the contrast of the reported run values is {expected!r}; every "
                "contrast metric is a difference of the run metrics")


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------
def test_diagnosis_structure(diagnosis):
    assert isinstance(diagnosis, dict), "diagnosis.json must hold a JSON object"
    assert diagnosis.get("schema_version") == "kepler-o3-diagnosis-1.0", (
        f"diagnosis.json schema_version is {diagnosis.get('schema_version')!r}")
    rows = diagnosis.get("rows")
    assert isinstance(rows, list), "diagnosis.json needs a rows list"
    keys = [(r.get("volume"), r.get("window")) for r in rows]
    assert len(keys) == len(set(keys)), "diagnosis.json repeats a volume and window"
    assert set(keys) == {(v, w) for v in VOLUMES for w in WINDOWS}, (
        f"diagnosis.json covers {sorted(keys)}")


def test_diagnosis_labels(diagnosis):
    expected = {(r["volume"], r["window"]): r for r in TRUTH["diagnosis"]}
    for row in diagnosis["rows"]:
        key = (row["volume"], row["window"])
        want = expected[key]
        where = f"diagnosis.json {key[0]}|{key[1]}"

        tau = float(row["tau_kg"])
        scale = SCALES[f"{key[0]}|{key[1]}"]
        assert abs(tau - want["tau_kg"]) <= ATOL_SCALE * scale, (
            f"{where}: tau_kg is {tau!r}, reference {want['tau_kg']!r}")

        chem = row["scenario_chemistry"]
        assert set(chem) == set(SCENARIOS), f"{where}: scenario_chemistry keys"
        for scenario in SCENARIOS:
            assert chem[scenario] in CHEM_LABELS, (
                f"{where}: scenario_chemistry[{scenario}] is {chem[scenario]!r}")
            assert chem[scenario] == want["scenario_chemistry"][scenario], (
                f"{where}: scenario_chemistry[{scenario}] is {chem[scenario]!r}, "
                f"reference {want['scenario_chemistry'][scenario]!r}")

        fire = row["fire_chemistry_contrast"]
        assert set(fire) == {"A0", "A1"}, f"{where}: fire_chemistry_contrast keys"
        for side in ("A0", "A1"):
            assert fire[side] in CONTRAST_LABELS, (
                f"{where}: fire_chemistry_contrast[{side}] is {fire[side]!r}")
            assert fire[side] == want["fire_chemistry_contrast"][side], (
                f"{where}: fire_chemistry_contrast[{side}] is {fire[side]!r}, "
                f"reference {want['fire_chemistry_contrast'][side]!r}")

        assert row["interaction_label"] in INTERACTION_LABELS
        assert row["interaction_label"] == want["interaction_label"], (
            f"{where}: interaction_label is {row['interaction_label']!r}, "
            f"reference {want['interaction_label']!r}")

        dominant = row["dominant_term"]
        assert set(dominant) == set(SCENARIOS), f"{where}: dominant_term keys"
        for scenario in SCENARIOS:
            assert dominant[scenario] in TERM_NAMES, (
                f"{where}: dominant_term[{scenario}] is {dominant[scenario]!r}")
            assert dominant[scenario] == want["dominant_term"][scenario], (
                f"{where}: dominant_term[{scenario}] is {dominant[scenario]!r}, "
                f"reference {want['dominant_term'][scenario]!r}")

        got = row["transport_sustains_anomaly"]
        assert isinstance(got, bool), (
            f"{where}: transport_sustains_anomaly must be a JSON boolean")
        assert got == want["transport_sustains_anomaly"], (
            f"{where}: transport_sustains_anomaly is {got!r}, reference "
            f"{want['transport_sustains_anomaly']!r}")


def test_labels_follow_the_reported_numbers(diagnosis, budget_map):
    """A label table has to be derivable from the tables the agent delivered."""
    for row in diagnosis["rows"]:
        volume, window = row["volume"], row["window"]
        where = f"diagnosis.json {volume}|{window}"
        tau = float(row["tau_kg"])
        scale = SCALES[f"{volume}|{window}"]
        slack = ATOL_SCALE * scale
        own = {s: budget_map[f"{s}|{volume}|{window}"] for s in SCENARIOS}
        assert abs(tau - 0.005 * abs(own["11"]["M0"])) <= slack, (
            f"{where}: tau_kg is not 0.005 times the reported M0 of run 11")
        for scenario in SCENARIOS:
            value = own[scenario]["C"]
            label = row["scenario_chemistry"][scenario]
            if label == "net_production":
                assert value > tau - slack, f"{where}: C{scenario}={value:.6g} is not production"
            elif label == "net_destruction":
                assert value < -tau + slack, f"{where}: C{scenario}={value:.6g} is not destruction"
            else:
                assert abs(value) < tau + slack, f"{where}: C{scenario}={value:.6g} is not near zero"
