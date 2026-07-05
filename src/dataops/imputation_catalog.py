"""
imputation_catalog — the set of time-series imputation methods the pipeline can
route a gappy dataset to.

The pipeline does **not** run imputation itself (per the design: these modules
are reusable building blocks, not the imputation owner). It only advertises this
catalog in the handoff signal so an external orchestrator — or the user, via
``config.imputation`` — can pick an ``(app, method)`` pair. Each app's
``run_imputation.py`` then consumes the ``prepared-dir`` bundle:

    run_imputation.py --prepared-dir <bundle> --output-dir <out> \\
        --method <method> --inputs train test

The method lists mirror each app's ``--method`` choices (the single source of
truth on the app side):
  * Darts          — ``dockers/tools/Darts_app/run_imputation.py`` (``SUPPORTED``)
  * ImputeGAP      — ``dockers/tools/ImputeGAP_app/run_imputation.py`` (resolved registry)
  * PyPOTS         — ``dockers/tools/PyPOTS_app/run_imputation.py`` (``--method`` choices)
  * WaveStitchPlus — diffusion v1 + local-anchoring v2

``known_failing`` records method/install combinations observed to fail in the
``autofeat-6g`` experiment env, so a selection can be flagged without blocking.
"""
from __future__ import annotations

from typing import Optional

__all__ = ["CATALOG", "get_catalog", "list_apps", "validate_selection", "METADATA"]


# app -> {methods, default, known_failing}
CATALOG: dict[str, dict] = {
    "Darts": {
        "methods": ["auto", "linear", "quadratic", "cubic", "nearest",
                    "slinear", "zero", "kalman"],
        "default": "auto",
        "known_failing": ["kalman"],
    },
    "ImputeGAP": {
        "methods": [
            "mean", "mean_by_series", "min", "zero", "interpolation", "knn",
            "cdrec", "grouse", "iterative_svd", "rosl", "spirit", "svt",
            "soft_impute", "trmf", "dynammo", "stmvl", "tkcm",
            "iim", "mice", "miss_forest", "xgboost",
            "brits", "mrnn", "gain", "deep_mvi", "miss_net", "pristi",
        ],
        "default": "iim",
        "known_failing": [],
        "note": "available methods are resolved from the installed ImputeGAP; "
                "run `run_imputation.py --list` inside the app image to confirm.",
    },
    "PyPOTS": {
        "methods": ["saits", "brits", "transformer", "gpvae", "mrnn",
                    "csdi", "usgan", "timesnet"],
        "default": "saits",
        "known_failing": ["saits"],
    },
    "WaveStitchPlus": {
        "methods": ["v1", "v2", "v2_tuned"],
        "default": "v2",
        "known_failing": [],
    },
}


def get_catalog() -> dict[str, dict]:
    """Return a deep-ish copy of the imputation catalog for the handoff signal."""
    return {app: dict(spec) for app, spec in CATALOG.items()}


def list_apps() -> list[str]:
    """Names of the registered imputation apps."""
    return list(CATALOG.keys())


def validate_selection(
    app: Optional[str], method: Optional[str]
) -> dict:
    """Validate a configured ``(app, method)`` against the catalog.

    Returns ``{status, app, method, message}`` where ``status`` is one of
    ``none_configured`` | ``invalid`` | ``known_failing`` | ``ok``. This never
    raises — an invalid selection is reported, not enforced, so the host system
    keeps full control over what actually runs.
    """
    if not app and not method:
        return {
            "status": "none_configured",
            "app": app,
            "method": method,
            "message": "no imputation.app/method configured; "
                       "selection left to the external orchestrator",
        }
    if app not in CATALOG:
        return {
            "status": "invalid",
            "app": app,
            "method": method,
            "message": f"unknown app {app!r}; choose from {list_apps()}",
        }
    spec = CATALOG[app]
    chosen = method or spec["default"]
    if chosen not in spec["methods"]:
        return {
            "status": "invalid",
            "app": app,
            "method": chosen,
            "message": f"method {chosen!r} not in {app} catalog: {spec['methods']}",
        }
    if chosen in spec.get("known_failing", []):
        return {
            "status": "known_failing",
            "app": app,
            "method": chosen,
            "message": f"{app}/{chosen} is recorded as known-failing in the "
                       "autofeat-6g env; proceed only if it was since fixed",
        }
    return {
        "status": "ok",
        "app": app,
        "method": chosen,
        "message": f"{app}/{chosen} selected",
    }


METADATA = {
    "name": "imputation_catalog",
    "version": "0.1.0",
    "category": "configuration",
    "summary": "Catalog of time-series imputation apps/methods advertised in the handoff signal.",
    "entrypoint": "dataops.imputation_catalog:get_catalog",
    "gpu": False,
    "dependencies": [],
    "outputs": {
        "catalog": {
            "type": "dict",
            "schema": "imputation_catalog",
            "keys": ["Darts", "ImputeGAP", "PyPOTS", "WaveStitchPlus"],
        },
    },
    "artifacts": [],
}
