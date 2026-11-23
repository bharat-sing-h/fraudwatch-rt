"""Model registry on MLflow: every trained model is a run (params, metrics, model card,
artifacts) and a registered version; *aliases* say which version does what.

* ``champion``: makes the decisions in production.
* ``challenger``: scored in **shadow** on live traffic (logged, never acted on) until the
  promotion gate decides whether it replaces the champion.

The scorer resolves an alias to a version at startup and on reload, so promotion and rollback
are a single alias move with no redeploy.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fraudwatch.model.bundle import ModelBundle

MODEL_NAME = "fraudwatch"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class _PyfuncWrapper:
    """Makes the bundle a proper MLflow pyfunc (so ``mlflow models serve`` also works)."""

    def __init__(self) -> None:
        self.bundle: ModelBundle | None = None

    def load_context(self, context: Any) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    # deliberately unannotated: MLflow derives a signature from type hints and warns on Any
    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        assert self.bundle is not None
        return self.bundle.predict(model_input[self.bundle.features].to_numpy(dtype=float))


class Registry:
    def __init__(self, tracking_uri: str, experiment: str = "fraudwatch") -> None:
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.client = mlflow.MlflowClient()

    def log_and_register(
        self,
        bundle: ModelBundle,
        *,
        params: dict[str, Any],
        metrics: dict[str, float],
        model_card: str | None = None,
        alias: str | None = None,
    ) -> str:
        """Log a training run, register a new version, optionally point ``alias`` at it."""
        mlflow = self.mlflow
        from mlflow.pyfunc import PythonModel

        wrapper = type("FraudwatchModel", (_PyfuncWrapper, PythonModel), {})
        with (
            mlflow.start_run(run_name=f"train-{bundle.version}") as run,
            tempfile.TemporaryDirectory() as tmp,
        ):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            if model_card:
                mlflow.log_text(model_card, "model_card.md")
            bundle_dir = bundle.save(Path(tmp) / "bundle")
            from mlflow.models import ModelSignature
            from mlflow.types import ColSpec, Schema

            signature = ModelSignature(
                inputs=Schema([ColSpec("double", name) for name in bundle.features]),
                outputs=Schema([ColSpec("double")]),
            )
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper(),
                artifacts={"bundle": str(bundle_dir)},
                pip_requirements=["lightgbm", "numpy", "pandas"],
                signature=signature,
            )
            version = mlflow.register_model(info.model_uri, MODEL_NAME)
            self.client.set_model_version_tag(
                MODEL_NAME, version.version, "bundle_version", bundle.version
            )
            self.client.set_model_version_tag(
                MODEL_NAME, version.version, "run_id", run.info.run_id
            )
        if alias:
            self.set_alias(alias, version.version)
        return str(version.version)

    def set_alias(self, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(MODEL_NAME, alias, version)

    def alias_version(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(MODEL_NAME, alias).version)
        except Exception:  # MLflow raises a generic RestException when the alias is unset
            return None

    def load(self, alias: str) -> ModelBundle:
        version = self.alias_version(alias)
        if version is None:
            raise LookupError(f"no model version has alias {alias!r}")
        local = self.mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}@{alias}"
        )
        bundle_dirs = list(Path(local).rglob("bundle.json"))
        if not bundle_dirs:
            raise FileNotFoundError(f"bundle.json missing in {local}")
        bundle = ModelBundle.load(bundle_dirs[0].parent)
        bundle.metadata["registry_version"] = version
        bundle.metadata["alias"] = alias
        return bundle

    def promote(self, version: str) -> str | None:
        """Make ``version`` champion; the previous champion keeps a ``previous`` alias for
        one-step rollback."""
        previous = self.alias_version("champion")
        if previous is not None and previous != version:
            self.set_alias("previous", previous)
        self.set_alias("champion", version)
        if self.alias_version("challenger") == version:
            self.client.delete_registered_model_alias(MODEL_NAME, "challenger")
        return previous
