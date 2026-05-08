import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn.hmm import CategoricalHMM

import config

def score_rows(model: CategoricalHMM, obs: np.ndarray) -> np.ndarray:
    log_sp = np.log(model.startprob_ + 1e-12)

    # Clip to valid emission index range — guards against a test bin that was
    # absent from normal_train (which would cause hmmlearn to allocate fewer
    # symbols than N_BINS if the highest bin never appeared in training).
    n_sym     = model.emissionprob_.shape[1]
    obs_safe  = np.clip(obs.astype(int), 0, n_sym - 1)

    log_emit = np.log(model.emissionprob_[:, obs_safe].T + 1e-12)

    return logsumexp(log_sp + log_emit, axis=1) # type: ignore

def run_hmm_inference(
    normal_train: pd.DataFrame, 
    test_df: pd.DataFrame,
    n_components: int = config.N_COMPONENTS,
    n_iter: int = config.N_ITER,
    tol: float = config.TOL,
    threshold_perc: float = config.THRESHOLD_PERC
):
    print(f"\nTraining per-sensor CategoricalHMMs "
    f"(n_components={n_components}, n_iter={n_iter})...")

    results_df = pd.DataFrame(index=test_df.index)

    for stage, sensors in config.STAGE_SENSORS.items():
        print(f"\n  [{stage}]  {len(sensors)} sensor(s)...")

        stage_train_ll = np.zeros(len(normal_train))
        stage_test_ll  = np.zeros(len(test_df))

        for sensor in sensors:
            train_obs = normal_train[sensor].values.astype(int)
            test_obs  = test_df[sensor].values.astype(int)

            X_train = train_obs.reshape(-1, 1) # type: ignore

            model = CategoricalHMM(
                n_components=n_components,
                n_iter=n_iter,
                tol=tol,
                random_state=config.RANDOM_SEED,
                verbose=False,
            )

            # Suppress ConvergenceWarning on sensors that converge in fewer iterations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, lengths=[len(X_train)])

            stage_train_ll += score_rows(model, train_obs) # type: ignore
            stage_test_ll  += score_rows(model, test_obs) # type: ignore

            print(f"    {sensor:12s}  converged={model.monitor_.converged}"
                f"  iter={model.monitor_.iter}")

        threshold = np.percentile(stage_train_ll, threshold_perc)

        results_df[f"{stage}_Prediction"] = np.where(
            stage_test_ll < threshold,
            config.ATTACK_LABEL,
            config.NORMAL_LABEL,
        )
        print(f"  → {stage} threshold ({threshold_perc}th pct): {threshold:.4f}")

    return results_df