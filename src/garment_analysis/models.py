from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class PCAResult:
    pca: PCA
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray


def run_pca(
    X: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
    random_state: int = 42,
) -> PCAResult:
    """
    PCA on numeric dataframe X (rows=samples, cols=features).
    """
    cols = list(X.columns)
    Xn = X.apply(pd.to_numeric, errors="coerce")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    pipe = Pipeline(steps)
    Xproc = pipe.fit_transform(Xn)

    pca = PCA(n_components=n_components, random_state=random_state)
    scores_arr = pca.fit_transform(Xproc)

    scores = pd.DataFrame(scores_arr, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)])

    # loadings: components_ shape (n_components, n_features)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return PCAResult(
        pca=pca,
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


@dataclass(frozen=True)
class PLSCVResult:
    model: PLSRegression
    pipeline_X: Pipeline
    best_n_components: int
    cv_mse: dict[int, float]
    x_scores: pd.Series  # first latent variable score (t1)
    y_pred: pd.Series
    y_pred_cv: pd.Series


def fit_pls_cv(
    X: pd.DataFrame,
    y: pd.Series,
    max_components: int = 8,
    n_splits: int = 10,
    random_state: int = 42,
) -> PLSCVResult:
    """
    Fit a monotarget PLSRegression with CV-based selection of n_components.
    Returns best model and first X-score (t1) to use as an index if desired.
    """
    Xn = X.apply(pd.to_numeric, errors="coerce")
    yn = pd.to_numeric(y, errors="coerce")

    # drop rows where y is NaN
    mask = ~yn.isna()
    Xn = Xn.loc[mask].copy()
    yn = yn.loc[mask].copy()

    if len(Xn) < 3:
        raise ValueError("At least three observations are required for PLS.")
    if Xn.shape[1] == 0:
        raise ValueError("The PLS predictor matrix has no columns.")

    n_splits = int(min(max(2, n_splits), len(Xn)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_mse: dict[int, float] = {}
    smallest_training_set = len(Xn) - int(np.ceil(len(Xn) / n_splits))
    max_components = int(
        max(1, min(max_components, Xn.shape[1], smallest_training_set))
    )

    splits = list(kf.split(Xn))
    cv_predictions: dict[int, np.ndarray] = {}

    for n_comp in range(1, max_components + 1):
        mses = []
        predictions = np.full(len(Xn), np.nan, dtype=float)
        for train_idx, test_idx in splits:
            Xtr_raw = Xn.iloc[train_idx]
            Xte_raw = Xn.iloc[test_idx]
            ytr, yte = yn.iloc[train_idx].to_numpy(), yn.iloc[test_idx].to_numpy()

            fold_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            Xtr = fold_pipeline.fit_transform(Xtr_raw)
            Xte = fold_pipeline.transform(Xte_raw)

            model = PLSRegression(n_components=n_comp, scale=False)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte).ravel()
            predictions[test_idx] = pred
            mses.append(mean_squared_error(yte, pred))

        cv_mse[n_comp] = float(np.mean(mses))
        cv_predictions[n_comp] = predictions

    best_n = min(cv_mse, key=cv_mse.get)

    pipeline_X = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xproc = pipeline_X.fit_transform(Xn)

    model = PLSRegression(n_components=best_n, scale=False)
    model.fit(Xproc, yn.to_numpy())

    y_pred = pd.Series(model.predict(Xproc).ravel(), index=Xn.index, name=f"PLS_pred_{y.name}")
    y_pred_cv = pd.Series(
        cv_predictions[best_n],
        index=Xn.index,
        name=f"PLS_CV_pred_{y.name}",
    )
    # x_scores_: (n_samples, n_components). Use first latent score as index
    t1 = pd.Series(model.x_scores_[:, 0], index=Xn.index, name="PLS_t1")

    return PLSCVResult(
        model=model,
        pipeline_X=pipeline_X,
        best_n_components=best_n,
        cv_mse=cv_mse,
        x_scores=t1,
        y_pred=y_pred,
        y_pred_cv=y_pred_cv,
    )
