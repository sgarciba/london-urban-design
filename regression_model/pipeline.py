from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

lst_pipeline = Pipeline(
    [
        # ===== MODEL =====
       ("rf", RandomForestRegressor(
            n_estimators=300, random_state=203, n_jobs=-1,
            max_depth=10, min_samples_leaf=5, max_features="sqrt"
            )
        )
    ]
)

