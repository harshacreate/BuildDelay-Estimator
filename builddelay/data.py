"""Synthetic dataset for BuildDelay-Estimator — Construction Schedule Risk Prototype; no public dataset is used."""
from __future__ import annotations
import numpy as np
FEATURES=['experience', 'size', 'change_orders', 'weather_days', 'crew', 'material_delay', 'budget_overrun']
GROUP_NAME="entity_id"
GROUP_SPLIT=True
def make_dataset(n=2400,seed=42):
    if n<160: raise ValueError("n must be at least 160")
    rng=np.random.default_rng(seed)
    contractors=max(24,n//40); contractor=np.arange(n)%contractors; experience=rng.uniform(1,30,contractors)[contractor]; size=np.clip(rng.lognormal(3.1,.55,n),10,900); change_orders=rng.poisson(1.2,n).astype(float); weather_days=rng.poisson(3,n).astype(float); crew=np.clip(rng.normal(32,10,n),5,90); material_delay=np.clip(rng.gamma(1.4,3,n),0,30); budget_overrun=np.clip(rng.normal(.05,.08,n),-.2,.5); z=-2.8+.0018*size+.28*change_orders+.19*weather_days-.012*crew+.095*material_delay+2.4*budget_overrun-.018*experience; p=1/(1+np.exp(-z)); y=(rng.random(n)<p).astype(int); groups=contractor
    X=np.column_stack([experience, size, change_orders, weather_days, crew, material_delay, budget_overrun]).astype(float)
    if X.shape!=(n,len(FEATURES)) or not np.isfinite(X).all(): raise ValueError("invalid feature matrix")
    if np.asarray(groups).shape!=(n,): raise ValueError("invalid group vector")
    return X,y.astype(int),np.asarray(groups)
