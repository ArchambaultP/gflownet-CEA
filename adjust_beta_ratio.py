import json
import math
import numpy as np

src = "precomputed/reward_table_sf0.15.json"
dst = "precomputed/reward_table_sf0.15_bratio50.json"

with open(src) as f:
    data = json.load(f)

losses = np.array([v["loss"] for v in data.values()], dtype=float)

# q10-q90 normalization, matching your calibrated setup
q10 = np.quantile(losses, 0.10)
q90 = np.quantile(losses, 0.90)
scale = max(q90 - q10, 1e-12)

# beta from target ratio 50
beta = math.log(50.0)

for k, v in data.items():
    L = float(v["loss"])
    L_norm = (L - q10) / scale
    v["reward"] = float(math.exp(-beta * L_norm))

with open(dst, "w") as f:
    json.dump(data, f, indent=2)

print("saved", dst)
print("beta =", beta)
print("q10 =", q10, "q90 =", q90, "scale =", scale)
