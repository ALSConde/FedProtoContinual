python scripts/grid_search.py \
  --param theta-exp=0.05,0.10,0.15,0.25,0.35 \
  --param theta-alpha=0.03,0.05,0.07,0.09,0.12 \
  --fixed local-epochs=5 \
  --fixed lambda-kd=0.5 \
  --fixed tau=250 \
  --fixed dirichlet-alpha=0.3 \
  --fixed classes-per-step=9 \
  --strategy random --n-trials 30 \
  --metric avg_inc_acc