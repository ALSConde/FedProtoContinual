python scripts/grid_search.py \
  --param local-epochs=3,5,7,9 \
  --param lambda-kd=0.0,0.5,1.0,2.0,4.0 \
  --param tau=15,30,60,120,250 \
  --fixed dirichlet-alpha=0.3 \
  --fixed classes-per-step=9 \
  --fixed theta-exp=0.15 \
  --fixed theta-alpha=0.5 \
  --strategy random --n-trials 30 \
  --metric avg_inc_acc

