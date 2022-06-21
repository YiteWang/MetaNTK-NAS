# This is a sample script for 5-cells experiments on TieredImageNet.

args=(
    --gpu 0 \

    # search space settings
    --space darts_fewshot \
    --max_nodes 3 \

    # Dataset setting
    --dataset MetaTieredImageNet \
    --dartsbs 3 \

    # Random seed
    --seed -1 \

    # If use only linear regions 
    --only_lrs false \

    # NTK/MetaNTK setting
    --ntk_type MetaNTK_anl \
    --algorithm MAML \
    --inner_lr_time 1000 \
    --reg_coef 1e-3 \

    # Search/evaluate architecture setting
    --ntk_channels 48 \
    --ntk_layers 5 \
    --train_after_search true \
)

python prune_launch.py "${args[@]}"