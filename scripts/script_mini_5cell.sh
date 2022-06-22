# This is a sample script for 5-cells experiments on MiniImageNet.

# Searched architectures examples:
# (1)
# Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], [('conv_3x3', 0), ('sep_conv_3x3', 2)]], 
# normal_concat=[2, 3, 4], reduce=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('conv_1x5_5x1', 1), ('dil_conv_3x3', 2)], [('dil_conv_3x3', 2), ('dil_conv_3x3', 3)]], 
# reduce_concat=[2, 3, 4])

# (2)
# Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], [('conv_3x3', 2), ('sep_conv_3x3', 3)]], 
# normal_concat=[2, 3, 4], reduce=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], [('conv_1x5_5x1', 1), ('sep_conv_3x3', 2)], [('sep_conv_3x3', 1), ('conv_1x5_5x1', 3)]], 
# reduce_concat=[2, 3, 4])

# (3)
# Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('max_pool_3x3BN', 2)], [('dil_conv_3x3', 2), ('sep_conv_3x3', 3)]], 
# normal_concat=[2, 3, 4], reduce=[[('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], [('conv_3x3', 1), ('sep_conv_3x3', 2)], [('sep_conv_3x3', 2), ('sep_conv_3x3', 3)]], 
# reduce_concat=[2, 3, 4])

args=(
    --gpu 0 \

    # search space settings
    --space darts_fewshot \
    --max_nodes 3 \

    # Dataset setting
    --dataset MetaMiniImageNet \
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