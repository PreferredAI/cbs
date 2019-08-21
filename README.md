## The implementation of twin networks in the 'Modeling Contemporaneous Basket Sequences with Twin Networks for Next-Item Recommendation'paper (IJCAI'18)

1. Input format(s): 
 + For each CBS instance, the basket sequences and the grouth-truth item are separated by '=>' 
    - e.g.,  support_basket_sequence=>target_basket_sequence=>ground-truth_item_id
 + For each basket sequence, baskets {b_i} are separated by '|'
    - e.g.,  b_1|b_2|b_3|...|b_n
 + For each basket b_i, items {v_j} are separated by a space ' '
    - e.g., v_1 v_2 v_3 ... v_m

2. How to run: main_gpu.sh
  + Use --train_mode to enable the training mode
  + Use --prediction_mode to generate evaluation metrics
  + We support 5 main model types namely: bseq_support, bseq_target, cbs_sn, cbs_cfn, cbs_dfn

3. How to collect results from different seeds: Use collect_result.sh

 

