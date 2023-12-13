# Log file

## 2023-12-12

Recording the steps to get to the combined catalog `/project/chihway/data/decade/metacal_gold_combined_20231212.hdf`. This file is close to what we want eventually, except that we do not have fitvd quantities.

* We have three tile lists to start with:
1) DR3_1_1: https://github.com/delve-survey/shearcat/blob/main/Tilelist/11072023/new_final_list_DR3_1_1.txt
2) DR3_1_2 (without rerun): https://github.com/delve-survey/shearcat/blob/main/Tilelist/07112023/new_final_list_DR3_1_2_without_rerun.txt
3) DR3_1_2 (with rerun): https://github.com/delve-survey/shearcat/blob/main/Tilelist/07112023/Tilelist_Reprocess_20231207.csv

* Run metacal:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/measurement
  submit_mcal_batch_midway3_DR3_1_1.sh
  submit_mcal_batch_midway3_DR3_1_2.sh
  submit_mcal_batch_midway3_DR3_1_2_rerun.sh
  ```
  
* Download SE catalogs:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebook
  python QueryGold_v2.py
  cd /project/chihway/chihway/shearcat/shear_catalog/measurement
  gold_cuts_v2.sh
  ```

* Make joint masks and matching metacal and SE catalogs:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebook
  submit_CombineGoldMetacal.sh 
  ```

* Combine column-by-column, then merge to form one giant file:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebooks/combine_column_20231212
  submit_CombineGoldMetacal_cols.sh
  python CombineGoldMetacal_final_merge.py
  ```
  
* Get grid of response and sigmae to for weights:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/response_s2n_size
  submit_calculate_response_size_s2n.sh
  ```
  
* Add weights, foreground, extinction-corrected flux, s/g flag to final catalog:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebooks
  CombineGoldMetacal_additional_columns.py X
  # X above can be sg, foreground, weights, dered
  ```

