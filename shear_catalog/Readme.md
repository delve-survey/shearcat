# Log file

## 2024-02-09

This is the instructions to get to the file `/project/chihway/data/decade/metacal_gold_combined_mask_20240209.hdf`.

* We combined all the tile lists into a single one using the script: https://github.com/delve-survey/shearcat/blob/main/Tilelist/final_20240209/reformat_final.py, the final list is here: https://github.com/delve-survey/shearcat/blob/main/Tilelist/final_20240209/Tilelist_final_DR3_1.csv.
  
* Metacal was run previously with separate batches, see log file 2023-12-12. All the files are now together in this directory: `/project/chihway/data/decade/shearcat_final`.  

* Download SE catalogs. They will all sit in this directory: `/project/chihway/data/decade/coaddcat_final`:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python QueryGold_final.py
  ```
* Make joint masks and matching metacal and SE catalogs:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python CombineGoldMetacal_mask_final.py
  ```

* Combine column-by-column, then merge to form one giant file:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat/combine_column_20240209
  submit_CombineGoldMetacal_cols.sh
  python CombineGoldMetacal_final_merge.py
  ```

* Add foreground, extinction-corrected flux, s/g flag to final catalog:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python CombineGoldMetacal_additional_columns.py 'sg'
  python CombineGoldMetacal_additional_columns.py 'foreground'
  python CombineGoldMetacal_additional_columns.py 'dered'
  ```
  
* Get grid of response and sigmae to for weights:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/response_s2n_size
  submit_calculate_response_size_s2n.sh
  ```
  
* Add weights to final catalog:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python CombineGoldMetacal_additional_columns.py weights
  ```

* Make mask for metacal cuts on catalog, without tomographic information. This outputs file to `/project/chihway/data/decade/metacal_gold_combined_mask_20240209.hdf`

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python MakeMcalMask_notomo.py
  ```
* Pass the h5 file at this state to photo-z to get tomographic binning.

* Make final mask that includes tomographic information and merge back to the main h5 file.

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/mastercat
  python MakeMcalMask_tomo.py
  ```

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

* Add foreground, extinction-corrected flux, s/g flag to final catalog:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebooks
  python CombineGoldMetacal_additional_columns.py X
  # X above can be sg, foreground, dered
  ```
  
* Get grid of response and sigmae to for weights:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/response_s2n_size
  submit_calculate_response_size_s2n.sh
  ```
  
* Add weights to final catalog:

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebooks
  python CombineGoldMetacal_additional_columns.py weights
  ```
* Make mask for metacal cuts on catalog, this outputs file to `/project/chihway/data/decade/metacal_gold_combined_mask_20231212.hdf`, which we may want to just add to the master catalog after combining with tomographic information

  ```
  cd /project/chihway/chihway/shearcat/shear_catalog/notebooks
  python MakeMcalMask.py
  ```
  

