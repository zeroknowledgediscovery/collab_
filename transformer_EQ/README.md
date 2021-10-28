# Notebooks:
1. `make_dataset_for_transformer.ipynb`: make dataset for transformer. It uses data contained in the split folder. It can be modified to work on other dataset.
1. `make_sbc_file.ipynb`: make sbc files for slurm. I wrote it for the cluster at BNL, but it should work for any cluster, I guess.

1. `collect_result.ipynb`: form files that compare test ground truth and predictions and save them to the `results/grt_prd` folder.
1. `precision_recall.ipynb`: calculate AUCs using files in the `results/grt_prd` folder.

# AUC file
./Transformer_Earthquake_AUC.csv


# Folders:
1. `checkpoints`: checkpoint files while training and running transformer;
1. `datasets`: Train and test data;
1. `results`: prediction made by transformer;
	- subfolder `grt_prd`: the first column is the ground truth and second column is the prediction. The file ends with `fraction` contain float prediction made by calculating the fraction of 1s in the predicted "sentence".
1. `runfiles`: sbc files and scripts for lauching batch jobs.
