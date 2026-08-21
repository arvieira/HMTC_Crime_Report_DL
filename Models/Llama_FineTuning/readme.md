# Instructions

* This directory contains the files required to perform the fine-tuning of Llama 3.1 8B.
* The development process started in the FineTuning.ipynb notebook on the police workstation.
* Once the dataset was prepared, an initial training run was executed to verify that the process was working correctly.
* A dedicated script, located at server_scripts/fine_tuning/fine_tuning.py, was created for execution on the server over a 19-day period.
* The resulting artifacts were generated in the directory server_scripts/fine_tuning/final_finetuned_model.
* These results were then copied to the finetuned_model directory, which would have been used by the original notebook had it been executed through to completion.
* The fine-tuned models stored in finetuned_model and server_scripts/fine_tuning/final_finetuned_model are identical. 