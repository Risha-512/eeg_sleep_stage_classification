# *EEG sleep stage calssification*

## Sources
 - [CNN Model](https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e)
 - [EEG database](https://www.physionet.org/content/sleep-edfx/1.0.0/#ref1)

RU version of description and code comments are placed in the "ru-version" branch

## Task
Classify EEG data by sleep stages using convolutional neural network

***

## Data parsing
To parse initial *EDF* files into *npz* files run **edf_to_npz_parsing/main.py**

### Parameters:

parameter          | description       | default value
------------------ | ----------------- | -------------
--input_directory  | path to EDF files | ../data_edf
--output_directory | path to npz files | ../data_npz

### ❕❗❕ some files may be broken and not be parsed ❕❗❕

### Example: 

python edf_to_npz_parsing/main.py --input_directory=./edf_dir --output_directory=./npz_dir


***

## Classification
To classify EEG data by sleep stages run **classifier/main.py**

### Parameters:

parameter          | description                                                            | default value
------------------ | ---------------------------------------------------------------------- | -------------
--input_directory  | path to npz files                                                      | ../data_npz
--model_file_path  | path to model file                                                     | ../models/(0_8__0_87).h5
--report_dir_path  | path to directory for reports                                          | ../reports
--plot_dir_path    | path to directory for plots                                            | ../plots
--do_fit           | set *True*, if needs to train model, *False* to load model from file   | *False*

### Example: 

python classifier/main.py --input_directory=./npz_dir --model_file_path=./model.h5 --report_dir_path=./report

### ❕❗❕ if classification results after training are low (accuracy <= 0.4) ❕❗❕
### ❕❗❕ try to replace for loop in classifier/model:36 by three times repeated loop body ❕❗❕
