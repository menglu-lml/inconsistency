## Training
The followng command runs the training and validation experiment.
```
python training.py --database_path="path/to/the/directory/of/your/database" --protocols_path="path/to/the/directory/of/your/protocols"
```

The default configurations are saved at `model_config.yaml`. If you would like change the configurations of the model, simply change the values in this file directly.

## Testing
The followng command is for testing the performance of trained-model on the evaluation set.
```
python testing_ASV.py --pre_trained_model_path="pre_trained.pth" --database_path="path/to/the/directory/of/your/database" --protocols_path="path/to/the/directory/of/your/protocols"
```
After running the previous testing command, an evaluation output file will be created and saved in the current directory, which is named as `eval_scores.txt`. 

# Pre-trained model
To access the pre-trained parameters, please [download](https://drive.google.com/drive/folders/1wMlT0yLUOknuTPM31xyniT3nB7BCyIBl?usp=sharing)

## Contact
If you have questions, please contact `menglu.li@torontomu.ca`.

