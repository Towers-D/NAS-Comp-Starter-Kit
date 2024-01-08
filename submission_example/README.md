# CVPR-NAS 2022 Submission Template

## Writing your Submission
For a valid submission, there are a number of functions with this template that need to be implemented. In the 
individual files within the template, the inputs and expected outputs are explained in more depth.

For a valid submission, you are asked to implement the following functions within the following classes:
* DataProcessor
  * `__init__()`: This function receives raw data in the form of numpy arrays for the train, valid, and test data, as well the dataset metadata 
  * `process()`: This function must output 3 PyTorch dataloaders for the train, valid, and test data splits
* NAS
  * `__init__()`: This function receives the dataloaders created by the DataProcessor, and the dataset metadata
  * `search()`: This function should search for an optimal architecture for this dataset, and should output a PyTorch model.
* Trainer
    * `__init__()`: This function receives the dataloaders created by the DataProcessor, and the model produced by the NAS class
    * `train()`: This function should fully train your model and return it
    * `predict(test_loader)`: This function should produce a list of predicted class labels over the test_dataloader

 In general, the evaluation script runs the following pipeline for each dataset:
 1. Raw Dataset -> `DataProcessor` -> Train, Valid, and Test dataloaders
 2. Train Dataloader + Valid Dataloaders -> `NAS` -> Model
 3. Model + Train Dataloader + Valid Dataloaders -> `TRAINER.train` -> Fully-trained model
 4. Fully-trained model + Test Dataloader -> `Trainer.predict=` -> Predictions
 