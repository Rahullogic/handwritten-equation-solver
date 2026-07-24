# Handwritten Equation Solver

Convert an image of a handwritten math equation into its LaTeX form and solve it.

This is an end-to-end deep learning project built in PyTorch, going from raw handwritten symbol images all the way to a trained system that understands math. It's being built in phases — this README covers Phase 1: Symbol Classifier.

Project Goal

Take a photo/scan of a handwritten equation (e.g. 2 + 3 = ?) and:

Detect and recognize each individual symbol in the image
Assemble the symbols into a correct LaTeX / math expression
(Eventually) solve the equation and return the answer.



Phase 1: Symbol Classifier ✅

Before the model can read a whole equation, it first needs to recognize individual handwritten math symbols (digits, +, -, x, y, (, ), and so on). Phase 1 builds and trains that symbol classifier.

Dataset
Handwritten Math Symbols (Kaggle)
82 symbol classes after extracting the full dataset (digits, operators, letters, etc.)
Model

A convolutional neural network (CNN) built from scratch in PyTorch:

*Conv2D → ReLU → MaxPool blocks (with BatchNorm)
*Trained on grayscale 28x28 images
*Input: single symbol image → Output: predicted class (82-way classification)
Training
*Optimizer: Adam (lr=0.01, weight_decay=1e-4)
*Loss: CrossEntropyLoss
*85/15 train/validation split
*Trained for 25 epochs on the full 82-class dataset
*Result: ~85% validation accuracy


How to run this phase:

1.Open the notebook in Google Colab
2.Mount Google Drive (for dataset storage) and set your Kaggle API credentials
3.Run the cells top to bottom:
*Downloads and extracts the dataset
*Builds the ImageFolder dataset and dataloaders
*Defines the CNN (MyNN)
*Trains the model and saves weights to Google Drive
*Trained weights are saved as symbol_classifier_82class.pth
