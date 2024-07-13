# Computer Vision Model Training for Kaggle Competition

This project trains a computer vision model using the provided dataset to compete in a Kaggle competition. The goal is to achieve the highest possible score by accurately predicting the labels of the test dataset.

## Project Context

This project was developed for a Kaggle competition hosted by Orbit Future Academy. The model achieved an accuracy of 0.99176, placing 8th out of 255 participants. You can view the leaderboard [here](https://www.kaggle.com/competitions/ofa-ai-mastery-computer-vision/leaderboard).

## Project Structure

- `train_model.py`: The script to train the model using the provided dataset.
- `show_eval.py`: The script to load the trained model, predict the test data, and visualize the accuracy.
- `Model.h5`: The pre-trained model file (ensure this file is in the same directory as `show_eval.py`).
- `Sample_submission.csv`: The sample submission CSV file.
- `./ofa-ai-mastery-computer-vision/`: Directory containing the training and test images.

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Matplotlib
- Pandas
- NumPy
- OpenCV

## Installation

1. Clone this repository.
2. Install the required Python packages using pip:

   ```bash
   pip install tensorflow matplotlib pandas numpy opencv-python
   ```

## Usage

### Training the Model

1. Ensure your training images are located in `./ofa-ai-mastery-computer-vision/train`.
2. Run the `train_model.py` script:

   ```bash
   python train_model.py
   ```

This script will train the model using the provided training data and save the trained model as `Model.h5`.

### Evaluating and Predicting

1. Ensure you have the trained model saved as `Model.h5`.
2. Ensure your test images are located in `./ofa-ai-mastery-computer-vision/test/test`.
3. Ensure you have the submission CSV file `Sample_submission.csv`.

4. Run the `show_eval.py` script:

   ```bash
   python show_eval.py
   ```

This script will load the trained model, predict the test data, and display the accuracy as a bar chart. It will also save the predictions in `submission_file.csv`.

## Script Explanation

### `train_model.py`

- **train_model()**: Trains the model using the provided dataset and saves the trained model as `Model.h5`.

### `show_eval.py`

- **load_model(model_path)**: Loads the pre-trained model from the specified path.
- **evaluate_model(model, val_ds)**: Evaluates the model on the validation dataset and prints the loss and accuracy.
- **plot_accuracies(evaluation)**: Plots the accuracy of the model as a bar chart.
- **predict_and_evaluate(model, sub_csv, path_test)**: Uses the model to predict labels for the test data, updates the submission DataFrame, and saves it to `submission_file.csv`.
- **prepare_validation_data(path, image_size, batch_size)**: Prepares the validation dataset.
- **main()**: The main function to orchestrate the loading of the model, preparation of the validation data, evaluation, plotting of accuracies, and predicting the test data.

## Example Output

- The model's accuracy will be displayed as a bar chart.
- The first 10 rows of the submission DataFrame will be printed.
- The submission file `submission_file.csv` will be saved in the current directory.

## Directory Structure

.
├── README.md
├── Model.h5
├── Sample_submission.csv
├── train_model.py
├── show_eval.py
└── ofa-ai-mastery-computer-vision
├── train
│ ├── class1
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ ├── class2
│ │ ├── image1.jpg
│ │ ├── image2.jpg
│ │ └── ...
│ └── ...
└── test
└── test
├── image1.jpg
├── image2.jpg
└── ...
