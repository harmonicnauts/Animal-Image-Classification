# OFA Computer Vision Final Project (Animal Image Classification)

This project is the final project for Computer Vision class on Orbit Future Academy (OFA). In this project, there's a data in a form of images and the objective is to train a machine learning model and achieve the highest score.

## Project Context & End Result

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

1. Download the training data images and place it in `./ofa-ai-mastery-computer-vision/train`.
2. Run the `train_model.py` script:

   ```bash
   python train_model.py
   ```

This script will train the model using the provided training data and save the trained model as `Model.h5`.

### Evaluation, Prediction, and Getting the Submission File

1. Ensure you have the trained model saved as `Model.h5`.
2. Ensure your test images are located in `./ofa-ai-mastery-computer-vision/test/test`.
3. Ensure you have the submission CSV file `Sample_submission.csv`.

4. Run the `show_eval.py` script:

   ```bash
   python show_eval.py
   ```

This script will load the trained model, predict the test data, and display the accuracy as a bar chart. It will also save the predictions in `submission_file.csv`.
