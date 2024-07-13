import tensorflow as tf
from data_preparation import visualize_data, prepare_data
from model_building import build_model
from training import train_model
from evaluation import evaluate_model
# from submission import prepare_submission
import datetime, os


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
    print(tf.config.list_physical_devices('GPU'))

# def main():
#     path = './ofa-ai-mastery-computer-vision'
#     setup_gpu()
#     visualize_data(path)
#     train_ds, val_ds = prepare_data(path)
#     model = build_model()
#     history = train_model(model, train_ds, val_ds)
#     evaluate_model(model, val_ds, history)
#     prepare_submission(model, './Sample_submission.csv', './ofa-ai-mastery-computer-vision/test/test')

def main():
    path = './ofa-ai-mastery-computer-vision'
    setup_gpu()
    visualize_data(path)
    train_ds, val_ds = prepare_data(path)
    model = build_model()
    
    # Set up the log directory for TensorBoard
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    history = train_model(model, train_ds, val_ds, log_dir)
    evaluate_model(model, val_ds, history)
    # prepare_submission(model, './Sample_submission.csv', './ofa-ai-mastery-computer-vision/test/test')


if __name__ == "__main__":
    main()
