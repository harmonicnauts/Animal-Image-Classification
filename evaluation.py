import matplotlib.pyplot as plt

def evaluate_model(model, val_ds, history):
    model.evaluate(val_ds)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'], loc='upper right')
    plt.show()
