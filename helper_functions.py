import urllib.request
import zipfile
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def unzip(url: str, filename: str) -> None:
    """
    Download and unzip a file from a given URL.

    Args:
        url (str): The URL of the file.
        filename (str): The local filename to save the downloaded content.

    Returns:
        None

    Example:
        unzip('https://example.com/myfile.zip', 'myfile.zip')
    """
    try:
        file, _ = urllib.request.urlretrieve(url, filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        zip_ref.close()
        print(f'Successfully unzipped {filename}')
    except Exception as e:
        print(f'Error: {e}')

def dirwalk(directory: str) -> None:
    """
    Recursively walk through a directory and print the count of directories and images in each subdirectory.

    Args:
        directory (str): The path to the directory to traverse.

    Returns:
        None

    Example:
        dirwalk('my_directory')
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"{len(dirnames)} directories and {len(filenames)} images in {dirpath}")

def tensorboard(dir_name: str, experiment_name: str) -> tf.keras.callbacks.TensorBoard:
    """
    Create a TensorBoard callback for logging training information.

    Args:
        dir_name (str): The base directory for saving TensorBoard logs.
        experiment_name (str): The name of the experiment or model.

    Returns:
        tf.keras.callbacks.TensorBoard: The TensorBoard callback for training.

    Example:
        tensorboard('logs', 'my_experiment')
    """
    log_dir = f'{dir_name}/{experiment_name}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to {log_dir}")
    return callback

def plot_loss(history):
  """
  Plot training and validation loss and accuracy curves.

  Args:
      history (tf.keras.callbacks.History): The training history object.

  Returns:
      None

  Example:
      plot_loss(training_history)
  """
  loss=history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  #Loss plot
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  plt.figure()
  #Accuracy Plot
  plt.plot(epochs, accuracy, label='accuracy')
  plt.plot(epochs, val_acc, label='val_accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()

def plot_augmented_image(train_dir, data_augmentation, train_data):
    target_class = random.choice(train_data.class_names)
    target_dir = os.path.join(train_dir, target_class)
    random_img = random.choice(os.listdir(target_dir))
    random_img_path = os.path.join(target_dir, random_img)

    img = mpimg.imread(random_img_path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Original random image of {target_class}')
    plt.axis(False)

    augmented_img = data_augmentation(tf.expand_dims(img, axis=0), training=True)
    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmented_img)/255.)
    plt.title(f'Augmented image of {target_class}')
    plt.axis(False)
    plt.show()

def compare_histories(prior_history: tf.keras.callbacks.History, new_history: tf.keras.callbacks.History, initial_epochs: int = 5) -> None:
    """
    Compare two TensorFlow History objects by plotting training and validation metrics.

    Args:
        prior_history (tf.keras.callbacks.History): The history of the model's prior training.
        new_history (tf.keras.callbacks.History): The history of the model's fine-tuning or new training.
        initial_epochs (int, optional): The number of initial epochs in the prior_history. Default is 5.

    Returns:
        None

    Example:
        compare_histories(prior_training_history, fine_tuning_history)
    """

    # Get prior metrics
    acc = prior_history.history['accuracy']
    loss = prior_history.history['loss']
    val_acc = prior_history.history['val_accuracy']
    val_loss = prior_history.history['val_loss']

    # Combine metrics from prior and new history
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    # Make accuracy plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), 'k--', label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Make loss plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), 'k--', label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')