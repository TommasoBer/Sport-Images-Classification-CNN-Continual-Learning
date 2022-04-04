# Sport-Images-Classification-CNN-Continual-Learning
In the framework of Continual Learning, I developed a CNN able to classify 99 different sport images.

Topics: Neural Networks, Computer Vision

Libraries: Tensorflow-keras

The project has been developed in the context of 'Neural Netwoks for Data Science' course held by prof. Simone Scardapane @SapienzaUniversity of Rome.

**********************

### Get data to analyze
The dataset has been provided by Kaggle. It has been uploaded directly to Colab for the first part of the project
and then going local first (for some modifications) and then on the drive, for the Continual Learning part.

Link to the data: [https://www.kaggle.com/gpiosenka/sports-classification](https://www.kaggle.com/gpiosenka/sports-classification)

The dataset contains images of 100 different sports and can be used for an image classification problem. The size of the images is 256x256 pixels with 3 channels. 
It is a pretty clean dataset and it is possible to reach models with 95 - 98 % accuracy according to the description. 
There are no bad images or duplications and it is already splitted in train, test and validation sets. ( A bit unbalanced in favour of training set).

For the Continual Learning Part:
In local, I modified the folders in order to split the tasks (3 tasks), then I uploaded the new directories on my Google-Drive and I read the datasets from there.
I created 3 different folders (one for each tasks), with 33 classes of images, 
splitting the initial folder of 100 classes and deleting one class of images. And I moved some images from the train set to test and validation set.

## Problem to be addressed: Catastrophic Forgetting in Classification (Task Incremental)

Artificial intelligence models must often be able to adapt and transform. Let's say that a model trained on a given dataset subjected
to a continuous flow of information, capable of performing a given task (e.g., recognizing and distinguishing images of soccer and American football) 
finds itself receiving images of other sports (basketball and swimming).
The model will have to adapt and learn to distinguish the new images, carrying out a new task (*Task incremental problem*)
but remembering how to classify what it had already learned (soccer and American Football) avoiding the common problem of **Catastrophic Forgetting**.
The different inputs come from the same domain, the outputs will change according to the taks to be solved and
the model always knows before which task it has to solve.

In order to handle the Task Incremental Continual Learning has been developed a model using as solution a Regularization technique: **Elastic Weight Consolidation**.
Through this solution, as it is trained on new tasks, the model is able to remember previous ones by changing slightly
the value of the most significant weights, but first learning to recognize them.

## Files descriptions
The directory of this homework consists of one notebook.

In this notebook developed on Colab, there are two different parts.

1 `Convolutional NN - VGG Architecture'
> In the first part has been created a model based on a VGG-11 Architecture, with Convolutional blocks, GlobalAveragePooling,
> dropout functions to normalize the model and a fully connected part.

2 `Convolutional NN - Continual Learning'
> Considering the same dataset, I tried to implement A Continual Learning framework able to avoid the *Catastrophic Forgetting* problem
> that affects Neaural Networks when new data arrive with a continuos flow and we need to manage a task incremental problem. This is the most interesting part
> at all. We can observe how the model is able to remember the classification on task1, after that it has been trained on task2 and task3. 
> In order to let it work, a customized loss function has been created, using the *ELASTIC WEIGHT CONSOLIDATION* method

**********************

