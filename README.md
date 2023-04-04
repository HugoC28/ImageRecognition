# Image Recognition
Image recognition project based on CNN for agricultural application.

## Figures Recognition
We first worked on [figure recognition](Figures/), writing a C program to familiarise ourselves with CNN. We designed our own CNN to recognise 25x35 pixel images of figures, based on our manually created [dataset](Figures/dataset/).
To simplify the pre-processing step, we collect all the data from PNG files with the Python program [readImg.py](Figures/readImg.py) to translate them into the desired format for our C algorithm. It creates 2 CSV files : [BDD_img.csv](Figures/BDD_img.csv) and [BDD_ans.csv](Figures/BDD_ans.csv). The first one corresponds to the values of all the pixels of all the images of the dataset, and the second one to the desired response vector, filled with 0 except a 1 for the index of the figure (ex : [0,0,1,0,0,0,0,0,0,0] if the figure is a 2). This response vector is used by our algorithm to calculate the error and adjust values of all the weights of the CNN in the backpropagation step.

## Weeds Recognition
After working on the figures, we had to change our method for the agricultural application. In fact, our weed [dataset](Weeds/dataset/) (based on [4Weed Dataset](https://arxiv.org/abs/2204.00080)) is much heavier than the figure dataset. With colour images of thousands of pixels, it is impossible to use our C program, it will take too much time and power to execute.
That's why we decided to use the Python library Keras from TensorFlow.
We randomly divided our dataset into 3 folders : [train](Weeds/dataset/train/) (about 80% of the dataset), [validation](Weeds/dataset/validation/) (about 10%) and [test](Weeds/dataset/test/) (about 10%). Each folder is used in a different step of the process.

### Use
First, you have to create what is called a model in Keras. This model is too heavy to upload to this repository, so you'll have to generate it yourself. You just have to run [modelGeneration.py](Weeds/modelGeneration.py), with the parameters of your choice (batch_size, epochs, IMG_HEIGHT, IMG_WIDTH). Depending on these parameters, it may take a long time to train the CNN. When the processing is finished, a new file called *model.h5* will be created.
You can now proceed with the recognition process.

For this part, you need to run [modelExecution.py](Weeds/modelExecution.py) with the same value for IMG_HEIGHT and IMG_WIDTH. The test process will use the [test dataset](Weeds/dataset/test/) and the model generated earlier. For each image in this dataset, it will return a prediction of this type :

```
The image file_name.png is a predicted_class with a probability of p %.
```

We have also created a basic graphical interface to demonstrate our results : [showPrediction.py](Weeds/showPrediction.py). It randomly selects an image from our dataset and shows the prediction in this form :

![image](https://user-images.githubusercontent.com/103205458/229785655-e344d125-e0ab-4ac3-994f-ed87285ccd82.png)
