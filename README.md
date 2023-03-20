# Image Recognition
Image recognition project based on CNN for farming application.

## Figures Recognition
We first worked on [figure recognition](Figures/), writing a C program to familiarise ourselves with CNN. We designed our own CNN to recognise 25x35 pixel images of figures, based on our manually created [dataset](Figures/dataset/).
To simplify the pre-processing step, we collect all the data from PNG files with the Python program [readImg.py](Figures/readImg.py) to translate them into the desired format for our C algorithm. It creates 2 CSV files : [BDD_img.csv](Figures/BDD_img.csv) and [BDD_ans.csv](Figures/BDD_ans.csv). The first one corresponds to the values of all the pixels of all the images of the dataset, and the second one to the desired response vector, filled with 0 except a 1 for the index of the figure (ex : [0,0,1,0,0,0,0,0,0,0] if the figure is a 2). This response vector is used by our algorithm to calculate the error and adjust values of all the weights of the CNN in the backpropagation step.

## Weeds Recognition
After working on figures, we had to change our method for the farming application. Indeed our weeds [dataset](Weeds/dataset/) (based on [4Weed Dataset](https://arxiv.org/abs/2204.00080)) is much heavier than the figures one. With colored pictures of thousands of pixels, it is impossible to use our basic C program, it will take to much time and power to proceed.
That's why we choose to use the Python library Keras from TensorFlow.
We randomly separated our dataset into 3 folders : [train](Weeds/dataset/train/) (around 80% of the dataset), [validation](Weeds/dataset/validation/) (around 10%) and [test](Weeds/dataset/test/) (around 10%). Each folder will be used at different step of the process.

### Usage
You first have to create what is called on Keras a model. This model is too heavy to be uploaded on this repository, so you'll need to generate it by your own. You just have to execute [modelGeneration.py](Weeds/modelGeneration.py), with the parameters of your choice (batch_size, epochs, IMG_HEIGHT, IMG_WIDTH). Depending on these parameters, it could take a lot of time to train the CNN. Once the processing is done, it will create a new file called *model.h5*.
You could now go to the recognition process.

For this part, you have to execute [modelExecution.py](Weeds/modelExecution.py) with the same value for IMG_HEIGHT and IMG_WIDTH. The test process will use the [test dataset](Weeds/dataset/test/) and the model generated before. For each image in this dataset, it will answer a prediction of this type :

```
The image file_name.png is a predicted_class with a probability of p %.
```
