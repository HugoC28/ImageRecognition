# Image Recognition
Image recognition project based on CNN for farming application.

## Figures Recognition
We firstly worked on [Figure Recognition](Figures/), making a C program to become familiar with CNN. We desined our own CNN to recognize 25x35pixels images of figures, based on our manualy created [dataset](Figures/dataset/).
To symplify the preprocessing step, we collect all data from PNG files with the [readImg.py](Figures/readImg.py) python program to translate it to the desired format for our C algorythm. It creates 2 CSV files : [BDD_img.csv](Figures/BDD_img.csv) and [BDD_ans.csv](Figures/BDD_ans.csv). The first one correspond to values of all pixels of all images of the dataset, and the second the desired answer vector, filled by 0 execpt a 1 for the index of the figure (ex : [0,0,1,0,0,0,0,0,0,0] if the figure is a 2). This answer vector is used by our algorythm to calculate the error and adjust values of all weihts of the CNN in the backpropagation step.

## Weeds Recognition
After working on figures, we had to change our method for the farming application. Indeed our weeds [dataset](Weeds/dataset/) (based on [4Weed Dataset](https://arxiv.org/abs/2204.00080)) is much heavier than the figures one. With colored pictures of thousands of pixels, it is impossible to use our basic C program, it will take to much time and power to proceed.
That's why we choose to use keras Python library.
We randomly separated our dataset into 3 folders : [train](Weeds/dataset/train/) (around 80% of the dataset), [validation](Weeds/dataset/validation/) (around 10%) and [test](Weeds/dataset/test/) (around 10%). Each folder will be used at different step of the process.

### Usage
You first have to create what is called on Keras a model. This model is too heavy to be uploaded on this repository, so you'll need to generate it by your own. You just have to execute [modelGeneration.py](/Weeds/modelGeneration.py), with the parameters of your choice (batch_size, epochs, IMG_HEIGHT, IMG_WIDTH). Depending on these parameters, it could take a lot of time to train the CNN. Once the processing is done, it will create a new file called *model.h5*.
You could now go to the recognition process.
