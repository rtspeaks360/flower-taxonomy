# AI application for indentifying Flower Taxanomy

Image classifier to recognize different species of flowers using Convolution Neural Networks.

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, we'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we would train this classifier, then export it for use in our application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 


![Flowers](https://raw.githubusercontent.com/rtspeaks360/flower-taxonomy/master/assets/Flowers.png)

The main things that we need to do here are:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

All these tasks are initially covered in the jupyter notebook. Apart from covering these parts in the jupyter notebook I also develop a command line application which can be used to train the model based on different architectures available in the `torchvision.models`. 

## Usage

* Training the model:

      `python train.py data-directory --args`

  Following are the arguments for `train.py` script:
  
  * `data_directory` - The path of directory where the dataset is
  * `--in_file` -  The name of the label map json file. stored in the working directory
  * `--save_dir` - Directory to store the models in.
  * `--batch_size` - minibatch size
  * `--arch` - architecture. Available choices are vgg11, vgg13, vgg16, vgg19
  * `--num_epochs` - Number of epochs
  * `--print_every` - Print the state after every n training steps
  * `--learning_rate` - Learning Rate
  * `--dropout_prob` - Dropout probability
  * `--gpu` - GPU Flag
  * `--init_from` - Incase you want to start from a checkpoint, path of the checkpoint

* Using the trained model to predict categories for images.

       `python predict.py input --args`
  
  Following are the arguments for the `predict.py` script
  
  * `input` - Path for the input image file.
  * `checkpoint` -  Path for the pytorch checkpoint you want to use to predict the result.
  * `--GPU` - GPU Flag
  * `--top_k` - K is the number of top categories you want to know about.
  * `--category_names` - Path to JSON file mapping categories to names
  
## Model in action

Here is the results produced by the model when the following image was given to it as an input.



