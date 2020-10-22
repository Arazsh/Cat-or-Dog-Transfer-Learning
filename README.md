# Cat or Dog-Transfer Learning with InceptionV3

In this code, a two-class model is developed to classify cat vs. dog images. For building the model, the convolutional block of InceptionV3 neural network and transfer learning with ImageNet weights is utilized. The dogs vs. cats dataset that is used for training and testing the model is provided by Kaggle at https://www.kaggle.com/c/dogs-vs-cats/data. In this code, a part of the original dataset which contains 3000 images (2000 fo training and 1000 for validation) is used. The fitted model indicates the strong accuracy of about 97% for both training and validation sets. Also, for 30 random images of dogs and cats outside the dataset, the fitted model demonstrated the perfect prediction of 100%.

## Loading InceptionV3
The first section of the code implements loading the InceptionV3 network and preventing its layers from being trained with the new data by freezing them. The InceptionV3 is called from keras applications. Since a relatively small dataset is used for this project, I used Transfer Learning by loading the weights that are acquired by training the model with ImageNet dataset. ImageNet contains over one million images in 1000 classes. Using such a big dataset with variety in images will deliver well-trained convolutional layers. The ImageNet weights can be directlly transfered while calling the model. In case transfer learning from other sources is desired, it is possible to use pre_trained_model.load_weights(local_weights_file). I sliced the InceptionV3's filter bank from the second layer (top layer excluded) to the 'mixed7' layer. 

## Building the model
For completing the model, 4 more layers are added to the convolutional layers from the previous step. The first added layer is fllatening the output of the filter bank. Then a fully connected layer with 1024 nodes, which is followed by a Dropout layer to reduce overfitting, is added. At the end, a Dense layer with sigmoid activation is added.

