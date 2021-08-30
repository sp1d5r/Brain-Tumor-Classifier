# Brain Tumor Classifier

I want to make a lung cancer classifier, but I couldn't find a decent dataset. Also my computer doesn't 
have a ton of memory left from other projects I've been building. So while I wait for my virtual machine 
to get ready I will do some practice on this brain tumor dataset.

The dataset can be found [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

The dataset is considerably small. There are only 3250 images in total, there are four different types of brain tumors
recorded - glioma tumor, meningioma tumor, and pituitary tumor. As a control group I've got a no tumor group. If you 
look at the files you'll be able to see an obvious difference between the MRI scans with a brain tumor and the ones 
without, but because i don't have any knowledge on what it's supposed to look like i'm not sure how long it will take.

## Data Pre-Processing 
I separated out all the files into an array with the following types: 
[image, category]. Then created the following indexes, Y = Category and 
X = image. Since the model requires us to squash down the images i reshaped 
it. 

```python
training_data = []

def populate_training_data():
    for category in CATEGORIES:
        new_path = os.path.join(TRAINING_PATH, category)
        category_index = CATEGORIES.index(category)
        
        for img in os.listdir(new_path):
            try:
                img_array = cv2.imread(os.path.join(new_path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) 
                training_data.append([new_array,category_index])
            except Exception as e:
                print("failed", e)
                pass

# This takes the files in the training path and places them in this order [image, int] 
# where image = the image data, and int is the category it's in.
# [glioma_tumor = 0, meningioma_tumor = 1, no_tumor = 2, pituitary_tumor = 3]
populate_training_data()

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
# this is used to reshape and flatten the data.
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE)
X = X/255.0 
X = X.reshape(-1,150,150,1)
```

Like that!

## Model Creation
Then i used tensorflow's keras model to create a Convolutional Neural Network, with a couple layers.

```python
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
y = to_categorical(y, num_classes = 4)

# Using a Sequential Model 
model = Sequential()

# Applying a convlutional layer
model.add(Conv2D(64, (5,5), input_shape = (150,150,1)))    # could be X.shape[1:]
model.add(Activation('relu')) # you could pass activation/pooling in whatever order
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), input_shape = (150,150,1)))    # could be X.shape[1:]
model.add(Activation('relu')) # you could pass activation/pooling in whatever order
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(4, activation = "softmax"))


model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 5 
batch_size = 40

model.fit(X_train, Y_train, batch_size=batch_size, epochs = epochs, validation_data = (X_val,Y_val))
```
```xml
Epoch 1/5
58/58 [==============================] - 36s 599ms/step - loss: 0.9557 - accuracy: 0.5858 - val_loss: 0.6231 - val_accuracy: 0.7544
Epoch 2/5
58/58 [==============================] - 35s 611ms/step - loss: 0.5196 - accuracy: 0.7979 - val_loss: 0.4790 - val_accuracy: 0.7892
Epoch 3/5
58/58 [==============================] - 33s 572ms/step - loss: 0.3134 - accuracy: 0.8794 - val_loss: 0.3965 - val_accuracy: 0.8397
Epoch 4/5
58/58 [==============================] - 34s 587ms/step - loss: 0.1901 - accuracy: 0.9303 - val_loss: 0.4452 - val_accuracy: 0.8397
Epoch 5/5
58/58 [==============================] - 33s 577ms/step - loss: 0.1204 - accuracy: 0.9608 - val_loss: 0.3579 - val_accuracy: 0.8746
```

## Results! 
The model I trained here has an accuracy of 87.46%.

## Social Media 
these are my social media's, stay tuned because I will publish the source code to the predecesor of the mechanics calculator, the statistics calculator, this app managed to gain over 3000+ app installs. 
- [Linkden - Elijah Ahmad](https://www.linkedin.com/in/elijah-ahmad-658a2b199/)
- [FaceBook - Elijah Ahmad](https://www.facebook.com/elijah.ahmad.71)
- [Instagram - @ElijahAhmad__](https://www.instagram.com/ElijahAhmad__)
- [Snapchat - @Elijah.Ahmad](https://www.snapchat.com/add/elijah.ahmad)