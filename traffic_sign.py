##Training code

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#####################parameters#############################
path="Trafficsigns/myData"
labelfile="Trafficsigns/labels.csv"      ###file with all names of classes

##training-testing-cross-validation
testRatio = 0.2    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation

batch_size_val=50  # how many to process together (during model training)
steps_per_epoch_val=2000  ##number of batches
epochs_val=15            ## how many iterations it will go through
imageDimesions = (32,32,3)


###################importing images #########################
##do not worry about folders and how many folders &how many classes
## once put everything in mydata --> code automatically detect how many classes & put all in one matrix
# after running: 1. detect each folder one by one
##               2. every image will have classID
# Initialize the counter for class directories


count = 0

# Create empty lists to store images and their corresponding class labels
images = []
classNo = []

# List all items (directories representing classes) in the specified path
myList = os.listdir(path)       ##myList=[1,2,3,4,5,..........,43]

print("Total Classes Detected:", len(myList))

noOfClasses = len(myList)

print("Importing Classes.....")

# Iterate over each class directory
for x in range(0, len(myList)):

    # List all image files in the current class directory
    myPicList = os.listdir(path + "/" + str(count))    ##myPicList=[image1,image2,image3,.........] in one folder

    # Iterate over each image file in the current class directory
    for y in myPicList:
        # Read the image file using OpenCV
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)

        # Append the loaded image to the images list
        images.append(curImg)

        # Append the class label to the classNo list
        classNo.append(count)

    # Print the current class directory number to show progress
    print(count, end=" ")

    # Increment the class directory counter
    count += 1

# Print a newline character to separate progress output from the next output
print(" ")

# Convert the list of images to a NumPy array for easier manipulation
images = np.array(images)

# Convert the list of class labels to a NumPy array for easier manipulation
classNo = np.array(classNo)


############################splitting data
##for training we use training data & validation
##x_train for images ,y_ for classnom
##if there is 1000 images it will split (200 images test ,180 images validation ,620 training)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)


############################Matching between the images and classnom
print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==imageDimesions)," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==imageDimesions)," The dimesionas of the Validation images are wrong "
assert(X_test.shape[1:]==imageDimesions)," The dimesionas of the Test images are wrong"


############################### READ CSV FILE
data=pd.read_csv(labelfile)
print("data shape ",data.shape,type(data))

############################### DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
##take samples from each class
num_of_samples = []
cols = 5     ###number of samples in each class
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))   ##grid rows=43,col=5
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():

        x_selected = X_train[y_train == j ]   ##select only images that belong to j index class "y_train save claessnom"
                                              ## to ensure it belong to this class already
        ##choose random image from x-selected to display it
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            ##ppends the number of selected images x_selected for class j to the num_of_samples list.
            # This keeps track of how many images are available in each class.
            num_of_samples.append(len(x_selected))

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

plt.show()

print ("Preprocessing ..........")

############################Preprocessing image #############
def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img=cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img=grayscale(img)
    img=equalize(img)
    img=img/255      # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

##the map function to apply the preprocessing function to each image in the training, validation, and test datasets.
print ("Starting preprocessing")
X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
print (" Preprocessing Done ")

cv2.imshow("Preprocessing  Images", X_train[random.randint(0, len(X_train) - 1)])  # Check preprocessing
cv2.waitKey(0)

#######################Using CNN --> Adding depth 1

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#######################Augmentation

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
##Efficiently produces data in batches, applying augmentations as needed.
##1.initialize the generator
batches = dataGen.flow(X_train, y_train, batch_size=20)
##2.fetching batch (20 images )
X_batch, y_batch = next(batches)

# TO SHOW AGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    ##Hides the axis lines and labels for a cleaner view of the images
    axs[i].axis('off')
plt.show()


##################Convert from int to one Hot encoding

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

###################Convolution Neural Network
def Model():
    no_of_filters = 60
    size_of_filter = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_of_node = 500  # NO. OF NODES IN HIDDEN LAYERS
    my_model = Sequential()
    my_model.add((Conv2D(no_of_filters, size_of_filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    my_model.add((Conv2D(no_of_filters, size_of_filter, activation='relu')))
    my_model.add(MaxPooling2D(pool_size=size_of_pool))

    my_model.add((Conv2D(no_of_filters // 2, size_of_filter2, activation='relu')))
    my_model.add((Conv2D(no_of_filters // 2, size_of_filter2, activation='relu')))
    my_model.add(MaxPooling2D(pool_size=size_of_pool))
    my_model.add(Dropout(0.5))

    my_model.add(Flatten())
    my_model.add(Dense(no_of_node, activation='relu'))   ###Fully connected network
    my_model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    my_model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    my_model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return my_model

######################Train

model = Model()
print("Summary")
print(model.summary())
## Trains the model using data generated by dataGen.flow()
history=model.fit(dataGen.flow(X_train,y_train,batch_size=batch_size_val),
                            steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
print("done")
######################plot
##1. plot history object (Training and validation loss )
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
## plot Training and validation accuracy )
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
##Evaluates the model's performance on the test dataset
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])  ## score[0]--> loss
print('Test Accuracy:', score[1])  ##score[1]-->accuracy

# Saving the model
pickle_out = open("model_trained.p", "wb")  # wb = WRITE BYTE MODE
## Serializes the model object and writes it to the file.
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)