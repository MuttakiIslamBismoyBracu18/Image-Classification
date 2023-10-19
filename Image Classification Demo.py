#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)


# In[3]:


import cv2
import imghdr


# In[4]:


data_dir = "C:/Users/Bismoy/Pictures/DNR"


# In[5]:


os.listdir(data_dir)


# In[6]:


from matplotlib import pyplot as plt

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)


# In[7]:


img = cv2.imread(os.path.join('C://','Users','Bismoy','Pictures','DNR','Clean','IMG_0096.jpg'))
#assert not isinstance(img,type(None)), 'image not found'

print(img)
#img.shape


# In[8]:


img.shape


# In[9]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show() #ommits the line above the picture


# In[10]:


image_exts = ['JPG','jpg','png','jpeg']


# In[11]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
        except Exception as e:
            print('Issue with image {}'.format(image_path))


# In[12]:


import numpy as np
from matplotlib import pyplot as plt
data = tf.keras.utils.image_dataset_from_directory('C:/Users/Bismoy/Pictures/DNR') #Convert my dataset under a variable & preprocess it a bit


# In[13]:


#tf.keras.utils.image_dataset_from_directory?? (Shortly preprocess the model)
#data = tf.keras.utils.image_dataset_from_directory('C:/Users/Bismoy/Pictures/DNR', batch_size = 8, image_size=(128,128)) #if we want to make it more short


# In[14]:


data_iterator = data.as_numpy_iterator() #so that we can get the data anytime


# In[15]:


data_iterator

# print(batch)


# In[16]:


#batch = []
batch = data_iterator.next()


# In[17]:


#print(batch)


# In[18]:


#batch[0].shape


# In[19]:


fig, ax = plt.subplots(ncols=4, figsize = (20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[20]:


#PREPROCESSING


# In[21]:


data = data.map(lambda x,y : (x/255, y)) 
#keeping the image batch between 0 and 1 so that I can fit any deep learning model perfectly
#data map helps to pull the information into the pipeline


# In[22]:


scaled_iterator = data.as_numpy_iterator()


# In[23]:


batch = scaled_iterator.next()


# In[24]:


batch[0].max()
#kept the value between 0 and 1


# In[25]:


fig, ax = plt.subplots(ncols=4, figsize = (20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
#rechecking another batch just to be sure if the image is still rgb after redeeming the lambda value


# In[26]:


#len(data)


# In[27]:


train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)


# In[28]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[29]:


#BUILDING THE MODEL - Sequential model - CNN - Fully Connected


# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[31]:


model = Sequential()


# In[32]:


model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3 Convolution block, 1 Flatten, 1 Dense layer
# first layer will be a input layer and add 16 filters and extract relevant info. 
# Filters are gonna be 3*3 pixels of each stride
# Takes maxmimum value from a 2,2 region
# Increase the filters by making it double and then again taking 16 filters
# Flatten makes it as a single output (Dimension in this case)
# then fully connected layers - 2 of them
# 256 neurons in Relu and a single output at sigmoid between 0 and 1
# relu Activation = any output that were preserved below 0 will be 0 so we can take all the positive value
# sigmoid Activation = reshaping the functions as a Sigmoid


# In[33]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# ADAM optimizer
# Accuracy will tell us how our model are gonna perform between 0 and 1


# In[34]:


model.summary()


# In[35]:


logdir = "C:/Users/Bismoy/Pictures/Logs"


# In[36]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[37]:


#TRAIN

hist = model.fit(train, epochs=32, validation_data=val, callbacks=[tensorboard_callback])

#hist2 = model.fit(train, y_train, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=val, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

# callback logs all the information from tensorboard and put it in that folder


# In[38]:


fig = plt.figure()

plt.plot(hist.history['loss'], color='green', label='loss')
plt.plot(hist.history['val_loss'],color='red',label='val_loss')
fig.suptitle('Loss',fontsize=14)
plt.legend(loc="upper left")

plt.show()


# In[39]:


fig = plt.figure()

plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=14)
plt.legend(loc="upper left")

plt.show()


# In[40]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[41]:


pre = Precision()
re= Recall()
acc = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    
print(pre.result().numpy())


# In[42]:


img_ef = cv2.imread('C:/Users/Bismoy/Pictures/d.JPG')

plt.imshow(cv2.cvtColor(img_ef, cv2.COLOR_BGR2RGB))
plt.show()


# In[43]:


resize = tf.image.resize(img_ef, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[44]:


resize.shape


# In[45]:


np.expand_dims(resize,0).shape


# In[46]:


yhat = model.predict(np.expand_dims(resize/255,0))


# In[47]:


if yhat> 0.45:
    print("The picture has Oak Wilt Trees")
else:
    print("The Picture Does not any diseased Trees")


# In[48]:


from tensorflow.keras.models import load_model


# In[49]:


model.save(os.path.join('C:/Users/Bismoy/Pictures/Models','oak_wilt_prediction.h5'))


# In[ ]:




