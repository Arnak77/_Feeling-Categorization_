#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os



# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


img=image.load_img(r"D:\NIT\DATASCIENCE\ARNAK TASK\CNN\tra\sad\ima 2.jpg")


# In[7]:


plt.imshow(img)


# In[8]:


i1 = cv2.imread(r"D:\NIT\DATASCIENCE\ARNAK TASK\CNN\tra\sad\ima 2.jpg")
i1


# In[9]:


i1.shape


# In[10]:


train = ImageDataGenerator(rescale = 1/255)
validataion = ImageDataGenerator(rescale = 1/255)


# In[11]:


train_dataset = train.flow_from_directory(r"D:\NIT\DATASCIENCE\ARNAK TASK\CNN\tra",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')


# In[12]:


validataion_dataset = validataion.flow_from_directory(r"D:\NIT\DATASCIENCE\ARNAK TASK\CNN\vali",
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')


# In[13]:


train_dataset.class_indices


# In[14]:


train_dataset.classes


# In[15]:


model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),    
                                    #                       
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2), 
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )


# In[16]:


model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy']
              )


# In[63]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 25,
                     validation_data = validataion_dataset)


# In[40]:


dir_path = r'D:\NIT\DATASCIENCE\ARNAK TASK\CNN\test'
for i in os.listdir(dir_path ):
    print(i)


# In[41]:


dir_path = r'D:\NIT\DATASCIENCE\ARNAK TASK\CNN\test'


for filename in os.listdir(dir_path):
    img_path = os.path.join(dir_path, filename)
    img = image.load_img(img_path, target_size=(200, 200))
    plt.imshow(img)
    plt.axis('off')  # Disable axis
    plt.show()


# In[42]:


dir_path = r'D:\NIT\DATASCIENCE\ARNAK TASK\CNN\test'


image_filenames = os.listdir(dir_path)

num_images = len(image_filenames)
fig, axes = plt.subplots(1, num_images, figsize=(20, 5))


for i, filename in enumerate(image_filenames):
    img_path = os.path.join(dir_path, filename)
    img = image.load_img(img_path, target_size=(200, 200))
    axes[i].imshow(img)
    axes[i].axis('off')  # Disable axis
    axes[i].set_title(filename)

plt.tight_layout()
plt.show()


# In[44]:


dir_path = r'D:\NIT\DATASCIENCE\ARNAK TASK\CNN\test'

for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
        
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0:
        print( ' i am  happy')
    else:
        print('i am not happy')


# In[45]:


dir_path = r'D:\NIT\DATASCIENCE\ARNAK TASK\CNN\test'


plt.figure(figsize=(15, 15))
columns = 3
rows = len(os.listdir(dir_path)) // columns + 1

for i, filename in enumerate(os.listdir(dir_path)):
    img_path = os.path.join(dir_path, filename)
    img = image.load_img(img_path, target_size=(200, 200))
    plt.subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.axis('off')  # Disable axis

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    val = model.predict(images)
    if val == 0:
        prediction = 'I am happy'
    else:
        prediction = 'I am not happy'
    plt.title(prediction)

plt.tight_layout()
plt.show()


# In[ ]:




