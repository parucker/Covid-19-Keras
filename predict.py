from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import cv2
import os
import random
import numpy as np
import imutils


# Config for error : Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

random.seed(1)

label = ""

# load model
model = load_model('covid19.h5')
# summarize model.
model.summary()

# take images name and dir to predict
covid = os.listdir("dataset/covid")
normal = os.listdir("dataset/normal/")
totalImages = []

for i in covid:
	name = "dataset/covid/"+i
	totalImages.append(name)
for i in normal:
	name = "dataset/normal/"+i
	totalImages.append(name)


for i in range(0,10):
	imageDir = totalImages[random.randint(0,(len(totalImages)-1))]
	
	print(imageDir)

	imageOriginal = cv2.imread(imageDir)

	image = load_img(imageDir, target_size=(224, 224))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image/255


	preds = model.predict(image)
	i = np.argmax(preds[0])
	prob = preds[0][i]*100
	if( i== 1):
		label = "Normal"
	else:
		label = "Covid-19"
	#print(label, prob)
	label = "{}: {:.2f}%".format(label, prob)
	print("[PREDICT] {}".format(label))

	imageOriginal = imutils.resize(imageOriginal, height=700)

	cv2.rectangle(imageOriginal, (0, 0), (340, 40), (0, 0, 0), -1)
	cv2.putText(imageOriginal, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)
	
	cv2.imshow("Covid Predict", imageOriginal)
	cv2.waitKey(0)