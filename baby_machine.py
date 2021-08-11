import cv2
import numpy as np
import os
import glob
from sklearn import tree
from fruit_feature import extract_features

# load the training dataset
train_path  = "ap-or-database/train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1

	for file in glob.glob(cur_path + "/*.jpg"):
		print ("Processing Image - {} in {}".format(i, cur_label))
		# read the training image
		image = cv2.imread(file)

		# extract texture and color from the image
		features = extract_features(image)

		# append the feature vector and label
		train_features.append(features)
		train_labels.append(cur_label)

		# show loop update
		i += 1


# create the classifier
clf=tree.DecisionTreeClassifier()

# train the classifier
print ("Training model..")
clf.fit(train_features, train_labels)

# loop over the test images
test_path = ("ap-or-database/test")
for file in glob.glob(test_path + "/*.jpg"):
	# read the input image
    image = cv2.imread(file)

		# extract texture and color from the image
    features = extract_features(image)

	# evaluate the model and predict label
    prediction = clf.predict(features.reshape(1, -1))[0]

	# show the label
    cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235,188,0), 3)
    print ("Prediction - {}".format(prediction))

	# display the output image
    cv2.imshow("Test_Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()