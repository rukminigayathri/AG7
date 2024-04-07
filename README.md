# AG7

Identying Ships Using Satelitte Images (Deep Learning)

### Team Details :
    1. Rukmini Gayathri Tadavarthi
    2. Meda Asritha
    3. Bandi Sai Harsha
### INTRODUCTION
When using remote sensing pictures for marine security, ship detection is essential. The deep learning method for identifying ships from satellite photos is covered in this research. 
In order to achieve integrity Hashing is included. This model makes use of a supervised method for classifying images, and then use YOLOv 3 for object recognition, feature extraction from Deep-CNN.
Using class labels, semantic segmentation and picture segmentation are used to determine the object category of each pixel. 
Next, with the satellite image's bounding box is defined and helps usto identify the position ofship and ship count. 
We have implemented hashing with the help of SHA- 512. It is used to encode the Ship count and locations. A dataset of around 2,30,000 photosfrom Kaggle ship detection is used for the proposed model.
The bounding box location and the ship count are the input data used by the hash algorithm. 
In order to achieve security, we use SHA-512 which maintains security for the transmission of data.
### Dataset
https://www.kaggle.com/pesssinaluca/ship-detection-visualizations-and-eda/data
### Deployment
Link- http:/54.242.19.163:5000
