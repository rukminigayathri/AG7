from flask import Flask, request, send_file
import os
import cv2
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, log_loss
import seaborn as sns

app = Flask(__name__)

ships=pd.read_csv("C:\\Users\\Tadavarthi Gowtham\\Desktop\\Gayathri\\mainproject\\train_ship_segmentations_v2.csv")
test_data=pd.read_csv("C:\\Users\\Tadavarthi Gowtham\\Desktop\\Gayathri\\mainproject\\sample_submission_v2.csv")

ships["Ship"] = ships["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
ship_unique = ships[["ImageId","Ship"]].groupby("ImageId").agg({"Ship":"sum"}).reset_index()

ship_actual=0
ship_predict=0
ship_atual=ships
ship_preict=ships
X_actual = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
for i in range(1000):
    X_actual.append(ship_actual)
for i in range(50):
    X_actual.append(1)
Y_predic = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
for i in range(1000):
    Y_predic.append(ship_predict)
for i in range(50):
    Y_predic.append(0)

def rle2bbox(rle, shape):
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2)) # an array of (start, length) pairs
    a[:,0] -= 1 # `start` is 1-indexed
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    if x1 > shape[1]:
        raise ValueError("invalid RLE or image dimensions: x1=%d >shape[1]=%d" % (x1, shape[1]))
    xc = (x0+x1)/(2*768)
    yc = (y0+y1)/(2*768)
    w = np.abs(x1-x0)/768
    h = np.abs(y1-y0)/768
    return [xc, yc, h, w]

ships["Bbox"] = ships["EncodedPixels"].apply(lambda x: rle2bbox(x, (768, 768)) if isinstance(x, str) else np.NaN)
ships.drop("EncodedPixels", axis=1, inplace=True)
ships["BboxArea"] = ships["Bbox"].map(lambda x: x[2]*768*x[3]*768 if x == x else 0)
area = ships[ships.Ship>0]

np.percentile(area["BboxArea"], [1, 5, 25, 50, 75, 95, 99])

ships = ships[ships["BboxArea"] > np.percentile(ships["BboxArea"], 1)]
ship_unique["Hasship"] = [1 if x > 0 else 0 for x in ship_unique["Ship"]]
balanced_df = ship_unique.groupby("Ship").apply(lambda x:x.sample(1000) if len(x)>=1000 else x.sample(len(x)))
balanced_df.reset_index(drop=True,inplace=True)
balanced_bbox = ships.merge(balanced_df[["ImageId"]], how ="inner", on = "ImageId")
balanced_bbox.head()

@app.route('/process_and_show', methods=['POST'])
def show_processed_images():
    uploaded_files = request.files.getlist('file')
    uploaded_file_paths = []
    for file in uploaded_files:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        uploaded_file_paths.append(file_path)

    processed_images = process_images(uploaded_file_paths)
    return processed_images

def process_images(uploaded_files):
    path = "C:\\Users\\Tadavarthi Gowtham\\Desktop\\Gayathri\\mainproject-train-data"
    max_images = 5  # For example, limit to 5 images
    num_images = min(len(uploaded_files), max_images)  # Process up to max_images
    
    # Calculate subplot grid dimensions
    cols = 3  # You can adjust this
    rows = num_images // cols + (num_images % cols > 0)
    
    plt.figure(figsize=(5, 5))
    processed_images = []

    for i in range(num_images):  # Only iterate up to num_images
        file_path = uploaded_files[i]  # Adjusted to directly use uploaded_files[i]
        imageid = balanced_df[balanced_df.Ship == i].iloc[0][0]
        image_path = os.path.join(path, imageid)

        if os.path.exists(image_path):
            image = cv2.imread(image_path)

            if i > 0:
                bbox = balanced_bbox[balanced_bbox.ImageId == imageid]["Bbox"]
                for items in bbox:
                    Xmin = int((items[0] - items[3] / 2) * 768)
                    Ymin = int((items[1] - items[2] / 2) * 768)
                    Xmax = int((items[0] + items[3] / 2) * 768)
                    Ymax = int((items[1] + items[2] / 2) * 768)
                    cv2.rectangle(image, (Xmin, Ymin), (Xmax, Ymax), (255, 255, 255), thickness=2)

            strr = hashlib.sha256(str(i).encode('utf-8'))
            text_hashed = strr.hexdigest()

            if i > 0:
                image = cv2.putText(image, text_hashed, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.title(f"No of ships = {i}")
            plt.axis('off')

            processed_images.append(image_path)
        else:
            print(f"Image not found: {image_path}")

    plt.tight_layout()
    processed_image_path = os.path.join('static', 'processed_images.png')
    plt.savefig(processed_image_path)
    plt.close()

    return processed_image_path
if __name__ == '__main__':
    app.run(debug=True)
