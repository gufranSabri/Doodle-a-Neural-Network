import os
import cv2

train_folder = r"C:\Users\gufra\Desktop\Work\Projects\MultiThemed\CamNN\Datasets\flowchartd1_arrows_yolov8\train"
valid_folder = r"C:\Users\gufra\Desktop\Work\Projects\MultiThemed\CamNN\Datasets\flowchartd1_arrows_yolov8\valid"

train_annot_folder = os.path.join(train_folder, "labels")
valid_annot_folder = os.path.join(valid_folder, "labels")

for img_file in os.listdir(os.path.join(train_folder, "images")):
    img_path = os.path.join(train_folder, "images", img_file)
    img = cv2.imread(img_path)

    annot_file = os.path.splitext(img_file)[0] + ".txt"
    annot_path = os.path.join(train_annot_folder, annot_file)
    with open(annot_path, "r") as f:
        annots = f.readlines()

    for annot in annots:
        class_id, x_center, y_center, width, height = map(float, annot.strip().split())

        x1 = int((x_center - width / 2) * img.shape[1])
        y1 = int((y_center - height / 2) * img.shape[0])
        x2 = int((x_center + width / 2) * img.shape[1])
        y2 = int((y_center + height / 2) * img.shape[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
