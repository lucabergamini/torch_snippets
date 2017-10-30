import os
import shutil

BASE_FOLDER = "/home/lapis-ml/Desktop/LAPIS_dataset"

folders = [os.path.join(BASE_FOLDER,i) for i in os.listdir(BASE_FOLDER) if "train" in i]

DEST_DATA = os.path.join(BASE_FOLDER,*["train","data"])
DEST_LABELS_RAW = os.path.join(BASE_FOLDER,*["train","labels_raw"])
DEST_LABELS_CLEAN = os.path.join(BASE_FOLDER,*["train","labels_clean"])
DEST_LABELS = os.path.join(BASE_FOLDER,*["train","labels"])

os.makedirs(DEST_DATA)
os.makedirs(DEST_LABELS)
os.makedirs(DEST_LABELS_CLEAN)
os.makedirs(DEST_LABELS_RAW)
index = 0

for folder in folders:
    base_data = os.path.join(folder,"data")
    base_labels = os.path.join(folder,"labels")
    base_labels_clean = os.path.join(folder,"labels_clean")
    base_labels_raw = os.path.join(folder,"labels_raw")

    data = [os.path.join(base_data,i) for i in sorted(os.listdir(base_data))]
    labels_raw = [os.path.join(base_labels_raw,i) for i in sorted(os.listdir(base_labels_raw))]
    labels_clean =  [os.path.join(base_labels_clean,i) for i in sorted(os.listdir(base_labels_clean))]
    labels = [os.path.join(base_labels,i) for i in sorted(os.listdir(base_labels))]
    for d,l,lr,lc in zip(data,labels,labels_raw,labels_clean):
        shutil.copy(d,os.path.join(DEST_DATA,"{}.png".format(index)))
        shutil.copy(l,os.path.join(DEST_LABELS,"{}.png".format(index)))
        shutil.copy(lc,os.path.join(DEST_LABELS_CLEAN,"{}.png".format(index)))
        shutil.copy(lr,os.path.join(DEST_LABELS_RAW,"{}.png".format(index)))
        index +=1

