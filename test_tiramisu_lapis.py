import torch
from torch.nn import BatchNorm2d
from torch.autograd import Variable
from model.tiramisu import Tiramisu
import numpy
import cv2
from matplotlib import pyplot
from collections import OrderedDict

def from_classes_to_color(batch_array):
    """

    :param batch_array: array di dimensione BxCxWxH 
    :return: 
    """

    batch_array = torch.squeeze(batch_array, 1).numpy()
    # LABEL COLOR
    label_c = numpy.zeros((len(batch_array), 224, 224, 3))
    for index_color, (name, color) in enumerate(colors.items()):
        # devo metterlo in scala
        color = numpy.array(color[0:3]) * 255

        color = color.astype("int32")

        # devo mettere in channel last

        # trovo maschera
        mask = batch_array == index_color
        label_c[mask] = color
        # mi serve sapere il numero di elementi a True
        # out_c[mask] = numpy.repeat(color,numpy.sum(mask[:,0]))
    label_c = numpy.transpose(label_c, (0, 3, 1, 2))
    return label_c


LABELS = OrderedDict([
    ("bolt", (1.0, 0.0, 0.0, 1)),
    ("cup", (1.0, 0.501960754395, 0.501960754395, 1)),
    ("hex", (1.0, 0.433333396912, 0.0, 1)),
    ("mouse", (1.0, 0.717777848244, 0.501960754395, 1)),
    ("pen", (1.0, 0.866666734219, 0.0, 1)),
    ("remote", (1.0, 0.933594822884, 0.501960754395, 1)),
    ("scrissor", (0.699999928474, 1.0, 0.0, 1)),
    ("washer", (0.850588202477, 1.0, 0.501960754395, 1)),
    ("bottle", (0.266666531563, 1.0, 0.0, 1)),
    ("fork", (0.634771168232, 1.0, 0.501960754395, 1)),
    ("keys", (0.0, 1.0, 0.16666674614, 1)),
    ("nails", (0.501960754395, 1.0, 0.584967374802, 1)),
    ("plate", (0.0, 1.0, 0.600000143051, 1)),
    ("screw", (0.501960754395, 1.0, 0.800784349442, 1)),
    ("spoon", (0.0, 0.966666460037, 1.0, 1)),
    ("wrench", (0.501960754395, 0.983398616314, 1.0, 1)),
    ("cellphone", (0.0, 0.533333063126, 1.0, 1)),
    ("hammer", (0.501960754395, 0.767581582069, 1.0, 1)),
    ("knife", (0.0, 0.0999999046326, 1.0, 1)),
    ("nut", (0.501960754395, 0.551764667034, 1.0, 1)),
    ("pliers", (0.333333492279, 0.0, 1.0, 1)),
    ("screwdriver", (0.667973935604, 0.501960754395, 1.0, 1)),
    ("tootbrush", (0.766666889191, 0.0, 1.0, 1)),
])

LABELS_BACK = OrderedDict([
    ("back", (0, 0, 0, 1)),
    ("table", (1, 1, 1, 1))
])

WEIGHTS = OrderedDict([
    ("back", 0.2),
    ("table", 0.002),
    ("bolt", 0.1),
    ("cup", 0.05),
    ("hex", 0.1),
    ("mouse", 0.05),
    ("pen", 0.1),
    ("remote", 0.025),
    ("scrissor", 0.025),
    ("washer", 0.05),
    ("bottle", 0.025),
    ("fork", 0.025),
    ("keys", 0.05),
    ("nails", 0.025),
    ("plate", 0.025),
    ("screw", 0.05),
    ("spoon", 0.025),
    ("wrench", 0.025),
    ("cellphone", 0.025),
    ("hammer", 0.025),
    ("knife", 0.025),
    ("nut", 0.1),
    ("pliers", 0.025),
    ("screwdriver", 0.025),
    ("tootbrush", 0.025),

])


colors = [(key, val) for key, val in LABELS_BACK.items()]
colors.extend([(key, val) for key, val in LABELS.items()])

colors = OrderedDict(colors)


model = Tiramisu(in_features=3,num_classes=25,layers=(4,5,7,10),bottleneck=15,compress=False).cuda()
model.load_state_dict(torch.load("models/Oct20_15-21-29_lapis-ml/Oct21_13-03-09_0.0360020418404_"))
model.eval()
#batch_norm fa andare tutto a merda
[i.train() for i in model.modules() if isinstance(i,BatchNorm2d)]

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()

    cv2.imshow('my webcam_original', numpy.copy(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = numpy.transpose(img, (2, 0, 1))

    data = Variable(torch.FloatTensor(img[numpy.newaxis, ...]).cuda(), volatile=True)
    out = model(data)
    _, out = torch.max(out, 1)

    out = numpy.squeeze(from_classes_to_color(out.data.cpu()))
    out = numpy.transpose(out, (1, 2, 0))

    cv2.imshow('my webcam', out.astype("uint8"))
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()


