from PIL import Image
import cv2
import numpy as np
import zlib

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

def display_image(frame):
    # load the camera data structure
    camera_name = dataset_pb2.CameraName.FRONT
    image = [obj for obj in frame.images if obj.name == camera_name][0]

    # convert the actual image into rgb format
    img = np.array(Image.open(io.BytesIO(image.image)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize the image to better fit the screen
    dim = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    resized = cv2.resize(img, dim)

    # display the image
    cv2.imshow("Front-camera image", resized)
    cv2.waitKey(0)