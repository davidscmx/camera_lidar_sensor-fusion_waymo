
def display_image(frame):
    # load the camera data structure
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