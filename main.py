import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from HandTracking import handTracker

global linetracking
global drawing_layer
global f_new_layer

RESOLUTION = {
    "width": 640,
    "height": 480
}
THRESHOLD = 3
PEN_RADIUS = 50


def finger():
    global linetracking
    global drawing_layer

    linetracking = np.empty((0, 2))
    drawing_layer = np.zeros(
        (RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)

    try:
        cap = cv2.VideoCapture(0)
        # set frame rate to 60 fps
        cap.set(cv2.CAP_PROP_FPS, 30)

        tracker = handTracker()
        fingersUp = []

        while True:
            success, image = cap.read()

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)

            if (len(lmList) != 0):
                fingersUp = tracker.fingersUp(image)

            if len(lmList) != 0 and fingersUp[1] == 1:
                # check if finger is in radius of previous
                if len(linetracking) < 3:
                    linetracking = np.append(
                        linetracking, [lmList[8][1:3]], axis=0)
                else:
                    if abs(lmList[8][1] - linetracking[-1][0]) > THRESHOLD or abs(lmList[8][2] - linetracking[-1][1]) > THRESHOLD:
                        linetracking = np.append(
                            linetracking, [lmList[8][1:3]], axis=0)

            if len(linetracking) > 3:
                # Generate some random points
                points = np.array(linetracking, dtype=np.int32)
                epsilon = 0.1 * cv2.arcLength(points, closed=True)

                # REALLY SLOW INTERPOLATION
                contour = np.array(points, dtype=np.float64)

                num_points = len(contour) * 10

                # Perform spline interpolation
                tck, u = splprep(contour.T, s=0, per=False)
                u_new = np.linspace(u.min(), u.max(), num_points)
                x_new, y_new = splev(u_new, tck)

                img = np.zeros(
                    (RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)
                for i in range(num_points-1):
                    cv2.line(img, (int(x_new[i]), int(y_new[i])),
                             (int(x_new[i+1]), int(y_new[i+1])), (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

                drawing_layer = img
                # pass

            image = cv2.addWeighted(image, 0.8, drawing_layer, 0.2, 0)
            cv2.imshow("Video", image)
            cv2.waitKey(1)

    except Exception as e:
        print(e)


def process_frame(frame):
    try:
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    except Exception as e:
        return frame, None, None

    # Define cursor X, Y
    cX = None
    cY = None

    # Define the color range for the pen tip
    lower_blue = np.array([100, 0, 63])
    upper_blue = np.array([130, 255, 255])

    # Create a mask using the color range
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assuming it's the pen tip)
    max_area = 0
    pen_tip_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            pen_tip_contour = contour

    if pen_tip_contour is not None:
        # Find the center of the contour (pen tip)
        M = cv2.moments(pen_tip_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Draw a circle at the pen tip on the frame
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    else:
        # If no pen tip is found, return black frame
        return np.zeros(frame.shape, dtype=np.uint8), None, None

    return frame, cX, cY


def pen():
    global linetracking
    global drawing_layer
    global f_new_layer
    global f_layer_timeout

    f_new_layer = True
    f_layer_timeout = 0

    linetracking = np.empty((0, 2))
    layers = np.empty((0, RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)
    drawing_layer = np.zeros(
        (RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)

    try:
        cap = cv2.VideoCapture(0)
        # set frame dimensions to 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION["height"])
        # set frame rate to 60 fps
        cap.set(cv2.CAP_PROP_FPS, 30)

        tracker = handTracker()
        fingersUp = []

        while True:
            success, image = cap.read()

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)

            if (len(lmList) != 0):
                fingersUp = tracker.fingersUp(image)

            if len(lmList) != 0:
                sub_image = np.zeros(
                    (PEN_RADIUS*2, PEN_RADIUS*2, 3), dtype=np.uint8)

                sub_image = image[lmList[8][2]-PEN_RADIUS:lmList[8][2] +
                                  PEN_RADIUS, lmList[8][1]-PEN_RADIUS:lmList[8][1]+PEN_RADIUS]

                pX = None
                pY = None

                if (sub_image.shape[0] == PEN_RADIUS*2 and sub_image.shape[1] == PEN_RADIUS*2):
                    marked_image, pX, pY = process_frame(sub_image)

                    if pX is None and pY is None and fingersUp[1] == 1:
                        if f_layer_timeout < 5:
                            f_layer_timeout += 1
                        else:
                            f_new_layer = True
                            f_layer_timeout = 0

                    elif pX is not None and pY is not None:
                        originalX = lmList[8][1] - PEN_RADIUS + pX
                        originalY = lmList[8][2] - PEN_RADIUS + pY

                        if len(linetracking) < 1:
                            linetracking = np.append(
                                linetracking, [[originalX, originalY]], axis=0)

                        if abs(originalX - linetracking[-1][0]) > THRESHOLD or abs(originalY - linetracking[-1][1]) > THRESHOLD:
                            linetracking = np.append(
                                linetracking, [[originalX, originalY]], axis=0)

                    cv2.imshow("Finger Cam", marked_image)
                    cv2.waitKey(1)

            if len(linetracking) > 3:
                points = np.array(linetracking, dtype=np.int32)
                contour = np.array(points, dtype=np.int32)
                num_points = len(contour) * 10

                # Perform spline interpolation
                try:
                    tck, u = splprep(contour.T, s=0, per=False)
                    u_new = np.linspace(u.min(), u.max(), num_points)
                    x_new, y_new = splev(u_new, tck)

                    img = np.zeros(
                        (RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)
                    for i in range(num_points-1):
                        try:
                            cv2.line(img, (int(x_new[i]), int(y_new[i])),
                                    (int(x_new[i+1]), int(y_new[i+1])), (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
                        except Exception as e:
                            print(e)
                            pass

                    drawing_layer = img
                except Exception as e:
                    print("Error raised in spline interpolation!")
                    print(e)
                    print("Dumping line tracking array...")
                    print(linetracking)
                    linetracking = np.empty((0, 2))
                    continue

                if f_new_layer == True:
                    layers = np.concatenate(
                        (layers, [drawing_layer]), axis=0)
                    new_layers = np.empty(
                        (0, RESOLUTION["width"], RESOLUTION["height"], 3), dtype=np.uint8)
                    
                    layers = np.concatenate((layers, new_layers), axis=0)

                    f_new_layer = False
                    linetracking = np.empty((0, 2))
                else:
                    layers[-1] = drawing_layer
            else:
                # draw without spline interpolation
                if len(linetracking) > 1:
                    for i in range(len(linetracking)-1):
                        cv2.line(drawing_layer, (int(linetracking[i][0]), int(linetracking[i][1])),
                                 (int(linetracking[i+1][0]), int(linetracking[i+1][1])), (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

            for layer in layers:
                image = cv2.addWeighted(image, 1.0, layer, 0.5, 0)

            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)


def main():
    # finger()
    try:
        pen()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
