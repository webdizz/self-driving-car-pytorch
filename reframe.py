import cv2
import logging
import numpy as np

logger = logging.getLogger('Vision')


def resize_frame(frame):
    # crop and resize by 4. If we resize directly we lose pixels that
    # aren't close enough to the pixel boundary.
    x1 = 36
    y1 = 116
    x2 = 640
    y2 = 536
    # cv2.imwrite('/tmp/universe-frame-original.jpg', processed_frame)
    processed_frame = frame[y1:y2, x1:x2]
    # reduce by 2 in 2 times
    y = processed_frame.shape[0]
    x = processed_frame.shape[1]
    ratio = 100.0 / x
    processed_frame = cv2.resize(processed_frame, (100, int(y * ratio)))
    # cv2.imwrite('/tmp/universe-frame-cropped.jpg', processed_frame)
    # after crop shape is 69x100
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('/tmp/universe-frame-gray.jpg', processed_frame)
    logger.debug("Current frame shape after preprocessing is {}".format(
        processed_frame.shape))
    processed_frame = np.reshape(processed_frame, [1, 69, 100])
    return processed_frame
