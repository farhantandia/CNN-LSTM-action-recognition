import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
import time
if (len(sys.argv) == 6):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    saved_model = sys.argv[3]
    video_file = sys.argv[4]
    output_video = sys.argv[5]
else:
    print ("Usage: python clasify.py sequence_length class_limit saved_model_name video_file_name")
    print ("Example: python clasify.py 75 2 lstm-features.095-0.090.hdf5 some_video.mp4")
    exit (1)

capture = cv2.VideoCapture(os.path.join(video_file))
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(str(output_video), fourcc, 15, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

# get the model.
extract_model = Extractor(image_shape=(height, width, 3))
saved_LSTM_model = load_model(saved_model)

frames = []
frame_count = 0
while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
    start_time = time.time()
    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)

    if frame_count < seq_length:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    # For each frame extract feature and prepare it for classification
    sequence = []
    for image in frames:
        features = extract_model.extract_image(image)
        sequence.append(features)

    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
    print(prediction)
    values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # Add prediction to frames and write them to new video
    for image in frames:
        for i in range(len(values)):
            cv2.putText(image, values[i], (20, 20 * i + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)
        video_writer.write(image)

        result = np.asarray(image)
    frames = []
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output Video", result)
video_writer.release()
