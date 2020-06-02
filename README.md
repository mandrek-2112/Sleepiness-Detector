# Sleepiness-Detector
Uses *python3 libraries* and *DLIB* to detect **drowsiness** in a video stream.

# Organization
```bash
.
├── alarm.wav
├── detect_drowsiness.py
├── evaluate_shape_predictor.py
├── eye_predictor.dat
├── ibug_300W_large_face_landmark_dataset
│   ├── afw
│   ├── helen
│   ├── ibug
│   ├── image_metadata_stylesheet.xsl
│   ├── labels_ibug_300W_test_eyes.xml
│   ├── labels_ibug_300W_test.xml
│   ├── labels_ibug_300W_train_eyes.xml
│   ├── labels_ibug_300W_train.xml
│   ├── labels_ibug_300W.xml
│   └── lfpw
├── ibug_300W_large_face_landmark_dataset.tar.gz
├── parse_xml.py
├── predict_eyes.py
└── train_shape_predictor.py
```

# Usage
```bash
python3 detect_drowsiness.py -p eye_predictor.dat -s 0.25 -a alarm.wav
```

# Pre-requisites 
* OpenCV 
* DLIB
* Playsound
* Scipy
* Imutils

# References 
1. V. Kazemi and J. Sullivan, "One millisecond face alignment with an ensemble of regression trees," 
   2014 IEEE Conference on Computer Vision and Pattern Recognition, Columbus, OH, 2014, pp. 1867-1874.
2. Soukupová, Tereza and Jan Cech. “Eye-Blink Detection Using Facial Landmarks.” (2016).
3. pyimagesearch.com

