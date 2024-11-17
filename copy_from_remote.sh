#!/bin/bash

scp leemd1@ama:~/deepfake-detection/data/util/cocofake.py ./data/util/cocofake.py
scp -r leemd1@ama:~/deepfake-detection/models/mobilenetv2/* ./models/mobilenetv2/
scp leemd1@ama:~/deepfake-detection/tests/* ./tests/

