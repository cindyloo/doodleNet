excerpted and adapted from [@yining1023](https://github.com/yining1023) and (@zaidalyafeai](https://github.com/zaidalyafeai) # DoodleNet 

I updated to tensorflow 2.0 and partitioned the training and reloading/testing the model, which is any subset as specified in classNames.txt



## Get started
memory constraints: I had a serious memory issue downloading the QuickDraw dataset npy files. Fortunately we have some machines at MIT that can handle the load (you'd hope, right?).

specify the files you want to download in classnames (see the full set at Quickdraw), and then use doodleNet.py to create your model. It will save it as a "keras.h5" file. DoodleTrain and DoodleTest load that file and train/test as many times as you want. If I did it right, I got up to 96% accuracy.


