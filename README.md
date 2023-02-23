# TIE-Net
TIE-Net structure code
* If you want to use our TIENet in the network, please add the structure of TIENet.py to the front of the network backone and keep the input and output channels consistent. For example, yolo's yaml file.
* Add TIENet, TIEModule and Maskconv modules together.
* Remember to adjust the size of the input image. It is 300 * 300 in SSD. Yolov5 can scale the image adaptively.
* The yaml configuration file takes yolov5 as an example. Add the module name to the yolo.py file when using it.
* For other questions, please contact the corresponding author.
