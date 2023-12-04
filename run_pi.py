from picamera2 import Picamera2
import time
from inference import RabbitInference

rabbit_inference = RabbitInference("rabbit_net_100.pt", jit_model=True)

cam = Picamera2()
config = cam.create_still_configuration()
cam.start()
time.sleep(1)

while True:
    im = cam.switch_mode_and_capture_image(config, "main")
    y = rabbit_inference(im)
    print(y)
    time.sleep(0.5)
