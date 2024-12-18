from swapper import *

source_img = [Image.open(
    "D:/Pycharm_Project/founder_faceswap/inswapper/data/man1.jpeg")]
target_img = Image.open(
    "D:/Pycharm_Project/founder_faceswap/inswapper/data/man2.jpeg")

model = "../checkpoints/inswapper_128.onnx"
result_image = process(source_img, target_img, -1, -1, model)
result_image.save("result.png")
