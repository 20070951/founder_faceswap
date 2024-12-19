from inswapper.restoration import *
from PIL import Image


class FounderRestoration:
    def __init__(self, model_dir):
        check_ckpts(model_dir)
        self.codeformer_path = os.path.join(
            model_dir, 'CodeFormer/codeformer.pth')
        self.facelib_detection_path = os.path.join(
            model_dir, 'facelib/detection_mobilenet0.25_Final.pth')
        self.facelib_parsing_path = os.path.join(
            model_dir, 'facelib/parsing_parsenet.pth')
        self.realesrgan_path = os.path.join(
            model_dir, 'realesrgan/RealESRGAN_x2plus.pth')

        self.upsampler = set_realesrgan(self.realesrgan_path)
        self.device = torch.device("cpu")
        # self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
        #                                                       codebook_size=1024,
        #                                                       n_head=8,
        #                                                       n_layers=9,
        #                                                       connect_list=[
        #                                                           "32", "64", "128", "256"],
        #                                                       ).to(self.device)
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                              codebook_size=1024,
                                                              n_head=8,
                                                              n_layers=9,
                                                              connect_list=[
                                                                  "32", "64", "128", "256"],
                                                              ).to(self.device)
        checkpoint = torch.load(self.codeformer_path)["params_ema"]
        self.codeformer_net.load_state_dict(checkpoint)
        self.codeformer_net.eval()

    def restoration_frame(self, image):
        # result_image = face_restoration(
        #     image, True, True, 2, 0.5, self.upsampler, self.codeformer_net, self.device)
        result_image = face_restoration(
            image, False, False, 1, 0.01, self.upsampler, self.codeformer_net, self.device)
        return result_image


if __name__ == "__main__":
    model_dir = 'D:/Pycharm_Project/founder_faceswap-1/checkpoints'
    restorationer = FounderRestoration(model_dir)
    image = cv2.imread('D:/Pycharm_Project/founder_faceswap-1/result.png')

    result = restorationer.restoration_frame(image)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
