from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from facelib.utils.misc import is_gray
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import imwrite, img2tensor, tensor2img
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torch
import cv2
import os
import sys
sys.path.append('./CodeFormer/CodeFormer')


def check_ckpts(model_dir='CodeFormer/CodeFormer/weights'):
    pretrain_model_url = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        # 'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }
    # download weights
    codeformer_path = os.path.join(model_dir, 'CodeFormer/codeformer.pth')
    facelib_detection_path = os.path.join(
        model_dir, 'facelib/detection_mobilenet0.25_Final.pth')
    facelib_parsing_path = os.path.join(
        model_dir, 'facelib/parsing_parsenet.pth')
    realesrgan_path = os.path.join(
        model_dir, 'realesrgan/RealESRGAN_x2plus.pth')

    if not os.path.exists(codeformer_path):
        load_file_from_url(
            url=pretrain_model_url['codeformer'], model_dir=os.path.dirname(codeformer_path), progress=True, file_name=None)
    if not os.path.exists(facelib_detection_path):
        load_file_from_url(
            url=pretrain_model_url['detection'], model_dir=os.path.dirname(facelib_detection_path), progress=True, file_name=None)
    if not os.path.exists(facelib_parsing_path):
        load_file_from_url(
            url=pretrain_model_url['parsing'], model_dir=os.path.dirname(facelib_parsing_path), progress=True, file_name=None)
    if not os.path.exists(realesrgan_path):
        load_file_from_url(
            url=pretrain_model_url['realesrgan'], model_dir=os.path.dirname(realesrgan_path), progress=True, file_name=None)

# set enhancer with RealESRGAN


def set_realesrgan(model_path="CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth"):
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


def face_restoration(img, background_enhance, face_upsample, upscale, codeformer_fidelity, upsampler, codeformer_net, device):
    """Run a single prediction on the model"""
    try:  # global try
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        upscale = int(upscale)  # convert type to int
        if upscale > 4:  # avoid memory exceeded due to too large upscale
            upscale = 4
        # avoid memory exceeded due to too large img resolution
        if upscale > 2 and max(img.shape[:2]) > 1000:
            upscale = 2
        # avoid memory exceeded due to too large img resolution
        if max(img.shape[:2]) > 1500:
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(
                        output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None, None
