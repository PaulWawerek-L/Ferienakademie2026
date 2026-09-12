"""Task 7 -- inference time per project.

One project per invocation, because each lives in its own container:

  python exp7_timing.py --project image-compression
  python exp7_timing.py --project video-compression
  python exp7_timing.py --project hybrid-vtm
  python exp7_timing.py --project style-transfer
  python exp7_timing.py --project super-resolution

Each writes outputs/experiments/07-timing/<project>.json. `run_timing.sh`
dispatches all five; `make_figures.py timing` draws the chart.

Inference only: model construction, checkpoint loading and file I/O sit outside
the timed region. Every project is timed on its own natural input, and the input
and output pixel counts are recorded so the chart can also show throughput --
seconds alone would compare a 512x768 image against a 128x128 crop.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_common import bench, env_info, save

OUT = "/work/outputs/experiments/07-timing"
KODIM = "/work/Dataset/kodim19.png"
SEQ = "/work/Dataset/RaceHorses_416x240_30.yuv"
WEIGHTS = "/work/weights"


def t_image_compression():
    """DCVC-UF-Intra: analysis transform -> hyperprior -> entropy model -> synthesis."""
    import torch
    from dcvc_yuv import eval_rate, pad, padding_for, read_yuv420, to_model_input
    from src.models.image_model import DMCI
    from src.utils.common import get_state_dict

    W, H = 512, 768
    ref = "/work/outputs/experiments/kodim19_420.yuv"
    if not os.path.exists(ref):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                        KODIM, "-pix_fmt", "yuv420p", "-f", "rawvideo", ref], check=True)

    net = DMCI().eval()
    net.load_state_dict(get_state_dict(f"{WEIGHTS}/dcvc/cvpr2026_image.pth.tar"))
    x = to_model_input(read_yuv420(ref, W, H, 1)[0][0])
    pad_r, pad_b = padding_for(H, W)
    x = pad(x, pad_r, pad_b)
    qp = torch.tensor([32])

    def run():
        with torch.no_grad(), eval_rate():
            net.forward_one_frame(x, qp)

    print(f"DCVC-UF-Intra, kodim19 {W}x{H} (padded {W + pad_r}x{H + pad_b}), qp 32")
    stats = bench(run, label="forward (encode + decode)")
    return {"label": "01 Image compression\n(DCVC-UF-Intra)", "unit": "per image",
            "input_px": W * H, "output_px": W * H,
            "detail": f"kodim19 {W}x{H}", "timing": stats}


def t_video_compression():
    """DCVC-UF HTS: one chunk of 8 inter frames, reported per frame."""
    import torch
    import torch.nn.functional as F
    from dcvc_yuv import eval_rate, pad, padding_for, read_yuv420, to_model_input
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, g_frame_delay
    from src.utils.common import get_state_dict, ModelStructure

    W, H = 416, 240
    frames = read_yuv420(SEQ, W, H, g_frame_delay + 1)
    pad_r, pad_b = padding_for(H, W)
    qp = torch.tensor([32])

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(f"{WEIGHTS}/dcvc/cvpr2026_image.pth.tar"))
    p_net = DMC(model_structure=ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(f"{WEIGHTS}/dcvc/cvpr2026_video_hts.pth.tar"))

    with torch.no_grad(), eval_rate():
        out = i_net.forward_one_frame(pad(to_model_input(frames[0][0]), pad_r, pad_b), qp)
    p_net.clear_dpb()
    p_net.ref_feature = F.pixel_unshuffle(out["x_hat"], 8)

    chunk = torch.cat([to_model_input(c[0]) for c in frames[1:]], dim=1)
    chunk = pad(chunk, pad_r, pad_b)

    def run():
        with torch.no_grad(), eval_rate():
            p_net.forward_one_frame(chunk, qp)

    print(f"DCVC-UF HTS, RaceHorses {W}x{H}, chunk of {g_frame_delay} inter frames, qp 32")
    stats = bench(run, label=f"forward ({g_frame_delay}-frame chunk)")
    # The chunk is one forward pass covering g_frame_delay frames; per-frame cost
    # is what compares to the other projects.
    per_frame = {k: (v / g_frame_delay if k.endswith("_s") and k != "samples_s"
                     else [s / g_frame_delay for s in v] if k == "samples_s" else v)
                 for k, v in stats.items()}
    print(f"  {'per frame':<34} {per_frame['median_s']:8.3f} s")
    return {"label": "02 Video compression\n(DCVC-UF HTS)", "unit": "per frame",
            "input_px": W * H, "output_px": W * H,
            "detail": f"RaceHorses {W}x{H}, chunk of {g_frame_delay}",
            "timing": per_frame, "chunk_timing": stats}


def t_hybrid_vtm():
    """VTM all-intra: one frame, QP 32. A reference encoder doing full RDO."""
    W, H = 512, 768
    ref = "/work/outputs/experiments/kodim19_420.yuv"
    if not os.path.exists(ref):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                        KODIM, "-pix_fmt", "yuv420p", "-f", "rawvideo", ref], check=True)
    os.makedirs(OUT, exist_ok=True)

    cmd = ["EncoderApp", "-c", f"{os.environ['VTM_CFG']}/encoder_intra_vtm.cfg",
           "-i", ref, f"--SourceWidth={W}", f"--SourceHeight={H}",
           "--InputBitDepth=8", "--FrameRate=1", "--FramesToBeEncoded=1",
           "--QP=32", "--OutputBitDepth=8",
           "-b", f"{OUT}/_timing.bin", "-o", f"{OUT}/_timing_rec.yuv"]

    def run():
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"VTM all-intra, kodim19 {W}x{H}, QP 32")
    # Fewer repeats: this one costs tens of seconds a go, and a subprocess has no
    # warm-up state to speak of.
    stats = bench(run, warmup=1, repeats=3, label="EncoderApp (1 frame)")
    for f in ("_timing.bin", "_timing_rec.yuv"):
        try:
            os.remove(f"{OUT}/{f}")
        except OSError:
            pass
    return {"label": "03 Hybrid codec\n(VTM intra)", "unit": "per image",
            "input_px": W * H, "output_px": W * H,
            "detail": f"kodim19 {W}x{H}, QP 32", "timing": stats}


def t_style_transfer():
    """AdaIN: VGG encode of content and style, statistics swap, decode."""
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    import net as adain_net
    from function import adaptive_instance_normalization

    decoder, vgg = adain_net.decoder, adain_net.vgg
    decoder.eval(); vgg.eval()
    decoder.load_state_dict(torch.load(f"{WEIGHTS}/adain/decoder.pth", map_location="cpu"))
    vgg.load_state_dict(torch.load(f"{WEIGHTS}/adain/vgg_normalised.pth", map_location="cpu"))
    vgg = nn.Sequential(*list(vgg.children())[:31])  # up to relu4_1, as test.py does

    tf = transforms.ToTensor()
    content = tf(Image.open(KODIM).convert("RGB")).unsqueeze(0)
    style_src = "/opt/AdaIN/input/style/brushstrokes.jpg"
    style = tf(Image.open(style_src).convert("RGB")).unsqueeze(0)
    W, H = content.shape[3], content.shape[2]

    def run():
        with torch.no_grad():
            cf, sf = vgg(content), vgg(style)
            decoder(adaptive_instance_normalization(cf, sf))

    print(f"AdaIN, content kodim19 {W}x{H}, style brushstrokes, alpha 1.0")
    stats = bench(run, label="encode + AdaIN + decode")
    return {"label": "04 Style transfer\n(AdaIN)", "unit": "per image",
            "input_px": W * H, "output_px": W * H,
            "detail": f"kodim19 {W}x{H}", "timing": stats}


def t_super_resolution():
    """Real-ESRGAN x4 on a 128x128 crop -- its natural demo size on CPU."""
    import numpy as np
    from PIL import Image
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    im = Image.open(KODIM).convert("RGB").crop((200, 150, 328, 278))
    lr = np.asarray(im)[:, :, ::-1].copy()   # RealESRGANer works in BGR
    W, H = im.size

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    up = RealESRGANer(scale=4, model_path=f"{WEIGHTS}/realesrgan/RealESRGAN_x4plus.pth",
                      model=model, tile=0, tile_pad=10, pre_pad=0, half=False)

    def run():
        up.enhance(lr, outscale=4)

    print(f"Real-ESRGAN x4, {W}x{H} -> {W * 4}x{H * 4}")
    stats = bench(run, warmup=1, repeats=3, label="enhance (x4)")
    return {"label": "05 Super resolution\n(Real-ESRGAN)", "unit": "per image",
            "input_px": W * H, "output_px": W * 4 * H * 4,
            "detail": f"{W}x{H} -> {W * 4}x{H * 4}", "timing": stats}


PROJECTS = {
    "image-compression": t_image_compression,
    "video-compression": t_video_compression,
    "hybrid-vtm": t_hybrid_vtm,
    "style-transfer": t_style_transfer,
    "super-resolution": t_super_resolution,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, choices=sorted(PROJECTS))
    ap.add_argument("--threads", type=int, default=None,
                    help="override torch thread count (default: leave as configured)")
    args = ap.parse_args()

    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass

    result = PROJECTS[args.project]()
    result["project"] = args.project
    result["env"] = env_info(args.threads)
    save(f"{OUT}/{args.project}.json", result)


if __name__ == "__main__":
    main()
