#!/usr/bin/env python3
"""
mask_hq.py - best-of-both SOTA masking in one call.

  Grounded-SAM  -> WHICH objects (semantic; drops crowd & see-through background)
  BiRefNet HR   -> crisp coarse edges (spokes/hair), gated to the SAM region
  ViTMatte      -> alpha-perfect edges + defringe

Pipeline: sam_region = GroundedSAM(query, main-subject)
          coarse     = BiRefNet(General-HR)  * dilate(sam_region)      # crisp edges, right objects
          alpha      = ViTMatte(image, coarse) + defringe

Usage:
  ~/comfy/ComfyUI/.venv/bin/python mask_hq.py IMAGE --query "bicycle. person." \
      --out alpha.png [--main-subject] [--gate-dilate 30] [--band 16] [--maxside 2048]
Runs on inferno (needs ComfyUI up for BiRefNet + transformers for GSAM/ViTMatte).
"""
from __future__ import annotations
import argparse, json, os, sys, time, uuid, shutil, urllib.request
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/comfy"))
import grounded_sam as GS
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

COMFY = "http://127.0.0.1:8188"
COMFY_INPUT = os.path.expanduser("~/comfy/ComfyUI/input")
COMFY_OUTPUT = os.path.expanduser("~/comfy/ComfyUI/output")

# ---------------- BiRefNet via ComfyUI API ----------------
def birefnet_mask(image_path, model_name="General-HR"):
    name = f"maskhq_{uuid.uuid4().hex[:8]}.jpg"
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    s = min(1.0, 2048/max(W, H))                         # cap for VRAM (P100)
    imr = im.resize((int(W*s), int(H*s)), Image.LANCZOS) if s < 1 else im
    imr.save(os.path.join(COMFY_INPUT, name), quality=94)
    rw, rh = imr.size
    rw = max(32, round(rw/32)*32); rh = max(32, round(rh/32)*32)  # BiRefNet needs dims %32
    wf = {
      "1": {"class_type":"LoadImage","inputs":{"image":name}},
      "2": {"class_type":"AutoDownloadBiRefNetModel","inputs":{"model_name":model_name,"device":"AUTO"}},
      "3": {"class_type":"GetMaskByBiRefNet","inputs":{"model":["2",0],"images":["1",0],"width":rw,"height":rh,"upscale_method":"bilinear","mask_threshold":0.5}},
      "5": {"class_type":"MaskToImage","inputs":{"mask":["3",0]}},
      "6": {"class_type":"SaveImage","inputs":{"images":["5",0],"filename_prefix":"maskhq_bire"}},
    }
    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        COMFY+"/prompt", data=json.dumps({"prompt":wf}).encode(),
        headers={"Content-Type":"application/json"}), timeout=30))["prompt_id"]
    out = None
    for _ in range(120):
        time.sleep(3)
        try: h = json.load(urllib.request.urlopen(COMFY+"/history/"+pid, timeout=12))
        except: continue
        if h:
            hv = list(h.values())[0]
            if hv.get("status",{}).get("status_str") == "error":
                raise RuntimeError("BiRefNet error: "+json.dumps(hv.get("status",{}))[:200])
            for v in hv.get("outputs",{}).values():
                if "images" in v: out = v["images"][-1]["filename"]
            if out: break
    if not out: raise RuntimeError("BiRefNet timed out")
    m = cv2.imread(os.path.join(COMFY_OUTPUT, out), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    os.remove(os.path.join(COMFY_INPUT, name))
    return m

# ---------------- ViTMatte + defringe (bbox-crop) ----------------
def vitmatte(image_rgb, coarse01, model_dir, band, maxside):
    H, W = image_rgb.shape[:2]
    mb = (coarse01 > 0.5).astype(np.uint8)
    ys, xs = np.where(mb > 0)
    if len(xs) == 0: return coarse01
    pad = 64
    x0,x1 = max(0,xs.min()-pad), min(W,xs.max()+pad); y0,y1 = max(0,ys.min()-pad), min(H,ys.max()+pad)
    crop = image_rgb[y0:y1,x0:x1]; cm = (coarse01[y0:y1,x0:x1]*255).astype(np.uint8)
    ch,cw = crop.shape[:2]; s = min(1.0, maxside/max(ch,cw))
    if s<1: crop_s=cv2.resize(crop,(int(cw*s),int(ch*s)),cv2.INTER_AREA); cm_s=cv2.resize(cm,(int(cw*s),int(ch*s)),cv2.INTER_LINEAR)
    else: crop_s,cm_s=crop,cm
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band+1,2*band+1))
    cmb=(cm_s>128).astype(np.uint8); fg=cv2.erode(cmb,k); bgo=cv2.dilate(cmb,k)
    tri=np.full(cm_s.shape,0.5,np.float32); tri[fg>0]=1.0; tri[bgo==0]=0.0
    proc=VitMatteImageProcessor.from_pretrained(model_dir)
    model=VitMatteForImageMatting.from_pretrained(model_dir,torch_dtype=torch.float32).to("cuda").eval()
    inp=proc(images=Image.fromarray(crop_s),trimaps=Image.fromarray((tri*255).astype(np.uint8)),return_tensors="pt")
    inp={kk:v.to("cuda") for kk,v in inp.items()}
    with torch.no_grad(): alpha=model(**inp).alphas[0,0].float().cpu().numpy()
    del model; torch.cuda.empty_cache()
    alpha=alpha[:crop_s.shape[0],:crop_s.shape[1]]
    # defringe on dark subject
    lum=(crop_s.astype(np.float32)@np.array([0.2126,0.7152,0.0722],np.float32))/255.0
    core=alpha>0.9
    if core.sum()>50:
        fl=float(np.median(lum[core])); edge=(alpha>0.03)&(alpha<0.97)
        supp=1.0-np.clip(np.clip(lum-(fl+0.12),0,1)*4.0,0,0.85)
        alpha=np.where(edge,alpha*supp,alpha)
    alpha=np.clip((alpha-0.5)*1.25+0.5,0,1)
    if s<1: alpha=cv2.resize(alpha,(cw,ch),cv2.INTER_LINEAR)
    out=np.zeros((H,W),np.float32); out[y0:y1,x0:x1]=np.clip(alpha,0,1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--query", required=True); ap.add_argument("--out", default="mask_hq.png")
    ap.add_argument("--box-threshold", type=float, default=0.3); ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--main-subject", action="store_true", default=True)
    ap.add_argument("--gate-dilate", type=int, default=30); ap.add_argument("--band", type=int, default=16)
    ap.add_argument("--maxside", type=int, default=2048); ap.add_argument("--vitmatte", default=os.path.expanduser("~/models/vitmatte-small"))
    ap.add_argument("--no-sam", action="store_true", help="skip semantic gating (BiRefNet+ViTMatte only)")
    a=ap.parse_args()
    image=Image.open(a.image).convert("RGB"); W,H=image.size
    img_np=np.array(image)

    print("[1/3] BiRefNet coarse (crisp edges)...")
    bire=birefnet_mask(a.image, "General-HR")

    if a.no_sam:
        coarse=bire
    else:
        print("[2/3] Grounded-SAM semantic region...")
        boxes,labels,scores=GS.detect(image,a.query,a.box_threshold,a.text_threshold)
        print("   detected:", ", ".join(f"{l}({s:.2f})" for l,s in zip(labels,scores)) or "none")
        if len(boxes)==0:
            print("   no objects matched -> falling back to BiRefNet only"); coarse=bire
        else:
            if a.main_subject and len(boxes)>1:
                ar=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]; an=int(np.argmax(ar))
                def inter(p,q):
                    x0=max(p[0],q[0]);y0=max(p[1],q[1]);x1=min(p[2],q[2]);y1=min(p[3],q[3])
                    ia=max(0,x1-x0)*max(0,y1-y0); return ia/max(1,min((p[2]-p[0])*(p[3]-p[1]),(q[2]-q[0])*(q[3]-q[1])))
                keep=[i for i in range(len(boxes)) if i==an or inter(boxes[an],boxes[i])>0.05]
                boxes=boxes[keep]; print("   main-subject kept:", ", ".join(labels[i] for i in keep))
            region=GS.segment(image,boxes).astype(np.uint8)
            reg_d=cv2.dilate(region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(a.gate_dilate,a.gate_dilate)))
            coarse=bire*reg_d.astype(np.float32)          # crisp edges gated to semantic objects

    print("[3/3] ViTMatte alpha + defringe...")
    alpha=vitmatte(img_np, coarse, a.vitmatte, a.band, a.maxside)
    cv2.imwrite(a.out,(np.clip(alpha,0,1)*255+0.5).astype(np.uint8))
    print("done ->", a.out, "| coverage %.1f%%"%(100*(alpha>0.5).mean()))

if __name__=="__main__":
    main()
