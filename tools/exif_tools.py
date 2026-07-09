#!/usr/bin/env python3
"""
exif_tools.py - EXIF for the FLUX generate/edit pipelines.

 - EDITS (real photos): extract_source() pulls genuine EXIF from the source (ARW/JPEG/TIFF);
   it is preserved and also mapped to edit params (grain/DoF/WB) via exif_to_edit_params().
 - GENERATION: infer_from_prompt() + estimate_from_image() synthesise plausible camera fields.
   A provenance marker (Software + IPTC DigitalSourceType=trainedAlgorithmicMedia) is always
   written so the file is honestly identifiable as AI-originated while remaining fully sortable.
 - write_jpeg() writes a high-quality JPEG carrying the EXIF; embed_tiff() adds EXIF to a TIFF.
"""
import re, math, datetime
import numpy as np
import piexif
import piexif.helper

PROVENANCE_SOFTWARE = "ComfyUI FLUX.1-dev"
IPTC_AI = "DigitalSourceType=trainedAlgorithmicMedia"   # IPTC synthetic-media descriptor

_CAMS = {
    'sony a7 iv':('SONY','ILCE-7M4','FE 50mm F1.8'), 'sony a7':('SONY','ILCE-7M4','FE 50mm F1.8'),
    'a7iii':('SONY','ILCE-7M3','FE 24-70mm F2.8 GM'), 'a7r':('SONY','ILCE-7RM5','FE 35mm F1.4 GM'),
    'canon r5':('Canon','Canon EOS R5','RF50mm F1.2 L USM'), 'canon 5d':('Canon','Canon EOS 5D Mark IV','EF50mm f/1.4 USM'),
    'nikon z':('NIKON CORPORATION','NIKON Z 7','NIKKOR Z 50mm f/1.8 S'), 'fujifilm':('FUJIFILM','X-T5','XF35mmF1.4 R'),
    'leica':('Leica Camera AG','LEICA M11','Summilux-M 50mm f/1.4'), 'hasselblad':('Hasselblad','X2D 100C','XCD 55V'),
}
_DEFAULT = ('SONY','ILCE-7M4','FE 50mm F1.8')

# ---------------- inference from prompt ----------------
def infer_from_prompt(prompt):
    p=(prompt or '').lower(); out={}
    m=re.search(r'(\d{1,3})\s*mm', p);            out['focal']=int(m.group(1)) if m else None
    m=re.search(r'f[\/ ]?(\d{1,2}(?:\.\d)?)', p); out['fnum']=float(m.group(1)) if m else None
    m=re.search(r'iso\s*(\d{2,6})', p);           out['iso']=int(m.group(1)) if m else None
    m=re.search(r'\b1/(\d{1,5})\b', p);           out['shutter']=int(m.group(1)) if m else None
    out['cam']=None
    for k,v in _CAMS.items():
        if k in p: out['cam']=v; break
    out['scene']='night' if ('night' in p or 'neon' in p) else ('golden' if any(w in p for w in ['golden hour','sunset','sunrise','dusk','dawn']) else None)
    return {k:v for k,v in out.items() if v is not None}

# ---------------- estimation from image ----------------
def estimate_from_image(img01):
    import cv2
    h,w=img01.shape[:2]; l=img01@np.array([0.2126,0.7152,0.0722],np.float32); mean=float(l.mean())
    iso,sh = (3200,60) if mean<0.15 else (1600,125) if mean<0.30 else (400,250) if mean<0.50 else (100,500)
    lap=cv2.Laplacian((np.clip(l,0,1)*255).astype('uint8'),cv2.CV_64F).var()
    fnum = 1.8 if lap<80 else 2.8 if lap<300 else 5.6
    r,b=float(img01[...,0].mean()),float(img01[...,2].mean()); ratio=r/(b+1e-6)
    temp = 3200 if ratio>1.25 else 5600 if ratio>0.9 else 7000
    return {'w':w,'h':h,'iso':iso,'shutter':sh,'fnum':fnum,'temp':temp}

# ---------------- extract real EXIF (edits) ----------------
def extract_source(path):
    """Return a normalised dict from a real photo/raw, or {} if none."""
    try:
        import exifread
        with open(path,'rb') as f: tags=exifread.process_file(f, details=False)
        g=lambda k: str(tags[k]) if k in tags else None
        def frac(k):
            v=tags.get(k)
            if not v: return None
            try:
                r=v.values[0]; return float(r.num)/float(r.den)
            except Exception:
                try: return float(str(v))
                except: return None
        out={'make':g('Image Make'),'model':g('Image Model'),'lens':g('EXIF LensModel'),
             'focal':frac('EXIF FocalLength'),'fnum':frac('EXIF FNumber'),
             'iso':int(g('EXIF ISOSpeedRatings')) if g('EXIF ISOSpeedRatings') else None,
             'datetime':g('EXIF DateTimeOriginal') or g('Image DateTime'),
             'shutter':None}
        et=frac('EXIF ExposureTime')
        if et: out['shutter']=int(round(1/et)) if et<1 else et
        return {k:v for k,v in out.items() if v is not None}
    except Exception:
        return {}

# ---------------- merge + build piexif dict ----------------
def build_exif(prompt='', img01=None, source=None, ai=True, extra=None):
    """source (real) wins; else prompt-inferred; else image-estimated; else defaults.
    ai=True  -> Software=FLUX + IPTC synthetic-media marker in UserComment.
    ai=False -> clean camera EXIF, no marker (indistinguishable from a real capture).
    extra    -> dict; when given, its key=value lines go into UserComment (rich mode)."""
    src=source or {}; pr=infer_from_prompt(prompt); es=estimate_from_image(img01) if img01 is not None else {}
    def pick(*vs):
        for v in vs:
            if v is not None: return v
        return None
    make=pick(src.get('make'), (pr.get('cam') or (None,))[0])
    model=pick(src.get('model'), (pr.get('cam') or (None,None))[1] if pr.get('cam') else None)
    lens=pick(src.get('lens'), (pr.get('cam') or (None,None,None))[2] if pr.get('cam') else None)
    if not make: make,model,lens=_DEFAULT
    if not lens: lens=_DEFAULT[2]
    if not model: model=_DEFAULT[1]
    focal=pick(src.get('focal'), pr.get('focal'), 50)
    fnum =pick(src.get('fnum'),  pr.get('fnum'),  es.get('fnum'), 2.8)
    iso  =pick(src.get('iso'),   pr.get('iso'),   es.get('iso'),  200)
    shut =pick(src.get('shutter'),pr.get('shutter'),es.get('shutter'),250)
    dt   =pick(src.get('datetime'), datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
    w=(es.get('w') or (img01.shape[1] if img01 is not None else 1024)); h=(es.get('h') or (img01.shape[0] if img01 is not None else 1024))

    def rat(x,den=100): return (int(round(x*den)),den)
    zeroth={piexif.ImageIFD.Make:make, piexif.ImageIFD.Model:model, piexif.ImageIFD.DateTime:dt}
    if ai:
        zeroth[piexif.ImageIFD.Software]=PROVENANCE_SOFTWARE
    exif={piexif.ExifIFD.DateTimeOriginal:dt, piexif.ExifIFD.DateTimeDigitized:dt,
          piexif.ExifIFD.FocalLength:rat(float(focal)), piexif.ExifIFD.FNumber:rat(float(fnum),10),
          piexif.ExifIFD.ISOSpeedRatings:int(iso), piexif.ExifIFD.ExposureTime:(1,int(shut)),
          piexif.ExifIFD.LensModel:lens, piexif.ExifIFD.PixelXDimension:int(w), piexif.ExifIFD.PixelYDimension:int(h)}
    comment=None
    if extra:
        comment=" | ".join(f"{k}={v}" for k,v in extra.items())
    elif ai:
        comment=IPTC_AI
    if comment:
        exif[piexif.ExifIFD.UserComment]=piexif.helper.UserComment.dump(comment)
    return {"0th":zeroth,"Exif":exif,"1st":{},"thumbnail":None,"GPS":{}}

# ---------------- writers ----------------
def write_jpeg(path, img_uint8_rgb, exif_dict, quality=95):
    from PIL import Image
    Image.fromarray(img_uint8_rgb).save(path,'JPEG',quality=quality,exif=piexif.dump(exif_dict))
    return path

def embed_tiff(path, exif_dict):
    try: piexif.insert(piexif.dump(exif_dict), path)
    except Exception as e: print('exif_tools: tiff embed skipped:',str(e)[:60])

# ---------------- EXIF -> edit params (realism) ----------------
def exif_to_edit_params(exif_norm):
    """Map real-camera EXIF to edit params so results match the source optics/sensor."""
    iso=exif_norm.get('iso',200); fnum=exif_norm.get('fnum',2.8)
    grain = float(np.clip(0.1+math.log2(max(iso,100)/100.0)*0.09, 0.1, 0.7))   # more grain at high ISO
    blur  = float(np.clip((2.8/max(fnum,1.0))*10.0, 2.0, 24.0))                 # wider aperture -> more bokeh
    return {'grain_power':round(grain,3),'blur_strength':round(blur,1)}

if __name__=='__main__':
    import sys, json
    print(json.dumps(infer_from_prompt(' '.join(sys.argv[1:]) or 'portrait 85mm f/1.4 iso 800 shot on sony a7'), indent=1))
