"""Synchronous review mosaic, read from the existing four render products."""
from PIL import Image,ImageDraw

LAYOUT=['follow_camera','front_camera_before_image_attack','wrist_camera_before_image_attack','rear_camera_before_image_attack']


def compose(c):
    frame=Image.new('RGB',(1280,720))
    for index,(camera,label) in enumerate(zip([3,0,2,1],LAYOUT)):
        rgb=c.recording_rgb(c.cameras[camera]) if camera==3 else c.rgb(c.cameras[camera])
        panel=Image.fromarray(rgb).convert('RGB').resize((640,360),Image.Resampling.BILINEAR)
        draw=ImageDraw.Draw(panel);draw.rectangle((0,0,640,20),fill=(0,0,0));draw.text((6,4),label,fill='white')
        frame.paste(panel,((index%2)*640,(index//2)*360))
    return frame
