import unittest,tempfile,subprocess,json,hashlib
from pathlib import Path
from processor import enhance
class ProcessorTest(unittest.TestCase):
 def test_real_interpolation_and_audio(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'source.mp4'; output=Path(tmp)/'output.mp4'
   subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x96:rate=24:duration=1.5','-f','lavfi','-i','sine=frequency=440:duration=1.5','-c:v','libx264','-c:a','aac','-shortest',str(source)],check=True)
   metadata=enhance(source,output,'1080p',60,lambda *_:None)
   self.assertEqual(metadata['enhancedFps'],60);self.assertEqual(metadata['enhancedResolution'],'1800x1080')
   self.assertAlmostEqual(metadata['durationSeconds'],1.5,places=2)
   audio=lambda p:subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a','-c:a','copy','-f','adts','-'])
   self.assertEqual(hashlib.sha256(audio(source)).digest(),hashlib.sha256(audio(output)).digest())
   # Downscale for comparison: a repeated-frame conversion has <=36 unique images.
   raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-vf','scale=160:96','-f','rawvideo','-pix_fmt','rgb24','-'])
   framebytes=160*96*3
   unique={hashlib.sha256(raw[i:i+framebytes]).digest() for i in range(0,len(raw),framebytes)}
   self.assertGreater(len(unique),70)
   self.assertIn('MCI',metadata['interpolationMethod'])
if __name__=='__main__':unittest.main()
