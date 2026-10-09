import tempfile
import unittest
from pathlib import Path
import numpy as np
import yaml
from train_custom import check_dataset
from pose_overlay import draw_skeletons


class TrainingPoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for split in ('train','val','test'):
            (self.root/'images'/split).mkdir(parents=True)
            (self.root/'labels'/split).mkdir(parents=True)
            # Validator checks label format and exact duplicate content, not decoding.
            (self.root/'images'/split/'a.jpg').write_bytes(split.encode())
            (self.root/'labels'/split/'a.txt').write_text('0 0.5 0.5 0.2 0.2')
        self.config=self.root/'data.yaml'
        self.config.write_text(yaml.safe_dump(dict(path='.',train='images/train',
            val='images/val',test='images/test',names={0:'knife'})))

    def test_valid_and_invalid_boxes(self):
        _,summary=check_dataset(self.config)
        self.assertEqual(summary['train']['boxes'],1)
        (self.root/'labels'/'val'/'a.txt').write_text('0 0.99 0.5 0.5 0.2')
        with self.assertRaisesRegex(ValueError,'bounds'):
            check_dataset(self.config)

    def test_cross_split_duplicate(self):
        (self.root/'images'/'test'/'a.jpg').write_bytes(b'train')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            check_dataset(self.config)

    def test_skeleton_confidence(self):
        frame=np.zeros((100,100,3),dtype=np.uint8)
        points=[[10+i*3,20+i*3,.9] for i in range(17)]
        self.assertEqual(draw_skeletons(frame,[points]),1)
        self.assertTrue(frame.any())
        frame[:]=0
        points=[[10,10,.1] for _ in range(17)]
        self.assertEqual(draw_skeletons(frame,[points]),0)
        self.assertFalse(frame.any())


if __name__=='__main__':
    unittest.main()
