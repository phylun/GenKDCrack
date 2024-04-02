import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.common_options import CommonOptions

class TestOptions(CommonOptions):
    def initialize(self, parser):        
        parser = CommonOptions.initialize(self, parser)                
        parser.add_argument('--testdataroot', default='./dataset_sample/labeled/valConc', help='path to image data folder')
        parser.add_argument('--testlistfile', type=str, default='valConc.txt', help='text file of data file list')
        parser.add_argument('--phase', type=str, default='val', help='[train | val | test]')                        
        parser.add_argument('--batch_size', type=int, default=1, help='input data size, for validation')
        parser.add_argument('--epoch', type=int, default=-1, help='specify exact saved weight number')        
        parser.add_argument('--flip', type=bool, default=False, help='whether image data flips or not')
        parser.add_argument('--crop', type=bool, default=False, help='whether image data crops or not. If true, assure crop size is needed')
                
        self.isTrain = True
                                
        return parser

