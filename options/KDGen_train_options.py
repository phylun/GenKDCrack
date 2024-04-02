import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.common_options import CommonOptions

class KDGenTrainOptions(CommonOptions):
    def initialize(self, parser):        
        parser = CommonOptions.initialize(self, parser)        
        # model parameter for knowledge distillation 
        parser.add_argument('--dataroot', default='./dataset_sample/labeled/trainConc', help='path to image data folder')
        parser.add_argument('--listfile', type=str, default='trainConc.txt', help='text file of data file list')
        
        parser.add_argument('--semidataroot', default='./dataset_sample/unlabeled/trainConc', help='path to image data folder')
        parser.add_argument('--semilistfile', type=str, default='gentrainConc.txt', help='text file of data file list')
                
        parser.add_argument('--phase', type=str, default='train', help='[train | val | test]')        
        parser.add_argument('--batch_size', type=int, default=32, help='input data size, should be even number')
        parser.add_argument('--flip', type=bool, default=False, help='whether image data flips or not')
        parser.add_argument('--crop', type=bool, default=True, help='whether image data crops or not. If true, assure crop size is needed')
        
        # train options
        parser.add_argument('--n_epoch', type=int, default=3000, help='# of total epochs')
        parser.add_argument('--save_freq', type=int, default=10, help='how often models are saved')                
                
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
        
        parser.add_argument('--lp_epoch', type=int, default=1000, help='save learning progress during the training')
        parser.add_argument('--lp_freq', type=int, default=6, help='save learning progress during the training')
                                        
        parser.add_argument('--consist_weight', type=float, default=0.001, help='consistency weight')                
        self.isTrain = True                                
        
        return parser

