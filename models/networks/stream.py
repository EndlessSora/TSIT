import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import StreamResnetBlock as StreamResnetBlock


# Content/style stream.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.
class Stream(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.res_0 = StreamResnetBlock(opt.semantic_nc, 1 * nf, opt)  # 64-ch feature
        self.res_1 = StreamResnetBlock(1  * nf, 2  * nf, opt)   # 128-ch  feature
        self.res_2 = StreamResnetBlock(2  * nf, 4  * nf, opt)   # 256-ch  feature
        self.res_3 = StreamResnetBlock(4  * nf, 8  * nf, opt)   # 512-ch  feature
        self.res_4 = StreamResnetBlock(8  * nf, 16 * nf, opt)   # 1024-ch feature
        self.res_5 = StreamResnetBlock(16 * nf, 16 * nf, opt)   # 1024-ch feature
        self.res_6 = StreamResnetBlock(16 * nf, 16 * nf, opt)   # 1024-ch feature
        self.res_7 = StreamResnetBlock(16 * nf, 16 * nf, opt)   # 1024-ch feature

    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

    def forward(self,input):
        # assume that input shape is (n,c,256,512)

        x0 = self.res_0(input) # (n,64,256,512)

        x1 = self.down(x0)
        x1 = self.res_1(x1)    # (n,128,128,256)

        x2 = self.down(x1)
        x2 = self.res_2(x2)    # (n,256,64,128)

        x3 = self.down(x2)
        x3 = self.res_3(x3)    # (n,512,32,64)

        x4 = self.down(x3)
        x4 = self.res_4(x4)    # (n,1024,16,32)

        x5 = self.down(x4)
        x5 = self.res_5(x5)    # (n,1024,8,16)

        x6 = self.down(x5)
        x6 = self.res_6(x6)    # (n,1024,4,8)

        x7 = self.down(x6)
        x7 = self.res_7(x7)    # (n,1024,2,4)

        return [x0, x1, x2, x3, x4, x5, x6, x7]
