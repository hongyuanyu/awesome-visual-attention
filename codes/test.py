from torchvision.models import resnet50
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from cbam import*
from GCNet import*
from emanet import*
from senet import*
from scse import*
from ecanet import*
from sknet import*#分类1000类
from triplet_attention import*
from non_local import*
#from fcanet import*
#from memory_compress_attention import*
#from set_transform import modules,set_transformer
from efficient_attention import*
model = [CBAM(64),SELayer(64),scSELayer(64),eca_layer(64),SKConv(64,244,3,8,2),TripletAttention(64),NONLocalBlock2D(in_channels=64,sub_sample=True,bn_layer=True), ContextBlock2d(64,64),EMAU(64,101),EfficientAttention(64,64,224,224)]

    
    

input = torch.randn(1, 64, 224, 224)
inputs1 = (torch.rand(1, 64, 224, 224),)
for e in model:
    print(flop_count_table(FlopCountAnalysis(e.eval(), inputs1)))
    # macs, params = profile(e, inputs=(input, ))
    # print("Flops: {} M Params: {}".format(macs/1e6, params/1e6))

