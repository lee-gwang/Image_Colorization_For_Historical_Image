import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

# ------------------------
#  Model
# ------------------------
class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm

class LABModel(nn.Module):
    def __init__(self, model, CFG):
        super().__init__()
        self.model = model
        self.CFG = CFG
        self.upsample = nn.Upsample(scale_factor=4)
    
    def forward(self, x):
        #x = F.interpolate(x, scale_factor=1/4, mode='bilinear')
        x = self.model(x)
        return x#self.upsample(x)

def build_model(CFG, encoder, decoder):
    
    if encoder == 'LeViT_UNet_384':
        model = Build_LeViT_UNet_384(
                pretrained=True,
                num_classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
            )

    elif decoder == 'Unet':
        model = smp.Unet(
            encoder_name=encoder,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=CFG.num_channel,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
            activation=CFG.activation,

        )
    elif decoder == 'FPN':
        model = smp.FPN(
            encoder_name=encoder,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=CFG.num_channel,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
            activation=CFG.activation,
        )
    elif decoder == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=encoder,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=CFG.num_channel,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
            activation=CFG.activation,
        )
        
    model = LABModel(model, CFG)

    return model
