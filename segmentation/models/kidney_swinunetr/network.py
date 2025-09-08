from monai.networks.nets import SwinUNETR
import torchinfo

if __name__ == '__main__':
    model = SwinUNETR(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        feature_size=48,
        norm_name="instance", 
        use_checkpoint=True,
    )
    torchinfo.summary(model, input_size=(2, 1, 128, 128, 128))