from self_attention_cv import UNETR
import torchinfo

if __name__ == '__main__':
    model = UNETR(img_shape=(32, 256, 256), input_dim=1, output_dim=2,
                embed_dim=512, patch_size=16, num_heads=10,
                ext_layers=[3, 6, 9, 12], norm='instance',
                base_filters=16,
                dim_linear_block=2048)
    torchinfo.summary(model, input_size=(2, 1, 32, 256, 256))

    # Input size: (2, 1, 32, 256, 256)
    # Output size: (2, 2, 32, 256, 256)