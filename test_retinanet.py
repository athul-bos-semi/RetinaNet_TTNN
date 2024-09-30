import ttnn
import torch
import torch.nn as nn

from ttnn_retinanet_utils import BottleNeck
    
device = ttnn.open_device(device_id=2, l1_small_size=32768)
in_channels = 3
image_height, image_width = 224,224
test_image = torch.randn(1, in_channels, image_height, image_width, dtype=torch.float)
# test_image = test_image.permute(0,2,3,1).reshape(1,1,-1,in_channels)
# test_image = ttnn.from_torch(test_image, ttnn.bfloat16)

# parameters = None
# layer_num = 0
# block_num = 0
# inplanes = 32
# planes = 64
# base_width = 64
# stride = 2
# downsample = False
# input_size = [image_height,image_width]
# config_overrides = None
# config_overrides = [None,{"act_block_h": 64},{"act_block_h": 64, "out_subblock_h": 64, "out_subblock_w": 64}]
dtype = ttnn.bfloat16

class RetinaNet(nn.Module):
    def __init__(self, image_size, in_channels, parameters, layers, dtype, device):
        super().__init__()
        self.in_channels = in_channels
        self.input_height = image_size[0]
        self.input_width = image_size[1]
        self.in_planes = 64
        self.expansion = 4
        self.layers = layers
        self.dtype = dtype
        self.device = device
        self.device_name = "wormhole_b0"

        self.first_conv_weights = None
        self.first_conv_bn_weights = None
        if parameters is not None:
            raise Exception("parameters not implemented for now!")
        else:
            self.first_conv_weights = torch.rand(self.in_planes, self.in_channels, 7, 7, dtype=torch.bfloat16)
            self.first_conv_weights = ttnn.from_torch(self.first_conv_weights, dtype=self.dtype)
        self.first_conv_use_shallow_conv_variant = False
        self.first_conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            # height_sharding=True,
            input_channels_alignment=16 if self.first_conv_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )
    
    def forward(self, inputs):
        inputs = inputs.permute(0,2,3,1).reshape(1,1,-1,in_channels)
        x = ttnn.from_torch(inputs, ttnn.bfloat16)

        print("Input = ", x.get_legacy_shape())

        if x.get_legacy_shape()[2] == 50176 and x.get_legacy_shape()[3] == 3:
            reshard_shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,6))}),
                [896,32],
                ttnn.ShardOrientation.COL_MAJOR,
                False
            )
            reshard_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, reshard_shard_spec
            )
            # x = ttnn.reshard(x, reshard_mem_config)
            x = ttnn.to_device(x, self.device, reshard_mem_config)
            print(">>> Sharded")

        x, feature_height, feature_width, self.conv1_weights, self.bn1_weights = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.first_conv_weights,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.in_planes,
            batch_size=1,
            input_height=self.input_height,
            input_width=self.input_width,
            kernel_size=[7,7],
            stride=[2,2],
            padding=[1,1],
            dilation=[1,1],
            groups=1,
            bias_tensor=self.first_conv_bn_weights,
            conv_config=self.first_conv_config,
            conv_op_cache={},
            # debug=False,
        )
        print("\nFirst Conv = ", x.shape)
        print(x.memory_config())


# model = BottleNeck(parameters, layer_num, block_num, inplanes, planes, base_width, stride, downsample, input_size, config_overrides, dtype, device)
model = RetinaNet([image_height,image_width], in_channels, None, [3,4,6,3], dtype, device)

print(model)

x = model(test_image)