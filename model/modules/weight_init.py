# #!/usr/bin/python3
# # coding=utf-8
# import torch.nn as nn


# def weight_init(module):
#     for n, m in module.named_children():
#         print('initialize: ' + n)
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
#             nn.init.ones_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Sequential):
#             weight_init(m)
#         elif isinstance(m, nn.ReLU):
#             pass
#         elif isinstance(m, nn.AdaptiveAvgPool2d):
#             pass
#         elif isinstance(m, nn.AdaptiveMaxPool2d):
#             pass
#         elif isinstance(m, nn.Sigmoid):
#             pass
#         else:
#             m.initialize()



import torch.nn as nn

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)

        # 初始化卷积层
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 初始化批归一化层
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 初始化全连接层
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 递归初始化 nn.Sequential 层
        elif isinstance(m, nn.Sequential):
            weight_init(m)

        # 不需要初始化的层（ReLU，池化等）
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass

        # 仅对有 initialize 方法的模块调用 initialize
        elif hasattr(m, 'initialize') and callable(getattr(m, 'initialize')):
            print(f"Calling initialize for {n}")
            m.initialize()

        # 如果不属于上述任何类型，不做处理
        else:
            print(f"Skipping {n}, no initialization function found")