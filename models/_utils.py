def patch_first_conv(model, n_channels):
    """Change first convolution layer input channels.
    In case:
        n_channels == 1 or n_channels == 2 -> reuse original weights
        n_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.n_channels = n_channels
    weight = module.weight.detach()
    reset = False

    if n_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif n_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels, module.n_channels // module.groups, *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()
