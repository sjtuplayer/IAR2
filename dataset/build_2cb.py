from dataset.imagenet_2cb import build_imagenet, build_imagenet_code



def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)

    raise ValueError(f'dataset {args.dataset} is not supported')