from models import *

@torch.no_grad()
def infer(src):
    net = init_model()
    image = pre_processing(src).to(device)
    result = net(image)
    output = post_processing(result, src)
    return output



def main():
    args = parse_args()
    src_name = get_name(args)
    output_prefix = save_dir(src_name)
    src = read(args)
    result = infer(src)
    write(f'{output_prefix / src_name}.png', result)


if __name__ == "__main__":
    main()
