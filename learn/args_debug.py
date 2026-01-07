import argparse

'the launch.json file is used to configure the debugging environment in VSCode'

def main():
    parser = argparse.ArgumentParser(description="示例参数解析")
    parser.add_argument('--size', type=int, default=3, help='尺寸参数')
    parser.add_argument('--echo', type=int, default=2, help='回显参数')
    args = parser.parse_args()

    print(f"size: {args.size}")
    print(f"echo: {args.echo}")

if __name__ == "__main__":
    main()