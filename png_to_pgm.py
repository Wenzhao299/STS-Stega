import os
import argparse
from PIL import Image
from tqdm import tqdm

def convert_images(input_dir, output_dir):
    """
    将输入目录中的彩色图片分解为R, G, B三个独立的灰度PGM文件，
    并分别存入输出目录下的 R, G, B 子目录中。
    """
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_formats]

    if not image_files:
        print(f"警告: 在 '{input_dir}' 中未找到支持的图片文件。")
        return

    print(f"开始处理目录 '{input_dir}' 中的 {len(image_files)} 张图片，将每张分解为R,G,B三个通道...")

    for filename in tqdm(image_files, desc="Processing images"):
        try:
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 分离R, G, B通道
                channels = img.split()
                channel_names = ['R', 'G', 'B']

                for i, channel_img in enumerate(channels):
                    channel_name = channel_names[i]
                    
                    # 创建特定于通道的输出子目录
                    channel_output_dir = os.path.join(output_dir, channel_name)
                    os.makedirs(channel_output_dir, exist_ok=True)

                    # 在子目录中创建PGM文件，文件名只保留数字一致
                    output_filename = f"{base_name.split('_')[-1]}.pgm"
                    output_path = os.path.join(channel_output_dir, output_filename)
                    
                    # 直接保存单通道图像
                    channel_img.save(output_path)

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("处理完成！")


def main():
    parser = argparse.ArgumentParser(description="将RGB图片分解为R,G,B通道，并分别存入不同子目录，用于SRM分析")
    parser.add_argument("--input", default='output_stc_parallel/2025-07-15_19-23-49/stega', help="包含源图片(PNG, JPG等)的目录")
    parser.add_argument("--output", default='/data/home/wls_cwz/data/dataset/sts_a/3-60k-20k_stc_pgm/stega', help="存放转换后的PGM文件的根目录")
    args = parser.parse_args()

    convert_images(args.input, args.output)

if __name__ == "__main__":
    main() 