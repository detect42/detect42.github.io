import os
import shutil


def process_files(name):
    # 定义输入输出路径
    input_folder = f"unformal_passage/{name}"
    md_file_path = os.path.join(input_folder, f"{name}.md")
    output_md_file = f"source/_posts/{name}.md"
    output_image_folder = f"source/post/{name}"

    # 确保目标文件夹存在
    os.makedirs(output_image_folder, exist_ok=True)

    # 处理Markdown文件
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 替换\(, \) 为 $, 并将 \[ 或 \] 替换为 $$
    content = content.replace("\\(", "$").replace("\\)", "$")
    content = content.replace("\\[", "$$").replace("\\]", "$$")

    # 替换图片路径为 {name}\image
    content = content.replace("image", f"{name}/image")

    # 保存修改后的Markdown文件
    with open(output_md_file, "w", encoding="utf-8") as f:
        f.write(content)

    # 复制图片到目标文件夹
    for file_name in os.listdir(input_folder):
        if file_name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            shutil.copy(os.path.join(input_folder, file_name), output_image_folder)

    print(f"Processing completed for {name}.")


# 调用函数处理具体文件夹
name = input("Enter the name of the folder: ")
process_files(name)
