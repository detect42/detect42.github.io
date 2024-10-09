import os
import shutil
import re


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

    # 替换 \( 和 \) 为 $
    content = content.replace("\\(", "$").replace("\\)", "$")

    # 使用正则表达式将 \[公式\] 替换为 $$公式$$，确保公式没有换行或多余空格
    content = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", content, flags=re.DOTALL | re.MULTILINE)

    # 替换图片路径，将 ![alt text](image-x.png) 替换为 <img src="{name}/image-x.png" alt="" width="70%" height="70%">
    # 正则表达式确保匹配所有image*.png格式
    content = re.sub(r"!\[.*?\]\((image.*?\.png)\)", rf'<img src="{name}/\1" alt="" width="70%" height="70%">', content)

    # 保存修改后的Markdown文件
    with open(output_md_file, "w", encoding="utf-8") as f:
        f.write(content)

    # 复制图片到目标文件夹
    for file_name in os.listdir(input_folder):
        if file_name.startswith("image") and file_name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            shutil.copy(os.path.join(input_folder, file_name), output_image_folder)

    print(f"Processing completed for {name}.")


# 调用函数处理具体文件夹
name = input("Enter the name of the folder: ")
process_files(name)
