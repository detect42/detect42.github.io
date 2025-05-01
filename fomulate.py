import os
import shutil
import re


def process_files(name):
    # Define input and output paths
    input_folder = f"unformal_passage/{name}"
    md_file_path = os.path.join(input_folder, f"{name}.md")
    output_md_file = f"source/_posts/{name}.md"
    output_image_folder = f"source/post/{name}"

    # Ensure the target folder exists
    os.makedirs(output_image_folder, exist_ok=True)

    # Process the Markdown file
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace \( ... \) with $...$
    content = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", content, flags=re.DOTALL | re.MULTILINE)

    # Replace \[ ... \] with $$...$$
    content = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", content, flags=re.DOTALL | re.MULTILINE)

    # Replace image paths
    content = re.sub(
        r"!\[.*?\]\(\<?(.*?\.png)\>?\)", rf'<img src="{name}/\1" alt="" width="70%" height="70%">', content
    )

    # Save the modified Markdown file
    with open(output_md_file, "w", encoding="utf-8") as f:
        f.write(content)

    # Copy images to the target folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            shutil.copy(os.path.join(input_folder, file_name), output_image_folder)

    print(f"Processing completed for {name}.")


# 调用函数处理具体文件夹
name = input("Enter the name of the folder: ")
process_files(name)
