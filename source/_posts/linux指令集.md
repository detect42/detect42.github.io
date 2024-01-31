---
title: linux常用指令集
tags: linux
categories: 
- tool
- linux
abbrlink: 4de2ff1e
date: 2023-09-02 14:58:46
---

## 指定服务器调用显卡

```export CUDA_VISIBLE_DEVICES=1,3```

## pip换源

python -m pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple

## 代理

### 一次性打开代理
```
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7891
```

### 打开代理：
```clash -d ~/.config/clash```

-----------


如果你想要临时关闭终端会话中的代理设置，可以运行以下命令：

```bash
unset https_proxy http_proxy all_proxy ALL_PROXY
```


--------------

### 关于科学上网：

要设置 `https_proxy` 环境变量为 `http://127.0.0.1:7890`，你可以使用以下方式：

临时设置，只在当前终端会话中生效（终端关闭后将失效）：

```bash
export https_proxy=http://127.0.0.1:7890
```

永久设置，对当前用户在所有终端会话中都生效：

1. 打开终端并运行以下命令编辑用户的 `~/.bashrc` 文件（如果你使用的是 Bash shell）：

```bash
nano ~/.bashrc
```

2. 在文件末尾添加以下行：

```bash
export https_proxy=http://127.0.0.1:7890
```

3. 按下 `Ctrl + O` 保存文件，然后按下 `Ctrl + X` 关闭文本编辑器。

4. 使更改生效：

```bash
source ~/.bashrc
```

这样，在每次打开新终端会话时，`https_proxy` 环境变量都会被自动设置为 `http://127.0.0.1:7890`。

请注意，如果你使用的是其他 shell，例如 Zsh，那么相应的配置文件可能不是 `~/.bashrc`，而是 `~/.zshrc`。在这种情况下，请将上述步骤中的文件名替换为正确的配置文件名。

---------

## ipynb转md

cd 到存放ipynb文件的文件夹里，输入以下命令：

```jupyter nbconvert --to markdown test.ipynb```

------




## 常见的Linux操作和命令：

1. **文件和目录操作**：
   - `ls`：列出当前目录的文件和子目录。
   - `cd`：改变当前工作目录。
   - `pwd`：显示当前工作目录的完整路径。
   - `mkdir`：创建新目录。
   - `touch`：创建新文件。
   - `cp`：复制文件或目录。
   - `mv`：移动或重命名文件或目录。
   - `rm`：删除文件或目录。

2. **文件查看和编辑**：
   - `cat`：查看文件内容。
   - `more` 或 `less`：分页查看文件内容。
   - `nano` 或 `vim`：文本编辑器。
   - `head` 和 `tail`：查看文件的开头和结尾部分。

3. **系统信息**：
   - `uname`：显示系统信息。
   - `top` 或 `htop`：查看系统资源使用情况。
   - `df`：显示磁盘空间使用情况。
   - `free`：显示内存使用情况。

4. **进程管理**：
   - `ps`：列出当前正在运行的进程。
   - `kill`：终止进程。
   - `bg` 和 `fg`：在后台和前台运行进程。

5. **用户和权限**：
   - `sudo`：以超级用户权限运行命令。
   - `useradd` 和 `userdel`：创建和删除用户。
   - `passwd`：更改用户密码。
   - `chmod`：更改文件和目录权限。
   - `chown`：更改文件和目录的所有者。

6. **包管理**：
   - `apt` 或 `apt-get`：Debian/Ubuntu系统的包管理工具。
   - `yum`：CentOS/RHEL系统的包管理工具。
   - `dnf`：Fedora系统的包管理工具。

7. **网络操作**：
   - `ping`：测试与主机的网络连接。
   - `ifconfig` 或 `ip`：查看和配置网络接口。
   - `ssh`：远程登录其他计算机。
   - `wget` 或 `curl`：下载文件或内容。

8. **压缩和解压缩**：
   - `tar`：打包和解包文件。
   - `gzip` 和 `gunzip`：压缩和解压缩文件。
   - `zip` 和 `unzip`：创建和解压ZIP文件。

9. **查找文件**：
   - `find`：按名称、类型等条件查找文件。
   - `grep`：在文本文件中搜索特定字符串。

10. **系统日志**：
    - `dmesg`：显示系统消息日志。
    - `journalctl`：查看systemd日志。

这只是Linux命令的一小部分，Linux系统非常强大且具有高度可定制性，可以根据需要执行各种任务。要了解有关特定命令的详细信息，你可以在终端中运行`man`命令，后跟命令名称，以查看其手册页。例如：`man ls` 或 `man mkdir`。这将提供有关该命令的详细信息和选项。