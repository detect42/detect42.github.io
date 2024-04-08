## ZSH

conda 命令行 激活环境: source activate base
conda 命令行 推出环境: conda deactivate

打开关闭代理：

alias proxy="
    export https_proxy=http://127.0.0.1:7890;
    export http_proxy=http://127.0.0.1:7890;
    export all_proxy=socks5://127.0.0.1:7890"
alias unproxy="
    unset http_proxy;
    unset https_proxy;
    unset all_proxy;
    unset no_proxy;
    unset HTTP_PROXY;
    unset HTTPS_PROXY;
    unset ALL_PROXY;
    unset NO_PROXY"

code + 文件夹或者文件 打开对应vscode


## MAC:

Command + c copy
Command + option + v 剪贴
Command + v 粘贴
Command + Shift + . 显示隐藏文件夹**

## VScode

Cmd + \  拆分窗口
Cmd + j 切换控制台/编辑器
Cmd + 左右 行首/行末
Cmd + / 添加注释

Cmd + Shift + f 全局搜索(当前项目文件夹&打开的文件)
Cmd + p 当前项目搜索文件名
Cmd + f 当前文件搜索
Cmd + U 撤销光标饿移动和选择

替换自动黏贴图片并设定size大小：





光标 + F2 重命名变量
Option + Shift + F 代码格式化

命令面板 + polacode 代码截图

Cmd + K then Z 禅模式

### Vim

super esc
Easymotion s代替一个字母的跳转，f则使用两个字母跳转
leader + i 显示代码信息
gd：跳转到对象定义
ge: goto error(warning, info)， 跳转到项目有 Error，Warning 或 Info 的地方
L: 行末
H：行首