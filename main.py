import gradio as gr
from chatModel import openai_chat

# 创建一个Gradio界面
iface = gr.Interface(
    fn=openai_chat,  # 绑定的函数
    inputs="text",  # 输入类型是文本
    outputs="markdown",  # 输出类型是文本
    title="chatBot",  # 界面的标题
    description="输入一些文本，看看他如何回复你。"  # 界面的描述
)

# 启动Gradio界面
iface.launch()

