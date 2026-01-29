import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw


ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "images"


def dummy_generate_code(prompt: str, language: str, temperature: float, max_tokens: int) -> str:
    """
    这里只做前端占位，不真正调用 Qwen3-Coder-30B-A3B-Instruct-MLX-4bit 模型。
    为了便于可视化展示，我们返回一段包含“推理中”“示意输出”的占位代码。
    """
    header = f"// Qwen3-Coder-30B-A3B-Instruct-MLX-4bit 示例输出（语言：{language}）\n"
    body = (
        "// 注意：当前 WebUI 仅用于界面与交互流程演示，未真实加载模型权重。\n"
        "// 在实际部署中，可在此处调用 MLX 推理接口或远程推理服务。\n\n"
    )
    user_prompt = f"// 用户输入的自然语言需求：{prompt or '（示例）实现一个简单的斐波那契函数'}\n\n"

    if language.lower() in {"python", "py"}:
        code = (
            "def fibonacci(n: int) -> int:\n"
            '    """示例：使用 Qwen3-Coder 自动生成的斐波那契函数占位实现。"""\n'
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        )
    else:
        code = (
            "int fibonacci(int n) {\n"
            "    // 示例：使用 Qwen3-Coder 自动生成的斐波那契函数占位实现。\n"
            "    if (n <= 1) return n;\n"
            "    int a = 0, b = 1;\n"
            "    for (int i = 2; i <= n; ++i) {\n"
            "        int c = a + b;\n"
            "        a = b;\n"
            "        b = c;\n"
            "    }\n"
            "    return b;\n"
            "}\n"
        )

    meta = (
        f"\n// 推理参数示意：temperature={temperature}, max_tokens={max_tokens}\n"
        "// 实际部署时可根据任务类型与代码长度进行调优。\n"
    )
    return header + body + user_prompt + code + meta


def get_placeholder_architecture_image() -> Image.Image:
    """
    尝试从 images 目录中加载占位示意图；
    如果不存在，则生成一张简单的模型架构示意图片。
    """
    placeholder = None
    for name in ["qwen_architecture.png", "flux2_model_page.png"]:
        candidate = IMAGE_DIR / name
        if candidate.exists():
            placeholder = candidate
            break

    if placeholder is not None:
        return Image.open(placeholder)

    img = Image.new("RGB", (960, 540), color=(20, 24, 40))
    draw = ImageDraw.Draw(img)
    title = "Qwen3-Coder-30B-A3B-Instruct-MLX-4bit\n模型架构与推理流程示意图（占位）"
    draw.text((40, 40), title, fill=(235, 235, 245))
    draw.text(
        (40, 160),
        "输入：自然语言需求 / 部分代码片段 / 项目结构\n"
        "编码：多粒度代码与上下文表示，支持 256K 以上长上下文\n"
        "推理：MoE 专家路由 + GQA 注意力，高效激活少量参数\n"
        "输出：补全后的代码、重构建议、调试思路与注释说明\n\n"
        "当前页面仅展示交互流程，不加载真实模型权重。",
        fill=(210, 210, 220),
    )
    return img


def build_interface():
    with gr.Blocks(title="Qwen3-Coder-30B-A3B-Instruct-MLX-4bit WebUI") as demo:
        gr.Markdown(
            """
            # Qwen3-Coder-30B-A3B-Instruct-MLX-4bit WebUI

            本界面用于演示基于 Qwen3-Coder-30B-A3B-Instruct-MLX-4bit 模型的代码生成与仓库级理解的典型交互流程。
            当前示例仅提供**前端可视化界面**和参数配置，不在本地下载和加载真实模型权重，
            便于在资源受限环境下快速体验界面设计与操作路径。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="自然语言需求 / 代码编辑指令",
                    placeholder="例如：请为一个微服务项目设计基于 JWT 的用户认证中间件，并给出 Python 代码示例。",
                    lines=6,
                )
                language = gr.Dropdown(
                    choices=["Python", "C++", "Java", "Go"],
                    value="Python",
                    label="目标编程语言",
                )
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.2,
                        value=0.7,
                        step=0.05,
                        label="采样温度（Temperature，仅界面参数）",
                    )
                    max_tokens = gr.Slider(
                        minimum=128,
                        maximum=8192,
                        value=1024,
                        step=128,
                        label="最大生成长度 Max New Tokens（仅界面参数）",
                    )
                run_btn = gr.Button("生成/补全代码（示意，不实际推理）", variant="primary")

                gr.Markdown(
                    """
                    在真实部署环境中，用户可以将本 WebUI 作为轻量级前端，
                    通过后端服务调用 MLX 版本的 Qwen3-Coder-30B-A3B-Instruct-MLX-4bit 模型，
                    以支持仓库级代码理解、跨文件重构与自动化调试等高级功能。
                    """
                )

            with gr.Column(scale=4):
                code_output = gr.Code(
                    label="模型生成的代码（示意输出）",
                    language="python",
                )
                arch_image = gr.Image(
                    label="模型架构 / 推理流程示意图（占位图，无需真实模型）",
                    type="pil",
                )

        run_btn.click(
            fn=dummy_generate_code,
            inputs=[prompt, language, temperature, max_tokens],
            outputs=[code_output],
        )

        demo.load(
            fn=lambda: get_placeholder_architecture_image(),
            inputs=None,
            outputs=[arch_image],
        )

        gr.Markdown(
            """
            **说明：**

            为了节约带宽和存储，本 WebUI 不会自动从互联网上下载任何模型权重，
            也不会在页面加载时触发长时间的推理或评测过程。开发者在后续集成时，
            可以将 `dummy_generate_code` 替换为实际的推理函数，并在后端使用 MLX、
            vLLM 或其他推理引擎进行代码生成与评估。
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
