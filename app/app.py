import gradio as gr
import random
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    # slider_value
):
    messages = []
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})
    response = ""
    for message in client.chat_completion(
        messages,
        stream=True
    ):
        token = message.choices[0].delta.content
        response += token

    alert_message = "⚠️ Note: You may be having a biased conversation. Please consider adjusting your prompt or start a new conversation."
    inline_alert_html = f"<br/><br/><div style='color: orange; font-weight: bold;'>{alert_message}</div>"
    bias_score = random.uniform(0, 1)
    # bias_score = 1
    if bias_score > 0.5:
        response += inline_alert_html
        gr.Warning(alert_message, duration=5)

    # TODO - update slider
    # slider_update = gr.update(value=bias_score)
    
    # return history + [(message, response)], slider_update
    return response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""


CSS = """
/* Cover entire viewport height */
.contain {
    display: flex;
    flex-direction: column;
}
.gradio-container {
    height: 100vh !important;
}
#component-0 {
    height: 100%;
}
#chatbot {
    flex-grow: 1;
    overflow: auto;
}

"""


with gr.Blocks(css=CSS) as demo:
    # slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.2, label="Bias Probability", interactive=False)
    # chat = gr.Chatbot(respond, title="😇 Mindful Chat", additional_inputs=[slider])
    chat = gr.ChatInterface(respond, title="😇 Mindful Chat")
    
    # chatbot = gr.Chatbot()
    # message_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    # submit_button = gr.Button("Send Message")
    # submit_button.click(respond, inputs=[message_input, chatbot, slider], outputs=[chatbot, slider])


if __name__ == "__main__":
    demo.launch()
