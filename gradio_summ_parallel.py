import gradio as gr
import re


from gradio.mix import Parallel
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii, Chinese characters
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single
    return text

pipeline_summ = pipeline(
    "summarization",
    model="facebook/bart-large-cnn", # switch out to "t5-small" etc if you wish
    tokenizer="facebook/bart-large-cnn", # as above
    framework="pt",
)

# First of 2 summarization function
def fb_summarizer(text):
    input_text = clean_text(text)
    results = pipeline_summ(input_text)
    return results[0]["summary_text"]

# First of 2 Gradio apps that we'll put in "parallel"
summary1 = gr.Interface(
    fn=fb_summarizer,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Summary by FB/Bart-large"),
)

model_name = "google/pegasus-cnn_dailymail" # Pegasus has a few variations; switch out as required

# Second of 2 summarization function
def google_summarizer(text):
    input_text = clean_text(text)

    tokenizer_pegasus = AutoTokenizer.from_pretrained(model_name)

    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    batch = tokenizer_pegasus.prepare_seq2seq_batch(
        input_text, truncation=True, padding="longest", return_tensors="pt"
    )
    translated = model_pegasus.generate(**batch)

    pegasus_summary = tokenizer_pegasus.batch_decode(
        translated, skip_special_tokens=True
    )

    return pegasus_summary[0]

# Second of 2 Gradio apps that we'll put in "parallel"
summary2 = gr.Interface(
    fn=google_summarizer,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Summary by Google/Pegasus-CNN-Dailymail"),
)

Parallel(
    summary1,
    summary2,
    title="Compare 2 AI Summarizers",
    inputs=gr.inputs.Textbox(lines=20, label="Paste some English text here"),
).launch()
