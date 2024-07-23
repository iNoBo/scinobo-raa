import os
import gradio as gr
import requests as req
from raa.pipeline.inference import extract_research_artifacts_text_list, extract_research_artifacts_text_list_fast_mode, extract_research_artifacts_pdf, extract_research_artifacts_pdf_fast_mode

# Retrieve HF space secrets
BACKEND_IP = os.getenv('BACKEND_IP')
BACKEND_PORT = os.getenv('BACKEND_PORT')
BACKEND_PATH = os.getenv('BACKEND_PATH')

# Define the functions to handle the inputs and outputs
def analyze_text(snippet, fast_mode, split_sentences, perform_deduplication, insert_fast_mode_gazetteers, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        if fast_mode:
            results = extract_research_artifacts_text_list_fast_mode([[snippet]], split_sentences=split_sentences)
        else:
            results = extract_research_artifacts_text_list([[snippet]], split_sentences=split_sentences, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)
    except Exception as e:
        results = {'error': str(e)}
    return results

def analyze_pdf(pdf_file, fast_mode, filter_paragraphs, perform_deduplication, insert_fast_mode_gazetteers, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        if fast_mode:
            results = extract_research_artifacts_pdf_fast_mode(pdf_file)['research_artifacts']
        else:
            results = extract_research_artifacts_pdf(pdf_file, filter_paragraphs=filter_paragraphs, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)['research_artifacts']
    except Exception as e:
        results = {'error': str(e)}
    return results

def analyze_input_doi(
    doi: str | None
):
    if (doi is None):
        results = {'error': 'Please provide the DOI of the publication'}
        return results
    if (doi == ''):
        results = {'error': 'Please provide the DOI of the publication'}
        return results
    try:
        url = f"http://{BACKEND_IP}:{BACKEND_PORT}{BACKEND_PATH}{doi}"
        response = req.get(url)
        response.raise_for_status()
        results = response.json()
        return results
    except Exception as e:
        results = {'error': str(e)}
    return results

# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo RAA - Text Mode")
    text_input = gr.Textbox(label="Snippet")
    fast_mode_toggle = gr.Checkbox(label="Fast Mode", value=False, interactive=True)
    split_sentences_toggle = gr.Checkbox(label="Split Sentences", value=False, interactive=True)
    perform_dedup_toggle = gr.Checkbox(label="Perform Deduplication", value=True, interactive=True)
    fast_mode_gazetteers_toggle = gr.Checkbox(label="Insert Fast Mode Gazetteers", value=False, interactive=True)
    process_text_button = gr.Button("Process")
    text_output = gr.JSON(label="Output")

    def update_visibility(fast_mode_toggle):
        if fast_mode_toggle:
            return gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True)
    
    fast_mode_toggle.change(update_visibility, inputs=[fast_mode_toggle], outputs=[perform_dedup_toggle, fast_mode_gazetteers_toggle])

    process_text_button.click(analyze_text, inputs=[text_input, fast_mode_toggle, split_sentences_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle], outputs=[text_output])

# Define the interface for the second tab (PDF Analysis)
with gr.Blocks() as pdf_analysis:
    gr.Markdown("### SciNoBo RAA - PDF Mode")
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    fast_mode_toggle = gr.Checkbox(label="Fast Mode", value=False, interactive=True)
    filter_paragraphs_toggle = gr.Checkbox(label="Filter Paragraphs", value=True, interactive=True)
    perform_dedup_toggle = gr.Checkbox(label="Perform Deduplication", value=True, interactive=True)
    fast_mode_gazetteers_toggle = gr.Checkbox(label="Insert Fast Mode Gazetteers", value=False, interactive=True)
    process_pdf_button = gr.Button("Process")
    pdf_output = gr.JSON(label="Output")

    def update_visibility(fast_mode_toggle):
        if fast_mode_toggle:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    fast_mode_toggle.change(update_visibility, inputs=[fast_mode_toggle], outputs=[filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle])

    process_pdf_button.click(analyze_pdf, inputs=[pdf_input, fast_mode_toggle, filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle], outputs=[pdf_output])

# Define the interface for the second tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### Sustainable Development Goal (SDG) Classifier - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier")
    process_doi_button = gr.Button("Process")
    doi_output = gr.JSON(label="Output")
    process_doi_button.click(analyze_input_doi, inputs=[doi_input], outputs=[doi_output])

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, pdf_analysis, doi_mode], ["Text Mode", "PDF Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch()
