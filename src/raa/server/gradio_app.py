import gradio as gr
from raa.pipeline.inference import extract_research_artifacts_text_list, extract_research_artifacts_text_list_fast_mode, extract_research_artifacts_pdf, extract_research_artifacts_pdf_fast_mode

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

# Define the interface for the third tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### SciNoBo RAA - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier", interactive=False)
    gr.HTML("<span style='color:red;'>This functionality is not ready yet.</span>")

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, pdf_analysis, doi_mode], ["Text Mode", "PDF Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch()
