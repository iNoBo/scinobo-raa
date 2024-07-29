import os
import pandas as pd
import gradio as gr
import requests as req
from raa.pipeline.inference import extract_research_artifacts_text_list, extract_research_artifacts_text_list_fast_mode, \
                                   extract_research_artifacts_pdf, extract_research_artifacts_pdf_fast_mode, \
                                   extract_research_artifacts_doimode, extract_research_artifacts_doimode_fast_mode
from raa.pipeline.inference import convert_mentions_to_table, convert_clusters_to_tables

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
            return None, pd.DataFrame(results['research_artifacts'])
        else:
            results = extract_research_artifacts_text_list([[snippet]], split_sentences=split_sentences, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)
            if perform_deduplication:
                research_artifacts, mentions = convert_clusters_to_tables(results['research_artifacts']['grouped_clusters'], text_mode=True, return_df=True)
                return research_artifacts, mentions
            else:
                mentions = convert_mentions_to_table(results['research_artifacts']['candidates_metadata'], text_mode=True, return_df=True)
                return None, mentions
    except Exception as e:
        results = {'error': str(e)}
        return None, results

def analyze_pdf(pdf_file, fast_mode, filter_paragraphs, perform_deduplication, insert_fast_mode_gazetteers, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        if fast_mode:
            results = extract_research_artifacts_pdf_fast_mode(pdf_file)
            return None, pd.DataFrame(results['research_artifacts'])
        else:
            results = extract_research_artifacts_pdf(pdf_file, filter_paragraphs=filter_paragraphs, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)
            if perform_deduplication:
                research_artifacts, mentions = convert_clusters_to_tables(results['research_artifacts']['grouped_clusters'], text_mode=False, return_df=True)
                return research_artifacts, mentions
            else:
                mentions = convert_mentions_to_table(results['research_artifacts']['candidates_metadata'], text_mode=False, return_df=True)
                return None, mentions
    except Exception as e:
        results = {'error': str(e)}
        return None, results

def analyze_input_doi(doi: str | None, fast_mode, filter_paragraphs, perform_deduplication, insert_fast_mode_gazetteers, progress=gr.Progress(track_tqdm=True)):
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

        # Get the data
        data = response.json()

        # Move the 'Abstract' in the 'sections' list to the first position
        if data['sections'][-1][0] == 'Abstract':
            data['sections'].insert(0, data['sections'].pop())
        
        # Call the function to extract the research artifacts
        if fast_mode:
            results = extract_research_artifacts_doimode_fast_mode(data)
            return data, None, pd.DataFrame(results['research_artifacts'])
        else:
            if perform_deduplication:
                results = extract_research_artifacts_doimode(data, filter_paragraphs=filter_paragraphs, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)
                research_artifacts, mentions = convert_clusters_to_tables(results['research_artifacts']['grouped_clusters'], text_mode=False, return_df=True)
                return data, research_artifacts, mentions
            else:
                results = extract_research_artifacts_doimode(data, filter_paragraphs=filter_paragraphs, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers)
                mentions = convert_mentions_to_table(results['research_artifacts']['candidates_metadata'], text_mode=False, return_df=True)
                return data, None, mentions
    except Exception as e:
        results = {'error': str(e)}
        return results, None, None


# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo RAA - Text Mode")
    text_input = gr.Textbox(label="Snippet")
    fast_mode_toggle = gr.Checkbox(label="Fast Mode", value=False, interactive=True)
    split_sentences_toggle = gr.Checkbox(label="Split Sentences", value=False, interactive=True)
    perform_dedup_toggle = gr.Checkbox(label="Perform Deduplication", value=True, interactive=True)
    fast_mode_gazetteers_toggle = gr.Checkbox(label="Insert Fast Mode Gazetteers", value=False, interactive=True)
    process_text_button = gr.Button("Process")

    with gr.Tabs() as output_tabs:
        with gr.TabItem("Research Artifacts"):
            text_output_1 = gr.DataFrame(label="Research Artifacts", headers=['RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Percentage', 'Owned Score', 'Reused', 'Reused Percentage', 'Reused Score', 'Licenses', 'Versions', 'URLs', 'Citations', 'Mentions Count'], row_count=1)
        with gr.TabItem("Mentions"):
            text_output_2 = gr.DataFrame(label="Mentions", headers=['Mention ID', 'RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Score', 'Reused', 'Reused Score', 'License', 'Version', 'URLs', 'Citations', 'Section', 'Indices', 'Trigger', 'Mention'], row_count=1)

    def update_visibility(fast_mode_toggle):
        if fast_mode_toggle:
            return gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True)
    
    fast_mode_toggle.change(update_visibility, inputs=[fast_mode_toggle], outputs=[perform_dedup_toggle, fast_mode_gazetteers_toggle])

    process_text_button.click(analyze_text, inputs=[text_input, fast_mode_toggle, split_sentences_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle], outputs=[text_output_1, text_output_2])


# Define the interface for the second tab (PDF Analysis)
with gr.Blocks() as pdf_analysis:
    gr.Markdown("### SciNoBo RAA - PDF Mode")
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    fast_mode_toggle = gr.Checkbox(label="Fast Mode", value=False, interactive=True)
    filter_paragraphs_toggle = gr.Checkbox(label="Filter Paragraphs", value=True, interactive=True)
    perform_dedup_toggle = gr.Checkbox(label="Perform Deduplication", value=True, interactive=True)
    fast_mode_gazetteers_toggle = gr.Checkbox(label="Insert Fast Mode Gazetteers", value=False, interactive=True)
    process_pdf_button = gr.Button("Process")

    with gr.Tabs() as output_tabs:
        with gr.TabItem("Research Artifacts"):
            pdf_output_1 = gr.DataFrame(label="Research Artifacts", headers=['RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Percentage', 'Owned Score', 'Reused', 'Reused Percentage', 'Reused Score', 'Licenses', 'Versions', 'URLs', 'Citations', 'Mentions Count'], row_count=1)
        with gr.TabItem("Mentions"):
            pdf_output_2 = gr.DataFrame(label="Mentions", headers=['Mention ID', 'RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Score', 'Reused', 'Reused Score', 'License', 'Version', 'URLs', 'Citations', 'Section', 'Indices', 'Trigger', 'Mention'], row_count=1)


    def update_visibility(fast_mode_toggle):
        if fast_mode_toggle:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    fast_mode_toggle.change(update_visibility, inputs=[fast_mode_toggle], outputs=[filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle])

    process_pdf_button.click(analyze_pdf, inputs=[pdf_input, fast_mode_toggle, filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle], outputs=[pdf_output_1, pdf_output_2])

# Define the interface for the second tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### SciNoBo RAA - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier")
    fast_mode_toggle = gr.Checkbox(label="Fast Mode", value=False, interactive=True)
    filter_paragraphs_toggle = gr.Checkbox(label="Filter Paragraphs", value=True, interactive=True)
    perform_dedup_toggle = gr.Checkbox(label="Perform Deduplication", value=True, interactive=True)
    fast_mode_gazetteers_toggle = gr.Checkbox(label="Insert Fast Mode Gazetteers", value=False, interactive=True)
    process_doi_button = gr.Button("Process")
    
    doi_metadata = gr.JSON(label="DOI Metadata")

    with gr.Tabs() as output_tabs:
        with gr.TabItem("Research Artifacts"):
            doi_output_1 = gr.DataFrame(label="Research Artifacts", headers=['RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Percentage', 'Owned Score', 'Reused', 'Reused Percentage', 'Reused Score', 'Licenses', 'Versions', 'URLs', 'Citations', 'Mentions Count'], row_count=1)
        with gr.TabItem("Mentions"):
            doi_output_2 = gr.DataFrame(label="Mentions", headers=['Mention ID', 'RA Cluster', 'Research Artifact', 'Type', 'Research Artifact Score', 'Owned', 'Owned Score', 'Reused', 'Reused Score', 'License', 'Version', 'URLs', 'Citations', 'Section', 'Indices', 'Trigger', 'Mention'], row_count=1)

    def update_visibility(fast_mode_toggle):
        if fast_mode_toggle:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    fast_mode_toggle.change(update_visibility, inputs=[fast_mode_toggle], outputs=[filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle])

    process_doi_button.click(analyze_input_doi, inputs=[doi_input, fast_mode_toggle, filter_paragraphs_toggle, perform_dedup_toggle, fast_mode_gazetteers_toggle], outputs=[doi_metadata, doi_output_1, doi_output_2])

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, pdf_analysis, doi_mode], ["Text Mode", "PDF Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch()
