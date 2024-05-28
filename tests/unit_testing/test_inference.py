""" 

This script is used to test the inference of the SciNoBo-RAA bulk inference.

"""
import unittest
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from raa.pipeline.inference import main

import argparse
from unittest.mock import patch

class TestInference(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_fast_mode(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output_fast/',
            filter_paragraphs=False,
            perform_deduplication=False,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=True,
            text_mode=False,
            text_mode_split_sentences=False,
            reevaluate=False,
            reevaluate_only=False,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Finished processing all PDF files. Output saved in:', 'examples/output_fast/')
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_w_reevaluate(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output/',
            filter_paragraphs=True,
            perform_deduplication=True,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=False,
            text_mode=False,
            text_mode_split_sentences=False,
            reevaluate=True,
            reevaluate_only=False,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Finished reevaluating all results. Output saved in:', 'examples/output/')

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_text_mode_fast_mode(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output_text_mode_fast/',
            filter_paragraphs=False,
            perform_deduplication=False,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=True,
            text_mode=True,
            text_mode_split_sentences=False,
            reevaluate=False,
            reevaluate_only=False,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Finished processing all text files. Output saved in:', 'examples/output_text_mode_fast/')
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_text_mode_w_reevaluate(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output_text_mode/',
            filter_paragraphs=True,
            perform_deduplication=True,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=False,
            text_mode=True,
            text_mode_split_sentences=False,
            reevaluate=True,
            reevaluate_only=False,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Reevaluation works only in PDF mode. Skipping reevaluation.')
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_reevaluate_only(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output/',
            filter_paragraphs=False,
            perform_deduplication=False,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=False,
            text_mode=False,
            text_mode_split_sentences=False,
            reevaluate=False,
            reevaluate_only=True,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Finished reevaluating all results. Output saved in:', 'examples/output/')

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_text_mode_reevaluate_only(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output_text_mode/',
            filter_paragraphs=False,
            perform_deduplication=False,
            insert_fast_mode_gazetteers=False,
            dataset_gazetteers=None,
            fast_mode=False,
            text_mode=True,
            text_mode_split_sentences=False,
            reevaluate=False,
            reevaluate_only=True,
            verbose=True
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with('Reevaluation works only in PDF mode. Skipping reevaluation.')

if __name__ == '__main__':
    unittest.main()