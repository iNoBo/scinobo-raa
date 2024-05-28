import json
import sys
sys.path.append("./src")
from tqdm import tqdm
from raa.pipeline.inference import extract_artifacts_candidate_text_list


""" FUNCTIONS """

def calculate_exact_match(targets, predictions):
    errors = []

    total_instances = len(targets)
    match_count = 0

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        if target.lower() == prediction.lower():  # Case-insensitive comparison
            match_count += 1
        else:
            errors.append((i, target, prediction))

    exact_match_score = match_count / total_instances
    return exact_match_score, errors


def calculate_lenient_match(targets, predictions):
    errors = []

    total_instances = len(targets)
    match_count = 0

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        if target.lower() in prediction.lower() or prediction.lower() in target.lower():  # Case-insensitive comparison
            match_count += 1
        else:
            errors.append((i, target, prediction))

    exact_match_score = match_count / total_instances
    return exact_match_score, errors


def calculate_prf_binary(targets, predictions):
    # NOTE: this function assumes that the targets and predictions are 1 or 0

    errors = []

    # Calculate the TP, FP, FN
    tp = 0
    fp = 0
    fn = 0

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        if target == 1 and prediction == 1:
            tp += 1
        elif target == 1 and prediction == 0:
            fn += 1
            errors.append((i, target, prediction))
        elif target == 0 and prediction == 1:
            fp += 1
            errors.append((i, target, prediction))
    
    # Calculate the precision, recall, f1
    if tp + fp == 0:
        precision = 0  # Handle division by zero
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0  # Handle division by zero
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0  # Handle division by zero
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1, errors


with open('eval/data/artifact_extraction_hybrid_data_v1_fix_aug_transformed_test_templateformat.json', encoding='utf-8') as fin:
    test_data = json.load(fin)

# test_data = test_data[:20]

predictions = []
for i, test_item in tqdm(enumerate(test_data), total=len(test_data)):
    tags_offsets = (test_item['Snippet'].find('<m>'), test_item['Snippet'].find('</m>'))
    snippet_wo_tags = test_item['Snippet'][:tags_offsets[0]] + test_item['Snippet'][tags_offsets[0]+3:tags_offsets[1]] + test_item['Snippet'][tags_offsets[1] + 4:]
    # Re-adjust snippet offsets due to the removal of the tags
    tags_offsets = (tags_offsets[0], tags_offsets[1] - 3)  # 4 is the length of "<m>"
    result = extract_artifacts_candidate_text_list(
        cand_sent_id = '{}'.format(i),
        cand_sent_text = snippet_wo_tags,
        cand_type = test_item['Type'],
        cand_trigger = '',
        cand_trigger_off = tags_offsets
    )
    # Convert to format
    valid = 'Yes' if result['artifact_answer']['Yes'] > result['artifact_answer']['No'] else 'No'
    if valid=='Yes':
        result = {
            'Valid': 'Yes',
            'Name': result['name_answer'],
            'Version': result['version_answer'],
            'License': result['license_answer'],
            'URL': result['url_answer'],
            'Ownership': result['ownership_answer_text'],
            'Usage': result['reuse_answer_text'],
        }
    else:
        result = {
            'Valid': 'No'
        }
    predictions.append(result)

# Calculate the metrics for the predictions and targets, the format is the following:
# {Valid: Yes or No, Name: text, Version: text, License: text, URL: text, Ownership: Yes or No, Usage: Yes or No}

# For Valid, Ownership and Usage we use the binary precision, recall, f1
# For Name, Version, License, URL we use the binary precision, recall, f1 for the identification (whether they are N/A or not) and exact match and lenient match for the extraction (the text itself)

valid_indices_map = []
valid_targets = []
valid_predictions = []
name_targets = []
name_predictions = []
version_targets = []
version_predictions = []
license_targets = []
license_predictions = []
url_targets = []
url_predictions = []
ownership_targets = []
ownership_predictions = []
usage_targets = []
usage_predictions = []
format_mistakes_indices = set()
for i, (target, prediction) in enumerate(zip(test_data, predictions)):
    valid_targets.append(1 if 'Yes' in target['Valid'] else 0)
    valid_predictions.append(1 if 'Yes' in prediction['Valid'] else 0)
    if 'Yes' in target['Valid'] and 'Yes' in prediction['Valid']:
        valid_indices_map.append(i)
        # Exclude all the mistakes of the format
        # TODO: MAYBE A BETTER WAY TO HANDLE THIS???
        if 'Name' in prediction:
            name_targets.append(target['Name'])
            name_predictions.append(prediction['Name'])
        else:
            format_mistakes_indices.add(i)
        if 'Version' in prediction:
            version_targets.append(target['Version'])
            version_predictions.append(prediction['Version'])
        else:
            format_mistakes_indices.add(i)
        if 'License' in prediction:
            license_targets.append(target['License'])
            license_predictions.append(prediction['License'])
        else:
            format_mistakes_indices.add(i)
        if 'URL' in prediction:
            url_targets.append(target['URL'])
            url_predictions.append(prediction['URL'])
        else:
            format_mistakes_indices.add(i)
        if 'Ownership' in prediction:
            ownership_targets.append(1 if 'Yes' in target['Ownership'] else 0)
            ownership_predictions.append(1 if 'Yes' in prediction['Ownership'] else 0)
        else:
            format_mistakes_indices.add(i)
        if 'Usage' in prediction:
            usage_targets.append(1 if 'Yes' in target['Usage'] else 0)
            usage_predictions.append(1 if 'Yes' in prediction['Usage'] else 0)
        else:
            format_mistakes_indices.add(i)

# Calculate metrics for Valid
valid_precision, valid_recall, valid_f1, valid_errors = calculate_prf_binary(valid_targets, valid_predictions)

# Calculate metrics for Name
name_precision, name_recall, name_f1, name_errors = calculate_prf_binary([1 if 'N/A' in t else 0 for t in name_targets], [1 if 'N/A' in p else 0 for p in name_predictions])
name_exact_match, name_exact_match_errors = calculate_exact_match(name_targets, name_predictions)
name_lenient_match, name_lenient_match_errors = calculate_lenient_match(name_targets, name_predictions)

# Calculate metrics for Version
version_precision, version_recall, version_f1, version_errors = calculate_prf_binary([1 if 'N/A' in t else 0 for t in version_targets], [1 if 'N/A' in p else 0 for p in version_predictions])
version_exact_match, version_exact_match_errors = calculate_exact_match(version_targets, version_predictions)
version_lenient_match, version_lenient_match_errors = calculate_lenient_match(version_targets, version_predictions)

# Calculate metrics for License
license_precision, license_recall, license_f1, license_errors = calculate_prf_binary([1 if 'N/A' in t else 0 for t in license_targets], [1 if 'N/A' in p else 0 for p in license_predictions])
license_exact_match, license_exact_match_errors = calculate_exact_match(license_targets, license_predictions)
license_lenient_match, license_lenient_match_errors = calculate_lenient_match(license_targets, license_predictions)

# Calculate metrics for URL
url_precision, url_recall, url_f1, url_errors = calculate_prf_binary([1 if 'N/A' in t else 0 for t in url_targets], [1 if 'N/A' in p else 0 for p in url_predictions])
url_exact_match, url_exact_match_errors = calculate_exact_match(url_targets, url_predictions)
url_lenient_match, url_lenient_match_errors = calculate_lenient_match(url_targets, url_predictions)

# Calculate metrics for Ownership
ownership_precision, ownership_recall, ownership_f1, ownership_errors = calculate_prf_binary(ownership_targets, ownership_predictions)

# Calculate metrics for Usage
usage_precision, usage_recall, usage_f1, usage_errors = calculate_prf_binary(usage_targets, usage_predictions)

# Add all results into a dictionary

results = {
    'Valid': {
        'precision': valid_precision,
        'recall': valid_recall,
        'f1': valid_f1,
        'errors': valid_errors
    },
    'Name': {
        'precision': name_precision,
        'recall': name_recall,
        'f1': name_f1,
        'exact_match': name_exact_match,
        'lenient_match': name_lenient_match,
        'errors': name_errors,
        'exact_match_errors': name_exact_match_errors,
        'lenient_match_errors': name_lenient_match_errors
    },
    'Version': {
        'precision': version_precision,
        'recall': version_recall,
        'f1': version_f1,
        'exact_match': version_exact_match,
        'lenient_match': version_lenient_match,
        'errors': version_errors,
        'exact_match_errors': version_exact_match_errors,
        'lenient_match_errors': version_lenient_match_errors
    },
    'License': {
        'precision': license_precision,
        'recall': license_recall,
        'f1': license_f1,
        'exact_match': license_exact_match,
        'lenient_match': license_lenient_match,
        'errors': license_errors,
        'exact_match_errors': license_exact_match_errors,
        'lenient_match_errors': license_lenient_match_errors
    },
    'URL': {
        'precision': url_precision,
        'recall': url_recall,
        'f1': url_f1,
        'exact_match': url_exact_match,
        'lenient_match': url_lenient_match,
        'errors': url_errors,
        'exact_match_errors': url_exact_match_errors,
        'lenient_match_errors': url_lenient_match_errors
    },
    'Ownership': {
        'precision': ownership_precision,
        'recall': ownership_recall,
        'f1': ownership_f1,
        'errors': ownership_errors
    },
    'Usage': {
        'precision': usage_precision,
        'recall': usage_recall,
        'f1': usage_f1,
        'errors': usage_errors
    }
}

# Save the results into a file
with open('eval/output/flan_t5_test_results.json', 'w', encoding='utf-8') as fout:
    json.dump(results, fout, indent=1)

print()
