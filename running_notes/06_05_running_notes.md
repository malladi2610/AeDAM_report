#### Question 1: About the improvement of the JSON parser

Hey, I need your help to modify this python script that parses the mapsapce created which is in json format.

Here is the script
"import json
import ijson

def filter_mappings(json_file, output_file):
    # Dictionary to store all spatial data
    spatial_data = {}
    # List to store all mappings with their spatial keys for finding max_I
    all_mappings = []

    # First pass: Parse the JSON and collect all data
    with open(json_file, 'rb') as f_in:
        for spatial_key, spatial_dict in ijson.kvitems(f_in, ''):
            # Extract required fields
            layer_id = spatial_dict['layer_id']
            spatial_mapping = spatial_dict['spatial_mapping']
            temporal_mappings = spatial_dict['temporal_mappings']
            
            # Store in spatial_data
            spatial_data[spatial_key] = {
                'layer_id': layer_id,
                'spatial_mapping': spatial_mapping,
                'temporal_mappings': temporal_mappings
            }
            
            # Collect all mappings with their spatial_key
            for mapping in temporal_mappings:
                all_mappings.append((spatial_key, mapping))

    # Find the maximum 'I' value across all mappings
    max_I = max(mapping['mac_level_data_stationary_cycle']['I'] for _, mapping in all_mappings)

    # Second step: Filter mappings and build the output dictionary
    filtered_data = {}
    for spatial_key, mapping in all_mappings:
        if mapping['mac_level_data_stationary_cycle']['I'] == max_I:
            # If this spatial_key isn't in filtered_data yet, initialize it
            if spatial_key not in filtered_data:
                filtered_data[spatial_key] = {
                    'layer_id': spatial_data[spatial_key]['layer_id'],
                    'spatial_mapping': spatial_data[spatial_key]['spatial_mapping'],
                    'temporal_mappings': []
                }
            # Append the mapping to the temporal_mappings list
            filtered_data[spatial_key]['temporal_mappings'].append(mapping)

    # Write the filtered data to the output file with indentation
    with open(output_file, 'w') as f_out:
        json.dump(filtered_data, f_out, indent=4, ensure_ascii=False)

# Usage
if __name__ == "__main__":
    json_file = 'all_mappings.json'    # Replace with your input file path
    output_file = 'filtered.json'  # Replace with your output file path
    filter_mappings(json_file, output_file)"

Right now it has two problems
1. My data has a class called spatial_0...spatial_21 and so on. The above code only works for a single spatial class and if I pass the entire mapping with all the mappings it would just select the one combinations which has the "mac_level_data_stationary_cycle" the greatest of all.

Where as I need the script to find all the combinations within a single class from spatial_0 to spatial_21 and return me all the valid combinations which has the highest mac_level_data_stationary_cycle for that particular class. Once that class is done then the next class is taken and the highest "mac_level_data_stationary_cycle" of that class is taken.

This process continues until all the classes are over.
2. The parser will have problems when the input data becomes huge. I found this parser, which is a SIMDparser - https://pysimdjson.tkte.ch/index.html

Can you help me in improving the script to match my requirements and also implement it using SIMD parser. So, that it can be faster too for future inputs that will be huge.


#### Answer 1: 