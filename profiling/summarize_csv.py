#!/usr/bin/env python

import os
import sys
import csv
import json
import argparse
import torch
from typing import Dict, List

class ProfilerSummary:
    def __init__(self, file_path: str, categories: dict):
        self.file_path = file_path
        self.others_details = {}
        self.lookup_table = categories['keywords']

        self.categories = {}
        for category, data in self.lookup_table.items():
            self.categories[category] = [0, 0]
        self.categories['others'] = [0, 0]

    def read_and_classify_csv(self):
        """Reads a CSV file, classifies each row, and aggregates data into categories."""
        with open(self.file_path, newline='') as csvfile:
            profreader = csv.reader(csvfile, delimiter=',')
            next(profreader, None)  # skip header
            for row in profreader:
                key = row[0]
                calls = int(row[1])
                duration = int(row[2])

                if duration < 0:
                    print(f"Warning: Negative duration detected for {key}. Ignoring this entry.")
                    continue

                category = self.categorize_key(key)
                if category == 'others':
                    self.others_details[key] = [self.others_details.get(key, [0, 0])[0] + calls,
                                                self.others_details.get(key, [0, 0])[1] + duration]
                self.categories[category][0] += calls
                self.categories[category][1] += duration

    def categorize_key(self, key):
        for category, data in self.lookup_table.items():
            key_lower = key.lower()
            if any(keyword.lower() in key_lower for keyword in data):
                return category
        return 'others'

    def display_results(self):
        """Displays the profiling results and sorts it by percentage of total."""

        # Calculate the total duration for percentage calculations
        total_duration = sum(val[1] for val in self.categories.values())

        # Prepare data for sorting
        results = []
        for key, val in self.categories.items():
            calls, duration = val
            avg_duration_per_call = duration / calls if calls else 0
            percentage_of_total = (duration / total_duration * 100) if total_duration else 0
            results.append((key, calls, duration, avg_duration_per_call, percentage_of_total))

        # Sort the results by percentage of the total duration (descending)
        sorted_results = sorted(results, key=lambda x: x[4], reverse=True)

        # Header
        print(f"{'Category':<20} {'Calls':<10} {'Total Duration':<15} {'Avg Duration/Call':<20} {'% of Total Duration':<20}")
        print("-" * 85)

        # Data Rows
        for result in sorted_results:
            print(f"{result[0]:<20} {result[1]:<10} {result[2]:<15} {result[3]:<20.2f} {result[4]:<20.2f}")

        # Sort 'others' details by total duration (descending)
        sorted_others_details = sorted(self.others_details.items(), key=lambda x: x[1][1], reverse=True)

        # Summary of 'others' category
        print("\nDetailed Summary of 'others' Category:")
        print(f"{'Function':<100} {'Calls':<10} {'Total Duration':<15}")
        print("-" * 125)
        for func, details in sorted_others_details:
            calls, duration = details
            print(f"{func:<100} {calls:<10} {duration:<15}")

def load_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process profiling data.')
    parser.add_argument('file_path', type=str, help='Full path to the CSV file')
    args = parser.parse_args()

    config_dir = os.path.dirname(sys.argv[0])
    if torch.cuda.is_available() and torch.version.hip is not None:
        categories = load_config(config_dir + "/categories_rocm.json")
    else:
        categories = load_config(config_dir + "/categories_cuda.json")

    profiler = ProfilerSummary(args.file_path, categories)

    # Read the CSV file, classify the data, and aggregate it
    profiler.read_and_classify_csv()

    # Display the results
    profiler.display_results()

if __name__ == "__main__":
    main()
