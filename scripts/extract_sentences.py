#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def main():
    parser = argparse.ArgumentParser(
        description="Extract sentences from a CSV file based on one or more query words or phrases."
    )
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--query', type=str, nargs='+', required=True, 
                        help="One or more query words or phrases to search for. Enclose multi-word expressions in quotes.")
    parser.add_argument('--whole_word', action='store_true', help="Match the query as a whole word only")
    args = parser.parse_args()

    # Build regex patterns for each query.
    patterns = []
    for q in args.query:
        if args.whole_word:
            patterns.append(r'\b' + re.escape(q) + r'\b')
        else:
            patterns.append(re.escape(q))
    # Combine patterns with an OR operator
    final_pattern = "(" + "|".join(patterns) + ")"

    # Load the CSV file.
    df = pd.read_csv(args.input_csv)

    # Filter rows in the "Content" column where the pattern appears (case-insensitive).
    mask = df['Content'].str.contains(final_pattern, case=False, regex=True, na=False)
    filtered_df = df[mask]

    # Write the filtered DataFrame to CSV.
    filtered_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"Extracted {len(filtered_df)} sentences to {args.output_csv}")

if __name__ == '__main__':
    main()
