# Functions to find pairwise 1 mut differences between sequences and compare their labels
import pandas as pd
import matplotlib.pyplot as plt

def find_one_char_diff_pairs(df_train, df_test):
    placeholder = '*'
    seen = {}
    pairs = []

    # Process the first set: store modified sequences with their identifiers
    for train_sequence in df_train['Slide']:
        for i in range(len(train_sequence)):
            modified = train_sequence[:i] + placeholder + train_sequence[i+1:]
            if modified in seen:
                seen[modified].append(train_sequence)
            else:
                seen[modified] = [train_sequence]

    # Process the second set and look for matches in the dictionary
    for test_sequence in df_test['Slide']:
        for i in range(len(test_sequence)):
            modified = test_sequence[:i] + placeholder + test_sequence[i+1:]
            if modified in seen:
                for train_seq in seen[modified]:
                    pairs.append([train_seq, test_sequence])

    return pairs

def get_leak_info(df_train,df_test, pairs):
    train_lookup_dict = df_train.set_index('Slide').to_dict(orient='index')
    test_lookup_dict = df_test.set_index('Slide').to_dict(orient='index')
    leak_info = []
    concordance_dict = {
    '11':'++',
    '00':'--',
    '10':'+-',
    '01':'-+'}
    for train_slide, test_slide in pairs: #mixed up
        train_label = train_lookup_dict[train_slide]['y']
        train_eng = train_lookup_dict[train_slide]['Energy']

        test_label = test_lookup_dict[test_slide]['y']
        test_eng = test_lookup_dict[test_slide]['Energy']
        label_concordance = str(train_label) + str(test_label)
        label_concordance = concordance_dict[label_concordance]
        eng_diff = train_eng - test_eng
        leak_info.append([train_slide, test_slide, label_concordance, eng_diff])
    leak_df = pd.DataFrame(leak_info, columns=['train_Slide', 'test_Slide', 'label_concordance', 'Energy_diff'])
    return leak_df

def plot_median_for_energy_diff(df, hue_order, x_grouping_column, x_order):
    median_values = df.groupby([x_grouping_column, 'label_concordance']).median().reset_index()
    n_hues = len(hue_order) # Number of hues
    width_per_group = 1
    for i, row in median_values.iterrows():
        x_group = row[x_grouping_column]
        label_concordance = row['label_concordance']
        median_value = row['Energy_diff_abs']

        if label_concordance in hue_order:

            x_index = x_order.index(x_group)
            base_x_position = x_index * width_per_group

            # Calculate the offset for the current hue within the group
            hue_index = hue_order.index(label_concordance)
            hue_offset = (hue_index - (n_hues - 1) / 2) * (width_per_group / n_hues)

            # Final x-position
            x_position = base_x_position + hue_offset
            y_position = -0.4

            # Place the text
            plt.text(x_position, y_position, f'{median_value:.2f}', ha='center', va='top', rotation=90)